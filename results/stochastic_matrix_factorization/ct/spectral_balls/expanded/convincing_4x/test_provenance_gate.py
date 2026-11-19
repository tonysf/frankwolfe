"""Small file-only tests of the independent gate and collector integration."""

from copy import deepcopy
import importlib.util
from itertools import product
from pathlib import Path
import tempfile

import audit_provenance as gate


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, continuation=False):
    (root / "runs").mkdir()
    for name in ("campaign.py", *gate.RUN_SOURCES, "confirm.py"):
        (root / name).write_text("# frozen " + name + "\n")
    (root / "audit_provenance.py").write_bytes((HERE / "audit_provenance.py").read_bytes())
    sources = {name: gate.digest(root / name) for name in ("campaign.py", *gate.RUN_SOURCES)}
    cohort = dict(evaluation_slices=["slice00100"], evaluation_training_views=[360],
                  evaluation_seeds=[202, 203, 204], deterministic_variants=["sframes_full", "palm"])
    protocol = dict(fixed_sframes=dict(beta_scale=16.0, gamma0=0.1), evaluation_budget=69120)
    selection = dict(status="selected", selected={
        "sframes": dict(beta_scale=16.0, gamma0=0.1), "sframes_full": dict(beta_scale=16.0, gamma0=0.1),
        "spring_sarah": dict(rate=1.6, refresh_period=16),
        "prox_sarah": dict(rate=2.3, inner_steps=12, snapshot_policy="full"), "palm": dict(rate=2.3)},
        grids={"spring_sarah": dict(rate=[1.6], refresh_period=[16]),
               "prox_sarah": dict(rate=[2.3], inner_steps=[12], snapshot_policy=["full"]), "palm": dict(rate=[2.3])})
    for name, payload in (("cohort.json", cohort), ("protocol_frozen.json", protocol),
                          ("selection.json", selection), ("baseline_selection.json", selection)):
        gate.write(root / name, payload)
    for name in gate.FROZEN_INPUTS:
        if not (root / name).exists():
            gate.write(root / name, {})
    environment = dict(numpy="2.5.4", astra="2.5.0", factor_dtype="float64",
                       projector="ASTRA CPU line_fanflat FP32", constraint_geometry="separate spectral norm balls and entrywise nonnegativity")
    anchor = dict(frozen_files={name: gate.digest(root / name) for name in gate.FROZEN_INPUTS},
                  sources=sources, data={"slice00100": "data"}, environment=environment, prior_run_metadata={})
    configs = []
    for variant in ("sframes", "sframes_full", "spring_sarah", "prox_sarah", "palm"):
        chosen = selection["selected"][variant]
        parameters = dict(chosen)
        if variant in ("sframes", "sframes_full"):
            parameters = dict(beta_scale=16.0, rate=0.1)
            label = "b16_g0p1"
        else:
            label = "a" + gate._number(parameters["rate"])
            if variant == "spring_sarah": label += "_q16"
            if variant == "prox_sarah": label += "_m12_sfull"
        seeds = [202] if variant in cohort["deterministic_variants"] else cohort["evaluation_seeds"]
        for seed in seeds:
            name = f"evaluation_slice00100_360_{variant}_seed{seed}_{label}"
            configs.append(dict(run_id=name, phase="evaluation", slice_id="slice00100", n_train=360,
                                method="sframes" if variant in ("sframes", "sframes_full") else variant,
                                variant=variant, trial=label, seed=seed, calibration_seed=201, grid=512, rank=64,
                                batch=360 if variant == "sframes_full" else 64, view_budget=69120,
                                record_every_iteration=True, **parameters))
    config = configs[0]
    origin = None
    if continuation:
        prior = root / "old_run"
        prior.mkdir()
        gate.write(prior / "metadata.json", {"old_wrapper_hash": "permitted old run.py"})
        config["continuation_directory"] = str(prior)
        anchor["prior_run_metadata"][str(prior)] = gate.digest(prior / "metadata.json")
        origin = dict(source_directory=str(prior),
                      source_sha256={name: sources[name] for name in gate.NUMERICAL} | {"run.py": "old-wrapper"})
    gate.write(root / "input_anchor.json", anchor)
    manifest = dict(sources=sources, data=anchor["data"], calibration={"slice00100": "calibration"},
                    selection_sha256=gate.digest(root / "selection.json"), cohort_sha256=gate.digest(root / "cohort.json"),
                    targets_sha256=gate.digest(root / "targets.json"), protocol_sha256=gate.digest(root / "protocol_frozen.json"))
    plan = dict(source_sha256=sources, selected_sha256=manifest["selection_sha256"], cohort_sha256=manifest["cohort_sha256"],
                expected_runs=len(configs), runs=configs)
    gate.write(root / "manifest.json", manifest)
    gate.write(root / "evaluation_plan.json", plan)
    evaluation = dict(input_anchor_sha256=gate.digest(root / "input_anchor.json"),
                      manifest_sha256=gate.digest(root / "manifest.json"), evaluation_plan_sha256=gate.digest(root / "evaluation_plan.json"),
                      selection_sha256=gate.digest(root / "selection.json"), baseline_selection_sha256=gate.digest(root / "baseline_selection.json"),
                      confirm_source_sha256=gate.digest(root / "confirm.py"),
                      calibration={"slice00100_seed201": "calibration"}, expected_runs=len(configs),
                      audit_helper_sha256=gate.digest(root / "audit_provenance.py"))
    gate.write(root / "evaluation_input_anchor.json", evaluation)
    meta = dict(config=config, curvature=[2.0, 3.0], data_sha256="data", calibration_sha256="calibration",
                source_sha256={name: sources[name] for name in gate.RUN_SOURCES}, continuation=origin, **environment)
    meta["optimizer"] = gate._optimizer(config, meta["curvature"])
    folder = root / "runs" / config["run_id"]
    folder.mkdir()
    gate.write(folder / "metadata.json", meta)
    return meta, folder


def rejected(call, label):
    try:
        call()
    except (ValueError, FileNotFoundError, KeyError):
        return label
    raise AssertionError("Gate accepted " + label)


def main():
    checked = []
    with tempfile.TemporaryDirectory(prefix="ct-provenance-fixture-") as temporary:
        root = Path(temporary)
        meta, folder = fixture(root)
        gate.validate_evaluation_metadata(root, meta)
        collector = load("ct_provenance_fixture_collector", HERE / "analyze.py")
        assert len(collector.collect(root)) == 1
        checked.append("valid public per-arm gate and native collector integration")
        altered = deepcopy(meta)
        altered["data_sha256"] = "changed in both result and metadata"
        gate.write(folder / "metadata.json", altered)
        checked.append(rejected(lambda: collector.collect(root), "self-consistent changed metadata data hash"))
        gate.write(folder / "metadata.json", meta)
        altered = deepcopy(meta)
        altered["source_sha256"]["methods.py"] = "changed"
        checked.append(rejected(lambda: gate.validate_evaluation_metadata(root, altered), "changed recorded helper source"))
        source = root / "methods.py"
        source.write_text("# changed current code\n")
        checked.append(rejected(lambda: gate.validate_evaluation_metadata(root, meta), "changed current helper source"))
    with tempfile.TemporaryDirectory(prefix="ct-prefix-fixture-") as temporary:
        root = Path(temporary)
        meta, _ = fixture(root, continuation=True)
        gate.validate_evaluation_metadata(root, meta)
        checked.append("old prefix wrapper accepted with unchanged numerical helpers")
        altered = deepcopy(meta)
        altered["continuation"]["source_sha256"]["methods.py"] = "changed"
        checked.append(rejected(lambda: gate.validate_evaluation_metadata(root, altered), "changed old prefix numerical helper"))
    print(gate.self_test())
    print({"file_only_integration_checks_passed": checked})


if __name__ == "__main__":
    main()
