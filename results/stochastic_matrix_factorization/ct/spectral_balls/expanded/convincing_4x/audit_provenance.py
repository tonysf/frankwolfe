"""Independent, dependency-free input and metadata gates for the frozen campaign.

The numerical runner and development orchestrator remain unchanged.  This
module anchors their records to inputs frozen before evaluation, rather than
accepting mutually consistent result/metadata files as sufficient evidence.
"""

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
NUMERICAL = ("ct_problem.py", "methods.py", "spectral_reference_core.py", "spectral_reference_baseline.py")
RUN_SOURCES = ("run.py",) + NUMERICAL
ENVIRONMENT = ("numpy", "astra", "factor_dtype", "projector", "constraint_geometry")
FROZEN_INPUTS = ("development_manifest.json", "protocol_frozen.json", "cohort.json", "targets.json",
                 "data_group_manifest.json", "data_preparation_manifest.json", "data_validation.json")
_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def require_equal(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def _partition(mapping):
    return {frozenset(sid for sid, group in mapping.items() if group == value)
            for value in set(mapping.values())}


def _headers(root, anchor):
    for name, expected in anchor["frozen_files"].items():
        require_equal(digest(root / name), expected, f"Frozen input changed: {name}")
    for name, expected in anchor["sources"].items():
        require_equal(digest(root / name), expected, f"Frozen development source changed: {name}")


def _old_metadata(root):
    prior = root.parent / "joint_tuning" / "runs"
    return {str(path.parent.resolve()): (path.resolve(), read(path))
            for path in sorted(prior.glob("*/metadata.json"))}


def freeze_input_anchor(root=ROOT):
    """Freeze validated preparation inputs, immutable old calibrations and smoke environment."""
    root = Path(root).resolve()
    output = root / "input_anchor.json"
    if output.exists():
        anchor = read(output)
        _headers(root, anchor)
        return anchor
    development, protocol = read(root / "development_manifest.json"), read(root / "protocol_frozen.json")
    cohort, groups = read(root / "cohort.json"), read(root / "data_group_manifest.json")
    prepared, validated = read(root / "data_preparation_manifest.json"), read(root / "data_validation.json")
    require_equal(prepared["status"], "complete", "Data preparation is incomplete")
    require_equal(validated["status"], "validated", "Data preprocessing has not been validated")
    require_equal(validated["preparation_manifest_sha256"], digest(root / "data_preparation_manifest.json"),
                  "Prepared inputs differ from the validated preprocessing manifest")
    require_equal(validated["data_group_manifest_sha256"], digest(root / "data_group_manifest.json"),
                  "The validated specimen groups changed")
    require_equal(_partition(cohort["evaluation_slice_groups"]), _partition(groups["evaluation_slice_groups"]),
                  "Cohort and preprocessing group partitions disagree")
    for filename, field in (("cohort.json", "cohort_sha256"), ("targets.json", "targets_sha256")):
        require_equal(protocol[field], digest(root / filename), f"Protocol does not anchor {filename}")
    sources = development["sources"]
    smoke = read(root / "smoke_runs" / "engine_validation.json")
    require_equal(smoke["source_sha256"], sources, "The production sources differ from the final smoke test")
    current_smokes = [(path, read(path)) for path in sorted((root / "smoke_runs").glob("*/metadata.json"))]
    current_smokes = [(path, meta) for path, meta in current_smokes
                      if all(meta["source_sha256"].get(name) == sources[name] for name in RUN_SOURCES)]
    if not current_smokes:
        raise ValueError("No smoke metadata matches the frozen production sources")
    environment = {key: current_smokes[0][1][key] for key in ENVIRONMENT}
    for _, meta in current_smokes:
        require_equal({key: meta[key] for key in ENVIRONMENT}, environment, "Smoke environments differ")
    prior = _old_metadata(root)
    calibrations, witnesses = {}, {}
    for sid in development["design"]["development_slices"]:
        seed = development["design"]["development_calibration_seed"]
        matched = [(path, meta) for path, meta in prior.values()
                   if meta["config"]["slice_id"] == sid and meta["config"]["calibration_seed"] == seed]
        expected = {meta["calibration_sha256"] for _, meta in matched}
        if len(expected) != 1:
            raise ValueError(f"Original development calibration hashes disagree for {sid}")
        name = f"{sid}_seed{seed}"
        calibrations[name] = expected.pop()
        require_equal(digest(root / "calibration" / f"{name}.npz"), calibrations[name],
                      f"Original development calibration changed: {sid}")
        witness = matched[0][0]
        witnesses[name] = dict(metadata=str(witness), metadata_sha256=digest(witness))
    records = prepared["records"]
    for sid, expected in development["data"].items():
        require_equal(records[sid]["sha256"], expected, f"Development/preparation data disagree for {sid}")
    anchor = dict(
        version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
        frozen_files={name: digest(root / name) for name in FROZEN_INPUTS}, sources=sources,
        data={sid: row["sha256"] for sid, row in records.items()},
        data_sidecars={sid: row["metadata_sha256"] for sid, row in records.items()},
        development_calibration=calibrations, calibration_witnesses=witnesses,
        environment=environment,
        smoke_metadata={str(path): digest(path) for path, _ in current_smokes},
        prior_run_metadata={folder: digest(path) for folder, (path, _) in prior.items()},
        purpose="Freeze validated data and final smoke sources/environment independently of per-run self-reported metadata.",
    )
    _headers(root, anchor)
    for sid, expected in anchor["data"].items():
        require_equal(digest(root / "data" / f"{sid}.npz"), expected, f"Prepared data changed: {sid}")
        require_equal(digest(root / "data" / f"{sid}.json"), anchor["data_sidecars"][sid],
                      f"Prepared data sidecar changed: {sid}")
    write(output, anchor)
    return anchor


def _optimizer(config, curvature):
    u, v = curvature
    rate, batch, method = config["rate"], config["batch"], config["method"]
    if method == "sframes":
        return dict(batch=batch, gamma0=rate, rho0=1.0, mu0=config.get("beta_scale", 1.0) / max(u, v))
    if method == "palm":
        return dict(eta_u=rate/u, eta_v=rate/v)
    if method == "spring_sarah":
        return dict(batch=batch, eta_u=rate/u, eta_v=rate/v, refresh_period=config["refresh_period"])
    if method != "prox_sarah":
        raise ValueError("Unexpected optimizer in provenance gate")
    policy = config.get("snapshot_policy", "minibatch")
    if policy not in ("minibatch", "full"):
        raise ValueError("Unexpected ProxSARAH snapshot policy")
    snapshot = config["n_train"] if policy == "full" else min(config["n_train"], 4*batch)
    return dict(batch=batch, eta=rate/(u+v), relaxation=0.8, inner_steps=config["inner_steps"], snapshot_batch=snapshot)


def _assert_metadata(meta, config, anchor, calibration):
    require_equal(meta["config"], config, "Run configuration differs from its frozen plan")
    sid = config["slice_id"]
    require_equal(meta["source_sha256"], {name: anchor["sources"][name] for name in RUN_SOURCES},
                  "Run source hashes differ from the frozen production sources")
    require_equal(meta["data_sha256"], anchor["data"][sid], "Run data differs from the validated input anchor")
    require_equal(meta["calibration_sha256"], calibration[f"{sid}_seed{config['calibration_seed']}"],
                  "Run calibration differs from its frozen input anchor")
    require_equal({key: meta[key] for key in ENVIRONMENT}, anchor["environment"],
                  "Run environment differs from the final production smoke environment")
    require_equal(meta["optimizer"], _optimizer(config, meta["curvature"]),
                  "Actual optimizer differs from the frozen parameter configuration")


def _number(value):
    return f"{value:.12g}".replace(".", "p")


def _development_configs(root, anchor):
    development = read(root / "development_manifest.json")["design"]
    selected = read(root / "selection.json") if (root / "selection.json").exists() else None
    grids = selected["grids"] if selected and selected["status"] == "selected" else read(root / "search_state.json")["grids"]
    configs = {}
    for variant, grid in grids.items():
        for values in product(*(grid[name] for name in grid)):
            parameters = dict(zip(grid, values))
            label = "a" + _number(parameters["rate"])
            if variant == "spring_sarah":
                label += f"_q{parameters['refresh_period']}"
            if variant == "prox_sarah":
                label += f"_m{parameters['inner_steps']}_s{parameters['snapshot_policy']}"
            seeds = development["development_seeds"][:1] if variant == "palm" else development["development_seeds"]
            for sid, n, seed in product(development["development_slices"], development["development_training_views"], seeds):
                run_id = f"development_{sid}_{n}_{variant}_seed{seed}_{label}"
                configs[run_id] = dict(run_id=run_id, phase="development", slice_id=sid, n_train=n,
                                      method=variant, variant=variant, trial=label, seed=seed,
                                      calibration_seed=development["development_calibration_seed"],
                                      grid=512, rank=64, batch=64, view_budget=development["development_view_budget"],
                                      record_every_iteration=False, **parameters)
    return configs


def audit_development(root=ROOT, *, require_complete=False, verify_files=True):
    root = Path(root).resolve()
    anchor = freeze_input_anchor(root)
    _headers(root, anchor)
    if verify_files:
        for sid, expected in anchor["data"].items():
            require_equal(digest(root / "data" / f"{sid}.npz"), expected, f"Prepared data changed: {sid}")
            require_equal(digest(root / "data" / f"{sid}.json"), anchor["data_sidecars"][sid],
                          f"Prepared data sidecar changed: {sid}")
        for name, expected in anchor["development_calibration"].items():
            require_equal(digest(root / "calibration" / f"{name}.npz"), expected,
                          f"Original development calibration changed: {name}")
    expected = _development_configs(root, anchor)
    statuses, observed = {}, 0
    for path in sorted((root / "runs").glob("development_*/metadata.json")):
        meta = read(path)
        name = meta["config"]["run_id"]
        if name not in expected:
            raise ValueError(f"Development arm is outside the frozen full grid: {name}")
        _assert_metadata(meta, expected[name], anchor, anchor["development_calibration"])
        observed += 1
        result_path = path.parent / "result.json"
        if result_path.exists():
            result = read(result_path)
            _assert_metadata(result, expected[name], anchor, anchor["development_calibration"])
            require_equal({key: result[key] for key in meta}, meta, "Development result and metadata disagree")
            if result["status"] not in ("completed", "candidate_failed"):
                raise ValueError(f"Development arm failed operationally: {name}")
            statuses[name] = result["status"]
    if require_complete and (set(statuses) != set(expected)):
        raise ValueError("Development audit requires every full-grid candidate to be completed or explicitly excluded")
    result = dict(status="validated", expected_candidates=len(expected), registered_candidates=observed,
                  completed=sum(status == "completed" for status in statuses.values()),
                  excluded_candidates=sum(status == "candidate_failed" for status in statuses.values()),
                  input_anchor_sha256=digest(root / "input_anchor.json"), all_sources_match_frozen_manifest=True)
    write(root / "development_provenance_audit.json", result)
    return result


def _npz_scalar_string(archive, key):
    with archive.open(key + ".npy") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("Calibration string has an invalid NPY header")
        major, _ = stream.read(2)
        count = 2 if major == 1 else 4
        length = int.from_bytes(stream.read(count), "little")
        header = ast.literal_eval(stream.read(length).decode("latin1"))
        if header["shape"] != () or not header["descr"].startswith("<U"):
            raise ValueError("Calibration provenance must be a scalar Unicode NPY value")
        return stream.read().decode("utf-32-le").rstrip("\x00")


def _assert_evaluation_plan(root, manifest, plan):
    cohort, selection = read(root / "cohort.json"), read(root / "selection.json")
    protocol = read(root / "protocol_frozen.json")
    require_equal(selection["status"], "selected", "Evaluation uses an unfinished development selection")
    require_equal(plan["selected_sha256"], digest(root / "selection.json"), "Evaluation plan uses a different selection")
    require_equal(plan["cohort_sha256"], digest(root / "cohort.json"), "Evaluation plan uses a different cohort")
    require_equal(manifest["selection_sha256"], digest(root / "selection.json"), "Evaluation manifest selection changed")
    require_equal(manifest["cohort_sha256"], digest(root / "cohort.json"), "Evaluation manifest cohort changed")
    require_equal(manifest["targets_sha256"], digest(root / "targets.json"), "Evaluation target definitions changed")
    require_equal(manifest["protocol_sha256"], digest(root / "protocol_frozen.json"), "Evaluation protocol changed")
    variants = ("sframes", "sframes_full", "spring_sarah", "prox_sarah", "palm")
    expected = {}
    for sid, n, variant in product(cohort["evaluation_slices"], cohort["evaluation_training_views"], variants):
        chosen = selection["selected"][variant]
        seeds = cohort["evaluation_seeds"][:1] if variant in cohort["deterministic_variants"] else cohort["evaluation_seeds"]
        if variant in ("sframes", "sframes_full"):
            parameters = dict(beta_scale=float(chosen["beta_scale"]), rate=float(chosen["gamma0"]))
            require_equal(parameters, dict(beta_scale=protocol["fixed_sframes"]["beta_scale"],
                                            rate=protocol["fixed_sframes"]["gamma0"]),
                          "SFRAMES settings differ from the frozen previous joint selection")
            label = f"b{_number(parameters['beta_scale'])}_g{_number(parameters['rate'])}"
        else:
            parameters = {name: chosen[name] for name in selection["grids"][variant]}
            label = "a" + _number(parameters["rate"])
            if variant == "spring_sarah":
                label += f"_q{parameters['refresh_period']}"
            if variant == "prox_sarah":
                label += f"_m{parameters['inner_steps']}_s{parameters['snapshot_policy']}"
        for seed in seeds:
            name = f"evaluation_{sid}_{n}_{variant}_seed{seed}_{label}"
            expected[name] = dict(run_id=name, phase="evaluation", slice_id=sid, n_train=n,
                                  method="sframes" if variant in ("sframes", "sframes_full") else variant,
                                  variant=variant, trial=label, seed=seed, calibration_seed=201, grid=512, rank=64,
                                  batch=n if variant == "sframes_full" else 64,
                                  view_budget=protocol["evaluation_budget"], record_every_iteration=True, **parameters)
    configs = {config["run_id"]: config for config in plan["runs"]}
    require_equal(len(configs), len(plan["runs"]), "Evaluation plan repeats an arm")
    require_equal(set(configs), set(expected), "Evaluation plan omits or adds a frozen scientific arm")
    require_equal(plan["expected_runs"], len(expected), "Evaluation plan count does not match its frozen cohort")
    for name, config in configs.items():
        actual = {key: value for key, value in config.items() if key != "continuation_directory"}
        require_equal(actual, expected[name], f"Evaluation arm differs from frozen selection/protocol: {name}")


def freeze_evaluation_anchor(root=ROOT):
    root = Path(root).resolve()
    anchor = freeze_input_anchor(root)
    _headers(root, anchor)
    manifest, plan = read(root / "manifest.json"), read(root / "evaluation_plan.json")
    require_equal(manifest["sources"], anchor["sources"], "Evaluation sources differ from frozen development")
    require_equal(plan["source_sha256"], anchor["sources"], "Evaluation plan sources differ from frozen development")
    _assert_evaluation_plan(root, manifest, plan)
    require_equal(set(manifest["data"]), set(anchor["data"]), "Evaluation manifest omits or adds frozen datasets")
    require_equal(set(manifest["data_sidecars"]), set(anchor["data_sidecars"]), "Evaluation preprocessing sidecars are incomplete")
    for sid, expected in manifest["data"].items():
        require_equal(expected, anchor["data"][sid], "Evaluation manifest data differs from validated preparation")
    require_equal(manifest["data_sidecars"], {sid: anchor["data_sidecars"][sid] for sid in manifest["data_sidecars"]},
                  "Evaluation preprocessing sidecar hashes changed")
    calibration = {}
    prior = _old_metadata(root)
    for sid, expected in manifest["calibration"].items():
        name = f"{sid}_seed201"
        path = root / "calibration" / (name + ".npz")
        require_equal(digest(path), expected, f"Prepared evaluation calibration changed: {sid}")
        with zipfile.ZipFile(path) as archive:
            require_equal(_npz_scalar_string(archive, "data_sha256"), anchor["data"][sid],
                          f"Evaluation calibration has different input data: {sid}")
            require_equal(_npz_scalar_string(archive, "problem_sha256"), anchor["sources"]["ct_problem.py"],
                          f"Evaluation calibration has different projector code: {sid}")
        old_hashes = {meta["calibration_sha256"] for _, meta in prior.values()
                      if meta["config"]["phase"] == "evaluation" and meta["config"]["slice_id"] == sid}
        if old_hashes and old_hashes != {expected}:
            raise ValueError(f"An original evaluation calibration was not preserved: {sid}")
        calibration[name] = expected
    value = dict(input_anchor_sha256=digest(root / "input_anchor.json"),
                 manifest_sha256=digest(root / "manifest.json"), evaluation_plan_sha256=digest(root / "evaluation_plan.json"),
                 selection_sha256=digest(root / "selection.json"), baseline_selection_sha256=digest(root / "baseline_selection.json"),
                 confirm_source_sha256=digest(root / "confirm.py"), calibration=calibration,
                 expected_runs=plan["expected_runs"], audit_helper_sha256=digest(Path(__file__)))
    output = root / "evaluation_input_anchor.json"
    if output.exists():
        require_equal(read(output), value, "Frozen evaluation input anchor changed")
    else:
        write(output, value)
    _CACHE.clear()
    return value


def _evaluation_context(root):
    names = ("input_anchor.json", "evaluation_input_anchor.json", "manifest.json", "evaluation_plan.json",
             "selection.json", "baseline_selection.json", "confirm.py", "audit_provenance.py",
             "campaign.py", *RUN_SOURCES, *FROZEN_INPUTS)
    stamp = tuple((name, (root / name).stat().st_mtime_ns, (root / name).stat().st_size) for name in names)
    cache_key = str(root), stamp
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    anchor, evaluation = read(root / "input_anchor.json"), read(root / "evaluation_input_anchor.json")
    _headers(root, anchor)
    require_equal(digest(root / "audit_provenance.py"), evaluation["audit_helper_sha256"],
                  "The independent evaluation provenance gate changed")
    for filename, field in (("input_anchor.json", "input_anchor_sha256"), ("manifest.json", "manifest_sha256"),
                            ("evaluation_plan.json", "evaluation_plan_sha256"), ("selection.json", "selection_sha256"),
                            ("baseline_selection.json", "baseline_selection_sha256"), ("confirm.py", "confirm_source_sha256")):
        require_equal(digest(root / filename), evaluation[field], f"Frozen evaluation input changed: {filename}")
    manifest, plan = read(root / "manifest.json"), read(root / "evaluation_plan.json")
    require_equal(manifest["sources"], anchor["sources"], "Evaluation manifest source anchor mismatch")
    require_equal(plan["source_sha256"], anchor["sources"], "Evaluation plan source anchor mismatch")
    _assert_evaluation_plan(root, manifest, plan)
    configs = {row["run_id"]: row for row in plan["runs"]}
    if len(configs) != plan["expected_runs"] or plan["expected_runs"] != evaluation["expected_runs"]:
        raise ValueError("Frozen evaluation plan has duplicate or missing configurations")
    context = anchor, evaluation, manifest, configs
    _CACHE.clear()
    _CACHE[cache_key] = context
    return context


def validate_evaluation_metadata(root, meta, *, expected_config=None):
    """Lightweight per-arm gate; caches the immutable parsed input context."""
    root = Path(root).resolve()
    anchor, evaluation, manifest, configs = _evaluation_context(root)
    name = meta["config"]["run_id"]
    if name not in configs:
        raise ValueError(f"Evaluation arm is outside the frozen plan: {name}")
    config = configs[name]
    if expected_config is not None:
        require_equal(config, expected_config, "Requested evaluation configuration differs from its frozen plan")
    _assert_metadata(meta, config, anchor, evaluation["calibration"])
    sid = config["slice_id"]
    require_equal(meta["data_sha256"], manifest["data"][sid], "Evaluation run differs from frozen manifest data")
    require_equal(meta["calibration_sha256"], manifest["calibration"][sid], "Evaluation run differs from frozen manifest calibration")
    origin = meta.get("continuation")
    if config.get("continuation_directory"):
        if not origin:
            raise ValueError("An expected continuation has no recorded prefix provenance")
        folder = str(Path(config["continuation_directory"]).resolve())
        require_equal(str(Path(origin["source_directory"]).resolve()), folder, "Continuation source differs from its frozen plan")
        if folder not in anchor["prior_run_metadata"]:
            raise ValueError("Continuation was not present in the immutable original cohort")
        require_equal(digest(Path(folder) / "metadata.json"), anchor["prior_run_metadata"][folder],
                      "Original continuation metadata changed")
        for filename in NUMERICAL:
            require_equal(origin["source_sha256"][filename], anchor["sources"][filename],
                          f"Continuation changes numerical helper {filename}")
    elif origin is not None:
        raise ValueError("Unexpected continuation outside the frozen evaluation plan")
    return True


def audit_evaluation(root=ROOT, *, require_complete=False):
    root = Path(root).resolve()
    _, evaluation, _, configs = _evaluation_context(root)
    present, complete = set(), set()
    for path in sorted((root / "runs").glob("evaluation_*/metadata.json")):
        metadata = read(path)
        validate_evaluation_metadata(root, metadata)
        name = metadata["config"]["run_id"]
        if name in present:
            raise ValueError("Duplicate evaluation run registration")
        present.add(name)
        result_path = path.parent / "result.json"
        if result_path.exists():
            result = read(result_path)
            validate_evaluation_metadata(root, result)
            require_equal({key: result[key] for key in metadata}, metadata, "Evaluation result and metadata disagree")
            require_equal(result["status"], "completed", "Evaluation result is not complete")
            complete.add(name)
    if require_complete and complete != set(configs):
        raise ValueError("Evaluation provenance audit requires the complete frozen cohort")
    result = dict(status="validated", expected_runs=len(configs), registered_runs=len(present), completed_runs=len(complete),
                  input_anchor_sha256=evaluation["input_anchor_sha256"], evaluation_input_anchor_sha256=digest(root / "evaluation_input_anchor.json"),
                  manifest_sha256=evaluation["manifest_sha256"], audit_helper_sha256=digest(Path(__file__)))
    write(root / "evaluation_provenance_audit.json", result)
    return result


def self_test():
    """Exercise source/data/calibration/environment/config/snapshot tamper rejection without numerics."""
    sources = {name: f"hash-{name}" for name in ("campaign.py", *RUN_SOURCES)}
    environment = dict(numpy="2.5.4", astra="2.5.0", factor_dtype="float64",
                       projector="ASTRA CPU line_fanflat FP32", constraint_geometry="spectral")
    anchor = dict(sources=sources, data={"slice1": "data"}, environment=environment)
    calibration = {"slice1_seed201": "calibration"}
    config = dict(run_id="test", slice_id="slice1", calibration_seed=201, n_train=1080,
                  method="prox_sarah", rate=2.3, batch=64, inner_steps=12, snapshot_policy="full")
    meta = dict(config=config, curvature=[2.0, 3.0], data_sha256="data", calibration_sha256="calibration",
                source_sha256={name: sources[name] for name in RUN_SOURCES}, **environment)
    meta["optimizer"] = _optimizer(config, meta["curvature"])
    _assert_metadata(meta, config, anchor, calibration)
    mutations = {
        "source": lambda m: m["source_sha256"].update({"methods.py": "changed"}),
        "data": lambda m: m.update(data_sha256="changed"),
        "calibration": lambda m: m.update(calibration_sha256="changed"),
        "environment": lambda m: m.update(astra="changed"),
        "configuration": lambda m: m["config"].update(rate=5.0),
        "snapshot": lambda m: m["optimizer"].update(snapshot_batch=256),
    }
    rejected = []
    for name, mutate in mutations.items():
        trial = deepcopy(meta)
        mutate(trial)
        try:
            _assert_metadata(trial, config, anchor, calibration)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"The independent gate accepted a changed {name}")
    return dict(valid_metadata_accepted=True, rejected_mutations=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("anchor", "development", "evaluation"), default="development")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        value = self_test()
    elif args.stage == "anchor":
        value = freeze_input_anchor(args.root)
    elif args.stage == "development":
        value = audit_development(args.root, require_complete=args.require_complete)
    else:
        value = audit_evaluation(args.root, require_complete=args.require_complete)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
