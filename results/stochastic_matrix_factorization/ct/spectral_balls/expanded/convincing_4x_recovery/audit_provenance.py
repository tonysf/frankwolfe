"""Audit a narrow, versioned recovery without changing historical artifacts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "convincing_4x"
NUMERICAL = ("ct_problem.py", "methods.py", "spectral_reference_core.py",
             "spectral_reference_baseline.py")
RUN_SOURCES = ("run.py",) + NUMERICAL
ENVIRONMENT = ("numpy", "astra", "factor_dtype", "projector", "constraint_geometry")
OLD_LINE = "        scale = min(1.0, radius / max(opnorm, np.finfo(float).tiny))"
NEW_LINE = "        scale = radius / opnorm if opnorm > radius else 1.0"
PRESERVED = ("cohort.json", "targets.json", "protocol_frozen.json", "selection.json",
             "baseline_selection.json", "sframes_selection.json", "development_manifest.json",
             "input_anchor.json", "data_group_manifest.json", "data_preparation_manifest.json",
             "data_validation.json")
_CACHE = {}
_OLD_AUDITOR = None


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


def old_auditor():
    global _OLD_AUDITOR
    if _OLD_AUDITOR is not None:
        return _OLD_AUDITOR
    spec = importlib.util.spec_from_file_location("historical_ct_provenance", ORIGINAL / "audit_provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _OLD_AUDITOR = module
    return _OLD_AUDITOR


def audit_historical_development(root=ROOT, *, verify_files=True):
    """Use the original frozen gate, redirecting only its output audit file."""
    root = Path(root).resolve()
    module = old_auditor()
    module.write = lambda path, value: write(root / "historical_development_provenance_audit.json", value)
    return module.audit_development(ORIGINAL, require_complete=True, verify_files=verify_files)


def patch_proof(root):
    """Verify the exact one-line change rather than trusting a declared hash alias."""
    old_path = ORIGINAL / "spectral_reference_baseline.py"
    new_path = root / "spectral_reference_baseline.py"
    old_text, new_text = old_path.read_text(), new_path.read_text()
    if old_text.count(OLD_LINE) != 1 or new_text != old_text.replace(OLD_LINE, NEW_LINE, 1):
        raise ValueError("Recovery helper differs from the narrowly approved division guard")
    for name in NUMERICAL:
        if name != "spectral_reference_baseline.py":
            require_equal(digest(root / name), digest(ORIGINAL / name),
                          f"Recovery changes an unrelated numerical helper: {name}")
    replay = read(root / "guard_replay_validation.json")
    for field in ("fixed_strict_step_passed", "old_permissive_and_fixed_strict_solver_state_bitwise_equal",
                  "old_permissive_and_fixed_strict_work_and_certificate_equal", "all_factors_finite"):
        require_equal(replay[field], True, f"The failed-step replay did not validate {field}")
    require_equal(replay["fixed_helper_sha256"], digest(new_path), "The replay tested a different helper")
    return {"old_helper_sha256": digest(old_path), "new_helper_sha256": digest(new_path),
            "unchanged_numerical_sha256": {name: digest(root / name) for name in NUMERICAL
                                           if name != "spectral_reference_baseline.py"},
            "replay_sha256": digest(root / "guard_replay_validation.json")}


def canonical_source_hash(root, name, value):
    root = Path(root).resolve()
    old = digest(ORIGINAL / name)
    new = digest(root / name)
    if name == "spectral_reference_baseline.py":
        patch_proof(root)
        if value not in (old, new):
            raise ValueError("An undeclared projection helper revision was used")
        return old
    if value != old or new != old:
        raise ValueError(f"An unrelated numerical source changed: {name}")
    return old


def _scientific(config):
    return {key: value for key, value in config.items()
            if key not in ("continuation_directory", "checkpoint_import_directory")}


def _headers(root, anchor):
    for name, expected in anchor["frozen_files"].items():
        require_equal(digest(root / name), expected, f"Recovery input changed: {name}")
    for name, expected in anchor["historical_files"].items():
        require_equal(digest(ORIGINAL / name), expected, f"Historical input changed: {name}")
    for name, expected in anchor["sources"].items():
        require_equal(digest(root / name), expected, f"Recovery source changed: {name}")
    patch_proof(root)


def _config_plan(root):
    old_plan, plan = read(ORIGINAL / "evaluation_plan.json"), read(root / "evaluation_plan.json")
    old_configs = {row["run_id"]: row for row in old_plan["runs"]}
    configs = {row["run_id"]: row for row in plan["runs"]}
    require_equal(len(configs), len(plan["runs"]), "Recovery plan repeats an arm")
    require_equal(set(configs), set(old_configs), "Recovery plan changes the 308-arm cohort")
    require_equal(len(configs), 308, "Recovery requires the complete original evaluation cohort")
    origins = plan["origins"]
    require_equal(set(origins), set(configs), "Recovery origin map omits or adds an arm")
    for name, config in configs.items():
        require_equal(_scientific(config), _scientific(old_configs[name]),
                      f"Recovery changes the selected scientific parameters: {name}")
        kind = origins[name]["kind"]
        if kind not in ("reuse_completed", "import_checkpoint", "continue_original_prefix", "fresh"):
            raise ValueError(f"Unknown recovery origin for {name}")
        if kind == "reuse_completed":
            require_equal(config, old_configs[name], "A reused completed arm was relabeled")
        elif kind == "import_checkpoint":
            require_equal(str(Path(config["checkpoint_import_directory"]).resolve()),
                          str(Path(origins[name]["source_directory"]).resolve()),
                          "Checkpoint source differs from its frozen origin")
        elif "checkpoint_import_directory" in config:
            raise ValueError("An unplanned checkpoint import was configured")
    return plan, configs, old_configs


def freeze_recovery_anchor(root=ROOT):
    root = Path(root).resolve()
    output = root / "recovery_input_anchor.json"
    if output.exists():
        anchor = read(output)
        _headers(root, anchor)
        return anchor
    for name in PRESERVED:
        if not (root / name).exists():
            shutil.copyfile(ORIGINAL / name, root / name)
        require_equal(digest(root / name), digest(ORIGINAL / name), f"A frozen input was relabeled: {name}")
    old_audit = audit_historical_development(root)
    proof = patch_proof(root)
    compatibility = read(root / "compatibility_manifest.json")
    require_equal(compatibility["old_helper_sha256"], proof["old_helper_sha256"], "Wrong historical helper in compatibility record")
    require_equal(compatibility["new_helper_sha256"], proof["new_helper_sha256"], "Wrong recovered helper in compatibility record")
    manifest = read(root / "manifest.json")
    plan, configs, old_configs = _config_plan(root)
    original_manifest = read(ORIGINAL / "manifest.json")
    for field in ("data", "data_sidecars", "calibration", "design", "cohort_sha256", "targets_sha256",
                  "protocol_sha256", "selection_sha256", "baseline_selection_sha256"):
        require_equal(manifest[field], original_manifest[field], f"Recovery changes frozen {field}")
    require_equal(manifest["sources"], plan["source_sha256"], "Recovery runtime source declarations disagree")
    for name, expected in manifest["sources"].items():
        require_equal(digest(root / name), expected, f"Recovery manifest records a different source: {name}")
    require_equal(plan["selected_sha256"], digest(root / "selection.json"), "Recovery selection changed")
    require_equal(plan["cohort_sha256"], digest(root / "cohort.json"), "Recovery specimen cohort changed")
    frozen = {name: digest(root / name) for name in (*PRESERVED, "manifest.json", "evaluation_plan.json",
                                                 "compatibility_manifest.json", "guard_replay_validation.json")}
    sources = dict(manifest["sources"])
    for name in ("recovery_confirm.py", "audit_provenance.py"):
        sources[name] = digest(root / name)
    for optional in ("compatibility.py", "recovery_compatibility.py"):
        if (root / optional).is_file():
            sources[optional] = digest(root / optional)
    source_files = {}
    for name, origin in plan["origins"].items():
        kind = origin["kind"]
        if kind == "fresh":
            continue
        folder = Path(origin["source_directory"]).resolve()
        if kind in ("reuse_completed", "import_checkpoint"):
            require_equal(folder, (ORIGINAL / "runs" / name).resolve(), "A recovery source is outside its original arm")
        elif str(folder) != str(Path(old_configs[name]["continuation_directory"]).resolve()):
            raise ValueError("An original continuation source was changed")
        files = {}
        for filename in ("metadata.json", "trace.json", "checkpoint.pkl", "result.json"):
            path = folder / filename
            if path.exists():
                files[filename] = digest(path)
        if kind in ("reuse_completed", "import_checkpoint"):
            old_auditor().validate_evaluation_metadata(ORIGINAL, read(folder / "metadata.json"),
                                                     expected_config=old_configs[name])
        if kind == "reuse_completed":
            require_equal(read(folder / "result.json")["status"], "completed", "A reused arm is not complete")
        source_files[name] = dict(source_directory=str(folder), hashes=files)
    value = dict(version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
                 frozen_files=frozen, historical_files={name: digest(ORIGINAL / name) for name in
                                                      (*PRESERVED, "manifest.json", "evaluation_plan.json", "audit_provenance.py")},
                 sources=sources, origins=source_files, expected_runs=308,
                 environment=read(ORIGINAL / "input_anchor.json")["environment"],
                 historical_development=old_audit, proof=proof)
    _headers(root, value)
    for sid, expected in manifest["data"].items():
        require_equal(digest(root / "data" / f"{sid}.npz"), expected, f"Recovery dataset changed: {sid}")
    for sid, expected in manifest["calibration"].items():
        require_equal(digest(root / "calibration" / f"{sid}_seed201.npz"), expected,
                      f"Recovery calibration changed: {sid}")
    write(output, value)
    _CACHE.clear()
    return value


def _context(root):
    root = Path(root).resolve()
    anchor_path = root / "recovery_input_anchor.json"
    anchor = read(anchor_path)
    names = (*anchor["frozen_files"], *anchor["sources"], "recovery_input_anchor.json")
    stamp = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                  for path in (root / name for name in names))
    key = str(root), stamp
    if key not in _CACHE:
        _headers(root, anchor)
        manifest = read(root / "manifest.json")
        plan, configs, old_configs = _config_plan(root)
        _CACHE.clear()
        _CACHE[key] = anchor, manifest, plan, configs, old_configs
    return _CACHE[key]


def _source_audit(root, name, record, anchor):
    pinned = anchor["origins"][name]
    folder = Path(record["source_directory"]).resolve()
    require_equal(str(folder), pinned["source_directory"], "Imported source differs from frozen recovery origin")
    fields = (("metadata.json", "source_metadata_sha256"), ("trace.json", "source_trace_sha256"),
              ("checkpoint.pkl", "source_checkpoint_sha256"))
    for filename, field in fields:
        require_equal(digest(folder / filename), pinned["hashes"][filename], "An imported source file changed")
        require_equal(record[field], pinned["hashes"][filename], "Imported source hash was relabeled")
    source_meta = read(folder / "metadata.json")
    source_trace = read(folder / "trace.json")
    require_equal(record["source_sha256"], source_meta["source_sha256"],
                  "Imported helper history was relabeled")
    require_equal(record["source_config"], source_meta["config"], "Imported configuration was relabeled")
    require_equal(record["source_final_iteration"], source_trace[-1]["iteration"], "Imported iteration was relabeled")
    require_equal(record["source_prefix_observations"], len(source_trace), "Imported prefix length changed")
    require_equal(record["source_forward_views"], source_trace[-1]["forward_views"], "Imported work count changed")
    require_equal(record["preserves_optimizer_state_and_rng"], True, "Checkpoint import did not preserve optimizer/RNG state")


def validate_evaluation_metadata(root, meta, *, expected_config=None):
    root = Path(root).resolve()
    anchor, manifest, plan, configs, old_configs = _context(root)
    name = meta["config"]["run_id"]
    if name not in configs:
        raise ValueError("An evaluation arm is outside the frozen recovery cohort")
    config, origin = configs[name], plan["origins"][name]
    require_equal(meta["config"], config, "Recovery arm differs from its frozen plan")
    if expected_config is not None:
        require_equal(config, expected_config, "Requested arm differs from the frozen recovery plan")
    kind = origin["kind"]
    if kind == "reuse_completed":
        folder = Path(anchor["origins"][name]["source_directory"])
        require_equal(digest(folder / "metadata.json"), anchor["origins"][name]["hashes"]["metadata.json"],
                      "Reused historical metadata changed")
        original = read(folder / "metadata.json")
        require_equal({key: meta[key] for key in original}, original, "A reused arm's provenance was relabeled")
        old_auditor().validate_evaluation_metadata(ORIGINAL, original, expected_config=old_configs[name])
        return True
    require_equal(meta["recovery_origin"], kind, "Recovery origin was relabeled")
    require_equal(meta["source_sha256"], {name: manifest["sources"][name] for name in RUN_SOURCES},
                  "Recovery arm does not record its actual executed runtime sources")
    sid = config["slice_id"]
    require_equal(meta["data_sha256"], manifest["data"][sid], "Recovery arm data changed")
    require_equal(meta["calibration_sha256"], manifest["calibration"][sid], "Recovery arm calibration changed")
    require_equal({key: meta[key] for key in ENVIRONMENT}, anchor["environment"], "Recovery environment changed")
    require_equal(meta["optimizer"], old_auditor()._optimizer(config, meta["curvature"]),
                  "Recovery changes selected optimizer parameters")
    if kind == "import_checkpoint":
        imported = meta.get("imported_checkpoint")
        if not imported:
            raise ValueError("An expected recovered checkpoint lacks its source audit")
        _source_audit(root, name, imported, anchor)
        require_equal(meta.get("continuation"), read(Path(imported["source_directory"]) / "metadata.json").get("continuation"),
                      "Recovery changed the original native-prefix provenance")
    elif meta.get("imported_checkpoint") is not None:
        raise ValueError("An unplanned partial checkpoint was imported")
    if kind == "continue_original_prefix":
        continuation = meta.get("continuation")
        if not continuation:
            raise ValueError("An expected original prefix is missing")
        pinned = anchor["origins"][name]
        require_equal(str(Path(continuation["source_directory"]).resolve()), pinned["source_directory"],
                      "Original prefix source changed")
        for filename, field in (("trace.json", "source_trace_sha256"), ("checkpoint.pkl", "source_checkpoint_sha256"),
                                ("result.json", "source_result_sha256")):
            require_equal(digest(Path(pinned["source_directory"]) / filename), pinned["hashes"][filename],
                          "An original prefix source file changed")
            require_equal(continuation[field], pinned["hashes"][filename], "Original prefix hash changed")
    elif kind == "fresh" and meta.get("continuation") is not None:
        raise ValueError("An unplanned original prefix was imported")
    return True


def audit_evaluation(root=ROOT, *, require_complete=False):
    root = Path(root).resolve()
    _, _, plan, configs, _ = _context(root)
    present, complete = set(), set()
    for path in sorted((root / "runs").glob("evaluation_*/metadata.json")):
        meta = read(path)
        validate_evaluation_metadata(root, meta)
        name = meta["config"]["run_id"]
        if name in present:
            raise ValueError("Duplicate recovered arm")
        present.add(name)
        result_path = path.parent / "result.json"
        if result_path.exists():
            result = read(result_path)
            validate_evaluation_metadata(root, result)
            require_equal({key: result[key] for key in meta}, meta, "Recovery result and metadata disagree")
            require_equal(result["status"], "completed", "A recovered result is incomplete")
            complete.add(name)
    if require_complete:
        require_equal(complete, set(configs), "Recovery requires all 308 completed arms")
    result = dict(status="validated", expected_runs=len(configs), registered_runs=len(present),
                  completed_runs=len(complete), recovery_input_anchor_sha256=digest(root / "recovery_input_anchor.json"),
                  compatibility_manifest_sha256=digest(root / "compatibility_manifest.json"),
                  provenance_gate_sha256=digest(Path(__file__)),
                  origin_counts=dict(Counter(row["kind"] for row in plan["origins"].values())))
    write(root / "evaluation_provenance_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    value = freeze_recovery_anchor(args.root) if args.freeze else audit_evaluation(args.root, require_complete=args.require_complete)
    print(json.dumps(value, indent=2))
