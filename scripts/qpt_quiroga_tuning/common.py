"""Pinned, calibration-first tuning protocol; no optimizer implementation here."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.qpt_gell_mann_diagnostic import support as base

require = base.require
GATE = {"fidelity_min": .99, "rmse_max": 1e-6, "normalized_tp_max": .001}
METHODS = {"printed-adafgd", "exact-linesearch-fgd"}
TRIAL_KEYS = {"id", "channel", "initialization_seed", "method", "alpha", "eta_scale", "n_steps", "metrics_frequency"}
SELECTION_KEYS = {"method", "alpha", "eta_scale", "n_steps", "metrics_frequency"}


def validate_plan(plan):
    require(plan["schema_version"] == 1 and plan["gate"] == GATE, "Schema/gate differs")
    require(plan["phase"] in {"calibration", "validation"}, "Unknown phase")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", plan["campaign_id"]), "Unsafe campaign ID")
    require(re.fullmatch(r"[0-9a-f]{40}", plan["source"]["commit"]), "Invalid commit")
    require(Path(plan["source"]["repo"]).is_absolute(), "Source path must be absolute")
    require(plan["resources"]["minimum_seconds_before_new_trial"] >= 120, "Time guard below 120 seconds")
    trials = plan["trials"]
    require(isinstance(trials, list) and trials, "No trials")
    require(len({t["id"] for t in trials}) == len(trials), "Duplicate trial ID")
    for trial in trials:
        require(set(trial) == TRIAL_KEYS and re.fullmatch(r"[A-Za-z0-9_-]+", trial["id"]), "Malformed trial")
        for key in ("channel", "initialization_seed", "n_steps", "metrics_frequency"):
            require(type(trial[key]) is int and trial[key] >= (1 if key in {"n_steps", "metrics_frequency"} else 0), "Invalid integer: " + key)
        require(trial["method"] in METHODS and all(base.finite_positive(trial[k]) for k in ("alpha", "eta_scale")), "Invalid method/scale")
        require(base.finite_positive(128 * trial["alpha"]), "Derived TP weight must be finite")
        require(trial["method"] != "exact-linesearch-fgd" or trial["eta_scale"] == 1, "Exact line search does not use eta_scale")
        require(trial["channel"] < 100 if plan["phase"] == "calibration" else 100 <= trial["channel"] <= 107,
                "Channel outside the authorized split")
    if plan["phase"] == "validation":
        selection = plan["selection"]
        require(re.fullmatch(r"[0-9a-f]{64}", selection["calibration_audit_sha256"]), "Missing calibration audit pin")
        require(set(selection["configuration"]) == SELECTION_KEYS, "Incomplete selected configuration")
        require(all({k: t[k] for k in SELECTION_KEYS} == selection["configuration"] for t in trials), "Validation retunes selected configuration")
    return trials


def source_hashes(repo, commit):
    repo = Path(repo)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == commit and not git("status", "--porcelain", "--untracked-files=all"), "Checkout must be clean at pinned commit")
    paths = git("ls-files", "paper", "frank_wolfe", "scripts/qpt_quiroga_tuning", "scripts/qpt_gell_mann_diagnostic/support.py").splitlines()
    paths = [p for p in paths if p.endswith((".py", ".sbatch"))]
    return {p: base.file_hash(repo / p) for p in paths}


def check_selection(plan, audit_path):
    if plan["phase"] != "validation":
        return
    require(audit_path is not None, "Validation requires the frozen calibration audit file")
    require(base.file_hash(audit_path) == plan["selection"]["calibration_audit_sha256"], "Calibration audit checksum differs")
    audit = json.loads(Path(audit_path).read_text())
    require(audit["kind"] == "qpt_quiroga_tuning_audit" and audit["audit_passed"] and not audit["partial_mode"]
            and not audit["missing_trial_ids"] and not audit["trial_ids_without_inventory_checksum"]
            and audit["provenance"]["phase"] == "calibration", "Selection audit is not complete calibration")
    matches = [case for case in audit["cases"] if {k: case["trial"][k] for k in SELECTION_KEYS} == plan["selection"]["configuration"]]
    require(matches and all(case["gate"]["pass"] and case["status"] in {"complete", "converged"} for case in matches)
            and {case["channel"] for case in matches} == {case["channel"] for case in audit["cases"]},
            "Selected configuration did not pass across audited calibration channels")


def provenance(plan, plan_path, expected_hash, repo):
    require(base.file_hash(plan_path) == expected_hash, "Plan checksum mismatch")
    hashes = source_hashes(repo, plan["source"]["commit"])
    require(hashes == plan["source"]["sha256"], "Source checksum mismatch")
    require(REPO == Path(repo).resolve(), "Harness must run from the pinned checkout")
    return {"campaign_id": plan["campaign_id"], "phase": plan["phase"], "plan_sha256": expected_hash,
            "source_commit": plan["source"]["commit"], "source_sha256": hashes}


def check_module_origins(repo):
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"paper", "frank_wolfe"} and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(Path(repo).resolve()), "Unpinned module: " + name)


def write_json(path, value, replace=False):
    """Atomic publication, refusing initial overwrite; updates replace inventory only."""
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()


def configuration(trial):
    result = dict(methods=[trial["method"]], rank=1, initialization_basis="gell-mann",
                  initialization_seed=trial["initialization_seed"], sampling_seed=300000 + 100 * trial["channel"],
                  n_steps=trial["n_steps"], metrics_frequency=trial["metrics_frequency"],
                  eta_scale=trial["eta_scale"], tp_weight=128 * trial["alpha"],
                  spectral_norm_method="dense", show_progress=False)
    if trial["method"] == "exact-linesearch-fgd":
        result.update(linesearch_gradient_atol=1e-12, linesearch_gradient_rtol=1e-10)
    return result


def actual_configuration(trial):
    """Independent explicit archive contract, including unused campaign defaults."""
    cfg = configuration(trial)
    beta = 128 / (2 * cfg["tp_weight"])
    result = {k: v for k, v in cfg.items() if k not in {"show_progress", "tp_weight"}}
    result.update(frames_steps=cfg["n_steps"], frames_metrics_frequency=cfg["metrics_frequency"],
                  tp_weight_lambda=cfg["tp_weight"], frames_beta=beta, frames_beta0=beta,
                  frames_initial_tp_weight_per_row_alpha=1 / (2 * beta),
                  frames_initial_tp_weight_lambda=128 / (2 * beta), frames_penalty_mode="legacy_constant_matched",
                  frames_smoothing_schedule={"type": "constant", "value": beta}, tau=2., batch_size=32,
                  dense_max_process_dimension=1024, spectral_norm_tolerance=1e-10, spectral_norm_maxiter=None,
                  armijo_initial_step=1., armijo_shrink=.5, armijo_constant=1e-4, armijo_max_backtracks=30,
                  armijo_gradient_atol=0., armijo_gradient_rtol=0.,
                  rho_schedule={"type": "PowerSchedule", "scale": 2., "offset": 4., "exponent": .6, "cap": 1.},
                  step_size_schedule={"type": "PowerSchedule", "scale": 2., "offset": 2., "exponent": 1., "cap": 1.})
    if trial["method"] == "exact-linesearch-fgd":
        result.update(linesearch_rule="exact_positive_factor_ray_quartic_v1", linesearch_root_imaginary_relative_tolerance=1e-8)
    return result


def array_hash(array, np):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def initial_reference(seed, np):
    """Independent dense HS-unit Gell-Mann basis, input-major vec convention."""
    basis = [np.eye(4, dtype=complex) / 2]
    for imaginary in (False, True):
        for j in range(4):
            for k in range(j + 1, 4):
                matrix = np.zeros((4, 4), dtype=complex)
                matrix[j, k], matrix[k, j] = (-1j, 1j) if imaginary else (1, 1)
                basis.append(matrix / np.sqrt(2))
    for level in range(1, 4):
        diagonal = np.zeros(4)
        diagonal[:level], diagonal[level] = 1, -level
        basis.append(np.diag(diagonal) / np.sqrt(level * (level + 1)))
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(size=(16, 1)) + 1j * rng.uniform(size=(16, 1))
    coefficients *= 2 / np.linalg.norm(coefficients)
    return np.column_stack([b.T.reshape(-1) for b in basis]) @ coefficients
