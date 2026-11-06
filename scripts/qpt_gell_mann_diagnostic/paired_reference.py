"""Independent dense n=2 initializer reference and archived-pair checks."""
import json
from pathlib import Path

from support import file_hash, require


def dense_gell_mann(np, d=4):
    basis = [np.eye(d, dtype=complex) / np.sqrt(d)]
    for imaginary in (False, True):
        for j in range(d):
            for k in range(j + 1, d):
                matrix = np.zeros((d, d), dtype=complex)
                matrix[j, k], matrix[k, j] = (-1j, 1j) if imaginary else (1, 1)
                basis.append(matrix / np.sqrt(2))
    for level in range(1, d):
        diagonal = np.zeros(d)
        diagonal[:level], diagonal[level] = 1, -level
        basis.append(np.diag(diagonal) / np.sqrt(level * (level + 1)))
    return np.column_stack([matrix.T.reshape(-1) for matrix in basis])


def load_prior(path, plan):
    require(file_hash(path) == plan["prior"]["audit_sha256"], "Prior audit checksum differs")
    report = json.loads(Path(path).read_text())
    require(report["audit_passed"] and report["audited_trials"] == 12
            and not report["missing_trial_ids"]
            and report["provenance"]["source_commit"] == plan["prior"]["commit"],
            "Prior audit is not the complete frozen baseline")
    return {case["trial_id"]: case for case in report["cases"]}


def check_prior(prior, trial, arrays, metadata, np):
    case = prior[trial["id"]]
    require(case["trial"] == trial, "Baseline trial differs beyond initializer")
    path = Path(case["json_path"])
    require(file_hash(path) == case["json_sha256"]
            and file_hash(path.with_suffix(".npz")) == case["npz_sha256"],
            "Baseline archive checksum differs")
    old = json.loads(path.read_text())
    cfg = dict(metadata["configuration"])
    require(cfg.pop("initialization_basis") == "gell-mann" and cfg == old["configuration"],
            "Actual configuration differs from baseline beyond initializer")
    requested = dict(metadata["diagnostic"]["requested_configuration"])
    requested.pop("initialization_basis")
    require(requested == old["diagnostic"]["requested_configuration"],
            "Requested configuration differs from baseline beyond initializer")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        require(json.loads(str(archive["metadata_json"].item())) == old, "Baseline JSON/NPZ differ")
        for key in ("active_rows", "active_observations", "truth_factor"):
            require(np.array_equal(archive[key], arrays[key]), "Baseline/new data or truth differs")
        coefficients = archive["initial_factor"].copy()
    rng = np.random.default_rng(200000 + 10 * trial["channel"])
    fresh = rng.uniform(size=(16, 1)) + 1j * rng.uniform(size=(16, 1))
    fresh *= 2 / np.linalg.norm(fresh, ord="fro")
    # The raw RNG draw is reproducible; its norm reduction can differ by an ULP
    # across the archived Ruche and local BLAS implementations.
    coefficient_error = float(np.max(np.abs(fresh - coefficients)))
    require(np.allclose(fresh, coefficients, rtol=1e-14, atol=1e-15),
            "Archived baseline is not the specified coefficient draw")
    w = dense_gell_mann(np)
    require(np.allclose(w.conj().T @ w, np.eye(16), rtol=0, atol=1e-14), "Reference basis not unitary")
    expected = w @ coefficients
    error = float(np.max(np.abs(arrays["initial_factor"] - expected)))
    require(np.allclose(arrays["initial_factor"], expected, rtol=1e-13, atol=1e-14),
            "Initializer differs from independent dense basis reference")
    require(not np.allclose(arrays["initial_factor"], coefficients), "Initializer basis was not changed")
    return {"trial_id": case["trial_id"], "status": case["status"],
            "endpoint_physical": case["endpoint_physical"], "gate": case["gate"],
            "completed_steps": case["completed_steps"],
            "actual_optimizer_row_accesses": case["actual_optimizer_row_accesses"],
            "json_path": str(path), "json_sha256": case["json_sha256"],
            "npz_sha256": case["npz_sha256"], "dense_initializer_max_absolute_error": error,
            "coefficient_draw_max_absolute_error": coefficient_error,
            "coefficient_draw_tolerance": {"rtol": 1e-14, "atol": 1e-15},
            "qualification": "Same coefficient draw, data, and algorithm configuration; only initializer coordinate basis changed. No trajectory equality is assumed."}
