"""Validate acquisition preprocessing and reference orientation, without optimization."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ct_problem import CTProblem
from prepare_expanded_data import NEW_SLICES, preparation_manifest


ROOT = Path(__file__).resolve().parent


def reference_diagnostics(problem, views):
    observed = problem.y[views].astype(np.float64)
    reference = problem.reference
    expected = problem.forward(reference, views).astype(np.float64)
    native_orientation = problem.forward(reference[::-1], views).astype(np.float64)
    scale = float(np.sum(expected * observed) / np.sum(expected**2))
    denominator = float(np.linalg.norm(observed))
    return dict(
        reference_min=float(reference.min()), reference_max=float(reference.max()),
        measured_rms=float(np.sqrt(np.mean(observed**2))),
        relative_reference_reprojection_error=float(np.linalg.norm(expected - observed) / denominator),
        fitted_reference_scale=scale,
        relative_fitted_reference_reprojection_error=float(np.linalg.norm(scale * expected - observed) / denominator),
        relative_unflipped_reference_reprojection_error=float(np.linalg.norm(native_orientation - observed) / denominator),
        reference_measurement_cosine=float(np.sum(expected * observed) / (np.linalg.norm(expected) * denominator)),
    )


def adjoint_diagnostic(problem, views):
    rng = np.random.default_rng(2601106)
    image = rng.standard_normal((problem.size, problem.size))
    sino = rng.standard_normal((len(views), problem.detectors))
    forward = problem.forward(image, views).astype(np.float64)
    backward = problem.backward(sino, views).astype(np.float64)
    left, right = float(np.sum(forward * sino)), float(np.sum(image * backward))
    error = abs(left - right) / max(abs(left), abs(right), 1e-12)
    if error > 1e-4:
        raise ValueError(f"Unmatched forward/adjoint pair: {error}")
    return dict(relative_adjoint_inner_product_error=error)


def main():
    manifest = preparation_manifest()
    if manifest["status"] != "complete":
        raise ValueError("All prespecified datasets must be complete before validation")
    views = np.arange(0, 3600, 300)
    records = {}
    # Include one unchanged detector-2 acquisition as an orientation/scale control.
    for sid in [3041] + NEW_SLICES:
        key = f"slice{sid:05d}"
        problem = CTProblem(ROOT / "data" / f"{key}.npz", 512, 360)
        try:
            values = reference_diagnostics(problem, views)
            values.update(adjoint_diagnostic(problem, views))
            values.update(grid=512, views=views.tolist(), detectors=problem.detectors)
            print(key, json.dumps(values), flush=True)
            if not all(np.isfinite(v) for v in values.values() if isinstance(v, (int, float))):
                raise ValueError(f"Nonfinite diagnostic for {key}")
            if values["reference_min"] < 0:
                raise ValueError(f"Negative reference attenuation for {key}")
            if not (0.8 < values["fitted_reference_scale"] < 1.2):
                raise ValueError(f"Reference scaling is inconsistent for {key}")
            if values["reference_measurement_cosine"] < 0.9:
                raise ValueError(f"Reference/measurement correlation mismatch for {key}")
            if (values["relative_reference_reprojection_error"]
                    >= values["relative_unflipped_reference_reprojection_error"]):
                raise ValueError(f"Reference orientation is inconsistent for {key}")
            records[key] = values
        finally:
            problem.close()
    output = dict(
        status="validated", purpose="Preprocessing/reference/adjoint validation only; no optimizer outcomes used",
        data_group_manifest_sha256=hashlib.sha256((ROOT / "data_group_manifest.json").read_bytes()).hexdigest(),
        preparation_manifest_sha256=hashlib.sha256((ROOT / "data_preparation_manifest.json").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records,
    )
    (ROOT / "data_validation.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
