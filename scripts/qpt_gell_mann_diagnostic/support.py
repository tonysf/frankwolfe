"""Small archival helpers retained from the 2026-10-06 diagnostic harness.

Original run_diagnostics.py SHA256:
13bcfa92d5d4876cc68f4d04fa4b0869d611c67d42ec52cc56c067925c3884b6.
No optimizer implementation is copied here.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

METHODS = ("printed-adafgd", "armijo-fgd", "frames")
SOURCE_PATHS = (
    "paper/experiments/qpt_quiroga_campaign.py",
    "paper/experiments/qpt_quiroga_frames.py",
    "paper/experiments/qpt_quiroga_sensing.py",
    "paper/experiments/qpt_generate_data.py",
    "paper/experiments/qpt_observation_noise.py",
    "paper/experiments/quantum_process_tomography.py",
    "frank_wolfe/__init__.py", "frank_wolfe/algorithms/base.py",
    "frank_wolfe/algorithms/frames.py", "frank_wolfe/core/objective.py",
)
PHYSICAL_METRICS = (
    "measurement_loss_sum", "measurement_loss_mean", "rmse", "tp_violation",
    "normalized_tp_violation", "process_fidelity",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def finite_positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    target = path.with_name(path.name + ".pending") if replace else path
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def gate_for(status, metrics, gate):
    def meets(key, threshold, lower):
        value = metrics.get(key)
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)
                and (value >= threshold if lower else value <= threshold))
    components = {
        "fidelity": meets("process_fidelity", gate["fidelity_min"], True),
        "full_design_rmse": meets("rmse", gate["rmse_max"], False),
        "normalized_tp": meets("normalized_tp_violation", gate["normalized_tp_max"], False),
    }
    return {"thresholds": gate, "components": components,
            "pass": status in {"complete", "converged"} and all(components.values()),
            "method_status": status}


def independent_endpoint(data, trace, targets, np, evaluator):
    """No fallback from undefined full-design metrics to prior checkpoints."""
    errors = {}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Zero reporting penalty keeps these physical diagnostics independent
            # of the method-native objective and its potentially huge penalty.
            metrics = evaluator(data, trace.final_factor,
                                np.arange(data.m, dtype=np.int64), targets, 0.0)
        physical = {key: float(metrics[key]) for key in PHYSICAL_METRICS}
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        if trace.status != "failed":
            raise
        physical = {key: None for key in PHYSICAL_METRICS}
        errors["full_design_metrics"] = f"{type(error).__name__}: {error}"
    for name, order in (("factor_frobenius_norm", "fro"), ("factor_operator_norm", 2)):
        try:
            with np.errstate(over="raise", invalid="raise"):
                value = float(np.linalg.norm(trace.final_factor, ord=order))
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite factor norm")
            physical[name] = value
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            if trace.status != "failed":
                raise
            physical[name] = None
            errors[name] = f"{type(error).__name__}: {error}"
    return physical, errors
