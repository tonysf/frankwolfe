"""Direct FRAMES versus printed-adaFGD comparisons on Quiroga sensing.

The validated campaign core already makes methods share sensing rows,
observations, and an initial factor.  This thin protocol layer additionally
forces the two-method comparison, matches nominal active-row budgets and
checkpoints, records a common penalty normalization and recovery gate, and
aggregates compatible JSON sidecars.

One nominal row epoch is ``m`` row accesses.  This is not compute-equivalent:
FRAMES samples with replacement, while printed adaFGD performs a full pass and
an additional spectral-norm calculation.  The archive records that caveat.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np

from .qpt_quiroga_campaign import (
    COMMON_METRICS,
    evaluate_common_metrics,
    run_campaign,
    save_campaign,
)
from .qpt_quiroga_sensing import generate_quiroga_sensing_data
from .quantum_process_tomography import PowerSchedule


SCHEMA_VERSION = 1
FORMAT = "qpt_quiroga_direct_comparison"
SUMMARY_FORMAT = "qpt_quiroga_direct_comparison_summary"
PROTOCOL = "shared_objective_matched_nominal_row_accesses_v1"
DIRECT_METHODS = ("printed-adafgd", "frames")

_ENDPOINT_METRICS = (
    *COMMON_METRICS,
    "full_design_rmse",
    "factor_frobenius_norm",
    "factor_operator_norm",
    "optimizer_seconds",
    "cumulative_data_row_accesses",
)
_SOURCE_PATHS = (
    "paper/experiments/qpt_quiroga_direct_comparison.py",
    "paper/experiments/qpt_quiroga_campaign.py",
    "paper/experiments/qpt_quiroga_frames.py",
    "paper/experiments/qpt_quiroga_sensing.py",
    "paper/experiments/qpt_generate_data.py",
    "paper/experiments/qpt_observation_noise.py",
    "paper/experiments/quantum_process_tomography.py",
    "frank_wolfe/__init__.py",
    "frank_wolfe/algorithms/base.py",
    "frank_wolfe/algorithms/frames.py",
    "frank_wolfe/core/objective.py",
)
_COHORT_CONFIGURATION_FIELDS = (
    "methods", "n_steps", "frames_steps", "metrics_frequency",
    "frames_metrics_frequency", "rank", "eta_scale", "tp_weight_lambda",
    "tp_weight_per_row_alpha", "frames_beta", "tau", "batch_size",
    "spectral_norm_method", "dense_max_process_dimension",
    "spectral_norm_tolerance", "spectral_norm_maxiter", "rho_schedule",
    "step_size_schedule", "observation_mode", "noise_std", "shots",
)


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value <= 0):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive finite number.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _validated_schedule(value, name):
    if value is None or isinstance(value, PowerSchedule):
        return value
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be None, a scalar, or PowerSchedule.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value):
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, str(array.dtype))).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _finite_number(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class MatchedRowBudget:
    active_row_count: int
    batch_size: int
    row_epochs: int
    checkpoint_epochs: int
    printed_adafgd_steps: int
    frames_steps: int
    printed_adafgd_metrics_frequency: int
    frames_metrics_frequency: int
    nominal_row_accesses: int

    def as_dict(self):
        return {
            "unit": "nominal_active_row_accesses",
            "active_row_count": self.active_row_count,
            "batch_size": self.batch_size,
            "row_epochs": self.row_epochs,
            "checkpoint_epochs": self.checkpoint_epochs,
            "nominal_row_accesses": self.nominal_row_accesses,
            "method_steps": {
                "printed-adafgd": self.printed_adafgd_steps,
                "frames": self.frames_steps,
            },
            "metrics_frequency_steps": {
                "printed-adafgd": self.printed_adafgd_metrics_frequency,
                "frames": self.frames_metrics_frequency,
            },
            "compute_equivalent": False,
            "qualification": (
                "FRAMES samples active rows with replacement; printed adaFGD "
                "uses full active-row passes and additional spectral-norm work. "
                "Checkpoint passes and TP/LMO work are excluded."
            ),
        }


def derive_matched_row_budget(active_row_count, batch_size, row_epochs,
                              checkpoint_epochs):
    """Derive exact method steps and row-access-aligned checkpoints."""
    active_row_count = _positive_integer(active_row_count, "active_row_count")
    batch_size = _positive_integer(batch_size, "batch_size")
    row_epochs = _positive_integer(row_epochs, "row_epochs")
    checkpoint_epochs = _positive_integer(checkpoint_epochs, "checkpoint_epochs")
    if checkpoint_epochs > row_epochs:
        raise ValueError("checkpoint_epochs cannot exceed row_epochs.")
    total = active_row_count * row_epochs
    checkpoint = active_row_count * checkpoint_epochs
    if total % batch_size:
        raise ValueError(
            "row_epochs * active_row_count must be divisible by batch_size.")
    if checkpoint % batch_size:
        raise ValueError(
            "checkpoint_epochs * active_row_count must be divisible by batch_size.")
    return MatchedRowBudget(
        active_row_count, batch_size, row_epochs, checkpoint_epochs, row_epochs,
        total // batch_size, checkpoint_epochs, checkpoint // batch_size, total,
    )


@dataclass(frozen=True)
class RecoveryGate:
    name: str = "noiseless_recovery_v1"
    process_fidelity_min: float = 0.99
    rmse_max: float = 1e-6
    normalized_tp_violation_max: float = 1e-3

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("gate name must be a nonempty string.")
        values = (float(self.process_fidelity_min), float(self.rmse_max),
                  float(self.normalized_tp_violation_max))
        if (not all(math.isfinite(value) for value in values)
                or not 0 <= values[0] <= 1 or values[1] < 0 or values[2] < 0):
            raise ValueError("recovery-gate thresholds are invalid.")
        object.__setattr__(self, "process_fidelity_min", values[0])
        object.__setattr__(self, "rmse_max", values[1])
        object.__setattr__(self, "normalized_tp_violation_max", values[2])

    def as_dict(self):
        return {
            "name": self.name,
            "thresholds": {
                "process_fidelity": {"operator": ">=", "value": self.process_fidelity_min},
                "rmse": {"operator": "<=", "value": self.rmse_max},
                "normalized_tp_violation": {
                    "operator": "<=", "value": self.normalized_tp_violation_max,
                },
            },
        }

    def evaluate(self, endpoint, *, rmse=None, rmse_source="active_rows"):
        if not isinstance(endpoint, Mapping):
            raise TypeError("endpoint must be a mapping.")
        values = {
            "process_fidelity": _finite_number(endpoint.get("process_fidelity")),
            "rmse": _finite_number(endpoint.get("rmse") if rmse is None else rmse),
            "normalized_tp_violation": _finite_number(
                endpoint.get("normalized_tp_violation")),
        }
        components = {
            "process_fidelity": (values["process_fidelity"] is not None
                                 and values["process_fidelity"] >= self.process_fidelity_min),
            "rmse": values["rmse"] is not None and values["rmse"] <= self.rmse_max,
            "normalized_tp_violation": (
                values["normalized_tp_violation"] is not None
                and values["normalized_tp_violation"] <= self.normalized_tp_violation_max),
        }
        eligible = endpoint.get("status") in {"complete", "converged"}
        return {
            "status_eligible": eligible,
            "metric_values": values,
            "rmse_source": rmse_source,
            "component_passes": components,
            "passed": bool(eligible and all(components.values())),
        }

    @classmethod
    def from_dict(cls, value):
        try:
            thresholds = value["thresholds"]
            fidelity = thresholds["process_fidelity"]
            rmse = thresholds["rmse"]
            tp = thresholds["normalized_tp_violation"]
            if (fidelity["operator"], rmse["operator"], tp["operator"]) != (
                    ">=", "<=", "<="):
                raise ValueError("gate threshold operators are unsupported.")
            return cls(value["name"], fidelity["value"], rmse["value"], tp["value"])
        except (KeyError, TypeError) as error:
            raise ValueError("gate metadata is incomplete.") from error


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _git(root, *arguments):
    result = subprocess.run(("git", *arguments), cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        return None, result.stderr.decode(errors="replace").strip()
    return result.stdout, None


def collect_source_provenance(repo_root=None):
    """Collect exact source hashes and non-mutating Git/runtime provenance."""
    root = _repo_root() if repo_root is None else Path(repo_root).resolve()
    source_hashes = {}
    for relative in _SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison source is missing: {path}")
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    commit, commit_error = _git(root, "rev-parse", "HEAD")
    branch, branch_error = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status, status_error = _git(root, "status", "--porcelain", "--untracked-files=all")
    diff, diff_error = _git(root, "diff", "--binary", "HEAD", "--")
    errors = [error for error in (commit_error, branch_error, status_error, diff_error)
              if error]
    git = {
        "available": not errors,
        "commit": None if commit is None else commit.decode().strip(),
        "branch": None if branch is None else branch.decode().strip(),
        "worktree_dirty": None if status is None else bool(status.strip()),
        "tracked_diff_sha256": None if diff is None else hashlib.sha256(diff).hexdigest(),
    }
    if errors:
        git["error"] = "; ".join(errors)
    slurm_keys = (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
    )
    return {
        "repository_root": str(root),
        "git": git,
        "source_sha256": source_hashes,
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "command": list(sys.argv),
            "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
        },
    }


def _data_instance_spec(metadata):
    comparison = metadata["comparison"]
    return {
        "design": metadata["design"],
        "active_row_count": metadata["active_row_count"],
        "active_rows_sha256": metadata["active_rows_sha256"],
        "active_observations_sha256": metadata["active_observations_sha256"],
        "full_observations_sha256": comparison["full_observations_sha256"],
        "truth_factor_sha256": metadata["truth_factor_sha256"],
    }


def _paired_instance_spec(metadata):
    comparison = metadata["comparison"]
    return {
        "data_instance_sha256": comparison["data_instance_sha256"],
        "initial_factor_sha256": metadata["initial_factor_sha256"],
        "rank": metadata["configuration"]["rank"],
    }


def _cohort_spec(metadata):
    comparison = metadata["comparison"]
    configuration = metadata["configuration"]
    provenance = metadata["provenance"]
    data_metadata = metadata.get("data_metadata", {})
    return {
        "protocol": comparison["protocol"],
        "cohort_id": comparison.get("cohort_id"),
        "split": comparison["split"],
        "design": metadata["design"],
        "active_row_count": metadata["active_row_count"],
        "observation": {
            "mode": metadata["observation_mode"],
            "noise_std": configuration["noise_std"],
            "shots": configuration["shots"],
            "channel_model": data_metadata.get("channel_model"),
            "truth_rank": data_metadata.get("truth_rank"),
            "noise_generator": data_metadata.get("noise_generator"),
            "shot_generator": data_metadata.get("shot_generator"),
        },
        "software": {
            "python_version": metadata["python_version"],
            "numpy_version": metadata["numpy_version"],
            "scipy_version": metadata["scipy_version"],
        },
        "objective": metadata["objective"],
        "configuration": {
            name: configuration[name] for name in _COHORT_CONFIGURATION_FIELDS
        },
        "method_domains": metadata["method_domains"],
        "budget": comparison["budget"],
        "shared_penalty": comparison["shared_penalty"],
        "gate": comparison["gate"],
        "source": {
            "git_commit": provenance["git"].get("commit"),
            "git_tracked_diff_sha256": provenance["git"].get("tracked_diff_sha256"),
            "source_sha256": provenance["source_sha256"],
        },
    }


def _endpoint_values(metadata, method):
    values = dict(metadata["methods"][method])
    diagnostics = metadata["comparison"]["endpoint_diagnostics"][method]
    values.update({
        "full_design_rmse": diagnostics["full_design_metrics"]["rmse"],
        "factor_frobenius_norm": diagnostics["factor_frobenius_norm"],
        "factor_operator_norm": diagnostics["factor_operator_norm"],
    })
    return values


def _endpoint_differences(metadata):
    printed = _endpoint_values(metadata, "printed-adafgd")
    frames = _endpoint_values(metadata, "frames")
    values = {}
    for metric in _ENDPOINT_METRICS:
        left, right = _finite_number(frames.get(metric)), _finite_number(printed.get(metric))
        values[metric] = None if left is None or right is None else left - right
    return {"orientation": "frames_minus_printed_adafgd", "values": values}


def _first_gate_passage(trace, gate, active_row_count, available):
    if not available:
        return {
            "available": False,
            "reason": (
                "Full-design checkpoint factors are not retained for subset or "
                "non-noiseless runs; only the endpoint is evaluated on the full design."
            ),
            "first_pass": None,
        }
    for index, step in enumerate(trace.checkpoint_steps):
        endpoint = {
            "status": "complete",
            "process_fidelity": trace.process_fidelity[index],
            "rmse": trace.rmse[index],
            "normalized_tp_violation": trace.normalized_tp_violation[index],
        }
        decision = gate.evaluate(endpoint, rmse_source="full_design")
        if decision["passed"]:
            accesses = int(trace.cumulative_data_row_accesses[index])
            return {
                "available": True, "reason": None,
                "first_pass": {
                    "completed_steps": int(step), "nominal_row_accesses": accesses,
                    "row_epochs": accesses / active_row_count,
                    "optimizer_seconds": float(trace.optimizer_seconds[index]),
                    "gate": decision,
                },
            }
    return {"available": True, "reason": None, "first_pass": None}


def run_direct_comparison(
    data, *, rows=None, row_epochs, checkpoint_epochs, tp_weight_per_row,
    eta_scale, rank=1, initialization_seed=0, sampling_seed=0, tau=None,
    batch_size=32, spectral_norm_method="auto",
    dense_max_process_dimension=1024, spectral_norm_tolerance=1e-10,
    spectral_norm_maxiter=None, rho_schedule=None, step_size_schedule=None,
    gate=None, trial_id=None, cohort_id=None, split="unspecified",
    expected_commit=None, require_clean=False, show_progress=False,
):
    """Run one direct pair using ``alpha`` as the portable TP coefficient."""
    if split not in {"calibration", "heldout", "unspecified"}:
        raise ValueError("split must be 'calibration', 'heldout', or 'unspecified'.")
    for value, name in ((trial_id, "trial_id"), (cohort_id, "cohort_id")):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string or None.")
    if not isinstance(require_clean, (bool, np.bool_)):
        raise TypeError("require_clean must be a boolean.")
    active_rows = (np.arange(data.m, dtype=np.int64) if rows is None
                   else data.design.validate_rows(rows))
    budget = derive_matched_row_budget(
        active_rows.size, batch_size, row_epochs, checkpoint_epochs)
    alpha = _finite_positive(tp_weight_per_row, "tp_weight_per_row")
    summed_lambda = active_rows.size * alpha
    expected_beta = 1.0 / (2.0 * alpha)
    gate = RecoveryGate() if gate is None else gate
    if not isinstance(gate, RecoveryGate):
        raise TypeError("gate must be a RecoveryGate.")
    if data.observation_mode != "noiseless" and gate.name == "noiseless_recovery_v1":
        raise ValueError(
            "The noiseless_recovery_v1 gate cannot be applied to non-noiseless data.")

    provenance = collect_source_provenance()
    effective_tau = math.sqrt(data.d) if tau is None else _finite_positive(tau, "tau")
    truth_operator_norm = float(np.linalg.norm(data.truth_factor, ord=2))
    if truth_operator_norm > effective_tau + 1e-10:
        raise ValueError(
            f"tau={effective_tau} excludes the truth factor with operator norm "
            f"{truth_operator_norm}.")
    rho_schedule = _validated_schedule(rho_schedule, "rho_schedule")
    step_size_schedule = _validated_schedule(step_size_schedule, "step_size_schedule")

    git = provenance["git"]
    if expected_commit is not None and git.get("commit") != expected_commit:
        raise ValueError(
            f"Git commit mismatch: expected {expected_commit}, got {git.get('commit')}.")
    if require_clean and git.get("worktree_dirty") is not False:
        raise ValueError("A clean Git worktree, including untracked files, is required.")

    metadata, payload, traces = run_campaign(
        data, methods=DIRECT_METHODS, rows=active_rows, rank=rank,
        initialization_seed=initialization_seed, sampling_seed=sampling_seed,
        n_steps=budget.printed_adafgd_steps, frames_steps=budget.frames_steps,
        metrics_frequency=budget.printed_adafgd_metrics_frequency,
        frames_metrics_frequency=budget.frames_metrics_frequency,
        eta_scale=eta_scale, tp_weight=summed_lambda, tau=tau,
        batch_size=budget.batch_size, spectral_norm_method=spectral_norm_method,
        dense_max_process_dimension=dense_max_process_dimension,
        spectral_norm_tolerance=spectral_norm_tolerance,
        spectral_norm_maxiter=spectral_norm_maxiter, rho_schedule=rho_schedule,
        step_size_schedule=step_size_schedule, show_progress=show_progress,
    )
    configuration = metadata["configuration"]
    configuration.update({
        "tp_weight_per_row_alpha": alpha,
        "observation_mode": data.observation_mode,
        "noise_std": float(data.noise_std),
        "noise_seed": int(data.noise_seed),
        "shots": data.shots,
        "shot_seed": int(data.shot_seed),
    })
    if not math.isclose(configuration["frames_beta"], expected_beta, rel_tol=1e-14):
        raise RuntimeError("FRAMES beta does not match the shared penalty.")

    initial_metrics = {}
    for metric in COMMON_METRICS:
        values = [float(getattr(traces[name], metric)[0]) for name in DIRECT_METHODS]
        if not np.allclose(values, values[0], rtol=1e-13, atol=1e-15):
            raise RuntimeError(f"Methods do not share initial {metric}.")
        initial_metrics[metric] = values[0]

    all_rows = np.arange(data.m, dtype=np.int64)
    all_targets = data.all_observations()
    full_lambda = data.m * alpha
    endpoint_diagnostics = {}
    for method, trace in traces.items():
        full_metrics = evaluate_common_metrics(
            data, trace.final_factor, all_rows, all_targets, full_lambda)
        frobenius = float(np.linalg.norm(trace.final_factor, ord="fro"))
        operator = float(np.linalg.norm(trace.final_factor, ord=2))
        endpoint_diagnostics[method] = {
            "full_design_metrics": full_metrics,
            "final_factor_sha256": _array_sha256(trace.final_factor),
            "factor_frobenius_norm": frobenius,
            "factor_operator_norm": operator,
            "factor_operator_norm_over_tau": operator / configuration["tau"],
        }

    metadata["base_format"] = metadata["format"]
    metadata["format"] = FORMAT
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["provenance"] = provenance
    metadata["source_sha256"] = dict(provenance["source_sha256"])
    comparison = {
        "protocol": PROTOCOL,
        "methods": list(DIRECT_METHODS),
        "cohort_id": cohort_id,
        "split": split,
        "full_observations_sha256": _array_sha256(all_targets),
        "shared_objective": metadata["objective"],
        "shared_penalty": {
            "normalization": "alpha=lambda/active_row_count; beta=1/(2*alpha)",
            "tp_weight_per_row_alpha": alpha,
            "summed_tp_weight_lambda": summed_lambda,
            "frames_beta": expected_beta,
        },
        "budget": budget.as_dict(),
        "initial_metrics": initial_metrics,
        "endpoint_diagnostics": endpoint_diagnostics,
        "first_gate_passage": {
            method: _first_gate_passage(
                traces[method], gate, active_rows.size,
                data.observation_mode == "noiseless"
                and np.array_equal(active_rows, all_rows))
            for method in DIRECT_METHODS
        },
        "gate": gate.as_dict(),
        "baseline_qualification": (
            "printed-adafgd is the literal printed adaptive rule with explicitly "
            "calibrated constants; the paper omits constants needed for an exact reproduction."
        ),
        "domain_audit": {
            "frames_tau": configuration["tau"],
            "truth_factor_operator_norm": truth_operator_norm,
            "truth_inside_frames_domain": truth_operator_norm <= configuration["tau"] + 1e-10,
            "initial_factor_operator_norm": float(np.linalg.norm(
                payload["initial_factor"], ord=2)),
        },
        "known_limitations": [
            "If FRAMES raises during optimization, the core campaign records only "
            "its shared initial point; partial FRAMES iterates are not retained.",
            "Each one-pair sidecar reruns deterministic printed-adaFGD when only "
            "the FRAMES sampling seed changes. Aggregation collapses those baseline "
            "endpoints per data/initialization instance and rejects disagreement; "
            "its repeated optimizer timings remain descriptive measurements.",
        ],
        "deterministic_baseline_reuse": {
            "execution": "rerun_once_per_pair_sidecar",
            "aggregation": (
                "collapse scientific endpoints per paired data/initialization instance; "
                "reject disagreement; summarize repeated timings descriptively"
            ),
        },
        "method_execution_order": list(DIRECT_METHODS),
        "optimizer_timing_interpretation": (
            "descriptive fixed-order timings, not a causal speed comparison"
        ),
    }
    metadata["comparison"] = comparison
    data_instance = _data_instance_spec(metadata)
    comparison["data_instance"] = data_instance
    comparison["data_instance_sha256"] = _json_sha256(data_instance)
    instance_spec = _paired_instance_spec(metadata)
    instance_hash = _json_sha256(instance_spec)
    comparison["paired_instance"] = instance_spec
    comparison["paired_instance_sha256"] = instance_hash

    endpoints = metadata["methods"]
    gate_rmse_source = "full_design" if data.observation_mode == "noiseless" else "active_rows"
    comparison["method_gate"] = {}
    for method in DIRECT_METHODS:
        full_rmse = endpoint_diagnostics[method]["full_design_metrics"]["rmse"]
        comparison["method_gate"][method] = gate.evaluate(
            endpoints[method], rmse=full_rmse if gate_rmse_source == "full_design" else None,
            rmse_source=gate_rmse_source)
    comparison["endpoint_differences"] = _endpoint_differences(metadata)

    cohort_spec = _cohort_spec(metadata)
    cohort_hash = _json_sha256(cohort_spec)
    trial_spec = {
        "cohort_fingerprint": cohort_hash,
        "paired_instance_sha256": instance_hash,
        "sampling_seed": int(sampling_seed),
    }
    trial_hash = _json_sha256(trial_spec)
    comparison.update({
        "cohort_spec": cohort_spec,
        "cohort_fingerprint": cohort_hash,
        "trial_fingerprint": trial_hash,
        "trial_id": trial_hash if trial_id is None else trial_id,
    })
    return metadata, payload, traces


def _validate_sidecar(sidecar, index):
    if not isinstance(sidecar, Mapping):
        raise TypeError(f"sidecar {index} must be a mapping.")
    if sidecar.get("format") != FORMAT or sidecar.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"sidecar {index} is not a supported direct comparison.")
    comparison, configuration = sidecar.get("comparison"), sidecar.get("configuration")
    if not isinstance(comparison, Mapping) or comparison.get("protocol") != PROTOCOL:
        raise ValueError(f"sidecar {index} has an incompatible protocol.")
    if tuple(comparison.get("methods", ())) != DIRECT_METHODS:
        raise ValueError(f"sidecar {index} does not contain the required method pair.")
    methods = sidecar.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(DIRECT_METHODS):
        raise ValueError(f"sidecar {index} has incomplete method endpoints.")
    try:
        active_rows = _positive_integer(sidecar["active_row_count"], "active_row_count")
        alpha = _finite_positive(configuration["tp_weight_per_row_alpha"], "alpha")
        summed_lambda = _finite_positive(configuration["tp_weight_lambda"], "lambda")
        frames_beta = _finite_positive(configuration["frames_beta"], "frames_beta")
        budget = comparison["budget"]
        expected_accesses = int(budget["nominal_row_accesses"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"sidecar {index} has incomplete penalty/budget metadata.") from error
    if (not math.isclose(summed_lambda, active_rows * alpha, rel_tol=1e-13)
            or not math.isclose(frames_beta, 1.0 / (2.0 * alpha), rel_tol=1e-13)):
        raise ValueError(f"sidecar {index} has inconsistent penalty normalization.")
    penalty = comparison["shared_penalty"]
    if (not math.isclose(float(penalty["tp_weight_per_row_alpha"]), alpha, rel_tol=1e-13)
            or not math.isclose(float(penalty["summed_tp_weight_lambda"]),
                                summed_lambda, rel_tol=1e-13)
            or not math.isclose(float(penalty["frames_beta"]),
                                frames_beta, rel_tol=1e-13)):
        raise ValueError(f"sidecar {index} has inconsistent shared penalty.")
    steps = budget["method_steps"]
    if (configuration["n_steps"] != steps["printed-adafgd"]
            or configuration["frames_steps"] != steps["frames"]):
        raise ValueError(f"sidecar {index} has inconsistent budget steps.")
    for method, endpoint in methods.items():
        if (endpoint.get("status") in {"complete", "converged"}
                and endpoint.get("cumulative_data_row_accesses") != expected_accesses):
            raise ValueError(f"sidecar {index} method {method} missed its matched budget.")

    gate = RecoveryGate.from_dict(comparison["gate"])
    source = "full_design" if sidecar["observation_mode"] == "noiseless" else "active_rows"
    for method in DIRECT_METHODS:
        diagnostics = comparison["endpoint_diagnostics"][method]
        factor_hash = diagnostics.get("final_factor_sha256")
        if (not isinstance(factor_hash, str) or len(factor_hash) != 64
                or any(character not in "0123456789abcdef" for character in factor_hash)):
            raise ValueError(
                f"sidecar {index} method {method} has no valid final-factor hash.")
        rmse = diagnostics["full_design_metrics"]["rmse"] if source == "full_design" else None
        if comparison["method_gate"][method] != gate.evaluate(
                methods[method], rmse=rmse, rmse_source=source):
            raise ValueError(f"sidecar {index} has inconsistent gate results.")
    data_instance = _data_instance_spec(sidecar)
    if (comparison.get("data_instance") != data_instance
            or comparison.get("data_instance_sha256") != _json_sha256(data_instance)):
        raise ValueError(f"sidecar {index} has inconsistent data-instance identity.")
    instance = _paired_instance_spec(sidecar)
    if (comparison.get("paired_instance") != instance
            or comparison.get("paired_instance_sha256") != _json_sha256(instance)):
        raise ValueError(f"sidecar {index} has inconsistent instance identity.")
    cohort = _cohort_spec(sidecar)
    cohort_hash = _json_sha256(cohort)
    if (comparison.get("cohort_spec") != cohort
            or comparison.get("cohort_fingerprint") != cohort_hash):
        raise ValueError(f"sidecar {index} has inconsistent cohort metadata.")
    trial = {"cohort_fingerprint": cohort_hash,
             "paired_instance_sha256": comparison["paired_instance_sha256"],
             "sampling_seed": int(configuration["sampling_seed"])}
    if comparison.get("trial_fingerprint") != _json_sha256(trial):
        raise ValueError(f"sidecar {index} has inconsistent trial identity.")
    if not isinstance(comparison.get("trial_id"), str) or not comparison["trial_id"]:
        raise ValueError(f"sidecar {index} has no trial_id.")
    return cohort, cohort_hash


def _quantile(values, probability):
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _statistics(values, total):
    finite = sorted(value for value in map(_finite_number, values) if value is not None)
    if not finite:
        return {"count": 0, "missing_count": total, "mean": None, "median": None,
                "q25": None, "q75": None, "min": None, "max": None}
    return {
        "count": len(finite), "missing_count": total - len(finite),
        "mean": math.fsum(finite) / len(finite), "median": _quantile(finite, 0.5),
        "q25": _quantile(finite, 0.25), "q75": _quantile(finite, 0.75),
        "min": finite[0], "max": finite[-1],
    }


def _mean_or_none(values):
    finite = [value for value in map(_finite_number, values) if value is not None]
    return None if not finite else math.fsum(finite) / len(finite)


def _printed_replicate_signature(record):
    endpoint = _endpoint_values(record, "printed-adafgd")
    return {
        "endpoint_without_timings": {
            key: value for key, value in endpoint.items() if "seconds" not in key
        },
        "final_factor_sha256": record["comparison"]["endpoint_diagnostics"][
            "printed-adafgd"
        ]["final_factor_sha256"],
        "gate": record["comparison"]["method_gate"]["printed-adafgd"],
    }


def aggregate_sidecars(sidecars: Sequence[Mapping]):
    """Purely aggregate compatible JSON metadata sidecars hierarchically."""
    records = tuple(sidecars)
    if not records:
        raise ValueError("At least one sidecar is required.")
    cohorts, cohort_hashes, trial_ids, trial_hashes = [], [], [], []
    for index, record in enumerate(records):
        cohort, cohort_hash = _validate_sidecar(record, index)
        cohorts.append(cohort)
        cohort_hashes.append(cohort_hash)
        trial_ids.append(record["comparison"]["trial_id"])
        trial_hashes.append(record["comparison"]["trial_fingerprint"])
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("Duplicate trial_id values cannot be aggregated.")
    if len(set(trial_hashes)) != len(trial_hashes):
        raise ValueError("Duplicate scientific trials cannot be aggregated.")
    if len(set(cohort_hashes)) != 1 or any(item != cohorts[0] for item in cohorts[1:]):
        raise ValueError("Direct-comparison sidecars belong to different cohorts.")

    ordered = sorted(records, key=lambda item: item["comparison"]["trial_id"])
    optimization_groups = defaultdict(list)
    data_groups = defaultdict(list)
    for record in ordered:
        comparison = record["comparison"]
        optimization_groups[comparison["paired_instance_sha256"]].append(record)
        data_groups[comparison["data_instance_sha256"]].append(record)

    for identity, group in optimization_groups.items():
        signatures = {
            _canonical_json(_printed_replicate_signature(record)) for record in group
        }
        if len(signatures) != 1:
            raise ValueError(
                "Deterministic printed-adaFGD results disagree across sampling "
                f"replicates for paired instance {identity}.")

    methods = {}
    for method in DIRECT_METHODS:
        endpoints = [_endpoint_values(record, method) for record in ordered]
        decisions = [record["comparison"]["method_gate"][method] for record in ordered]
        optimization_metrics = {}
        optimization_gate_rows = []
        for identity, group in sorted(optimization_groups.items()):
            metric_row = {}
            for metric in _ENDPOINT_METRICS:
                values = [_endpoint_values(record, method).get(metric) for record in group]
                if method == "printed-adafgd" and metric != "optimizer_seconds":
                    metric_row[metric] = _finite_number(values[0])
                else:
                    metric_row[metric] = _mean_or_none(values)
            optimization_metrics[identity] = metric_row
            group_decisions = [
                bool(record["comparison"]["method_gate"][method]["passed"])
                for record in group
            ]
            optimization_gate_rows.append({
                "paired_instance_sha256": identity,
                "data_instance_sha256": group[0]["comparison"]["data_instance_sha256"],
                "sampling_replicate_count": len(group),
                "pass_count": sum(group_decisions),
                "sampling_replicate_pass_rate": sum(group_decisions) / len(group),
                "all_sampling_replicates_pass": all(group_decisions),
                "any_sampling_replicate_passes": any(group_decisions),
            })

        optimization_gate_by_id = {
            row["paired_instance_sha256"]: row for row in optimization_gate_rows
        }
        data_metrics = {}
        data_gate_rows = []
        for identity, group in sorted(data_groups.items()):
            optimization_ids = sorted({
                record["comparison"]["paired_instance_sha256"] for record in group
            })
            data_metrics[identity] = {
                metric: _mean_or_none([
                    optimization_metrics[optimization_id][metric]
                    for optimization_id in optimization_ids
                ])
                for metric in _ENDPOINT_METRICS
            }
            optimization_pass_rates = [
                optimization_gate_by_id[optimization_id]["sampling_replicate_pass_rate"]
                for optimization_id in optimization_ids
            ]
            strict_flags = [
                optimization_gate_by_id[optimization_id][
                    "all_sampling_replicates_pass"]
                for optimization_id in optimization_ids
            ]
            data_gate_rows.append({
                "data_instance_sha256": identity,
                "optimization_instance_count": len(optimization_ids),
                "sampling_replicate_count": len(group),
                "initialization_sampling_pass_rates": [
                    {
                        "paired_instance_sha256": optimization_id,
                        "sampling_replicate_pass_rate": optimization_gate_by_id[
                            optimization_id]["sampling_replicate_pass_rate"],
                    }
                    for optimization_id in optimization_ids
                ],
                "nested_pass_rate": math.fsum(optimization_pass_rates) / len(
                    optimization_pass_rates),
                "optimization_instances_passing_all_sampling_replicates": sum(
                    strict_flags),
                "all_optimization_instances_and_sampling_replicates_pass": all(
                    strict_flags),
                "any_optimization_instance_passes_all_sampling_replicates": any(
                    strict_flags),
            })

        pass_count = sum(bool(item["passed"]) for item in decisions)
        cohort_nested_pass_rate = math.fsum(
            row["nested_pass_rate"] for row in data_gate_rows) / len(data_gate_rows)
        strict_optimization_pass_count = sum(
            row["all_sampling_replicates_pass"] for row in optimization_gate_rows)
        strict_data_pass_count = sum(
            row["all_optimization_instances_and_sampling_replicates_pass"]
            for row in data_gate_rows)
        methods[method] = {
            "sampling_trial_status_counts_descriptive": dict(sorted(Counter(
                str(endpoint.get("status")) for endpoint in endpoints).items())),
            "sampling_trial_gate_descriptive": {
                "pass_count": pass_count,
                "pass_rate": pass_count / len(ordered),
            },
            "sampling_trial_metrics_descriptive": {
                metric: _statistics([item.get(metric) for item in endpoints], len(ordered))
                for metric in _ENDPOINT_METRICS
            },
            "optimization_instance_mean_metrics": {
                metric: _statistics(
                    [row[metric] for row in optimization_metrics.values()],
                    len(optimization_groups))
                for metric in _ENDPOINT_METRICS
            },
            "optimization_instance_gate": {
                "primary_estimand": "sampling_replicate_pass_rate",
                "instances": optimization_gate_rows,
                "secondary_all_pass_robustness": {
                    "criterion": "all_sampling_replicates_pass",
                    "count": strict_optimization_pass_count,
                    "rate": strict_optimization_pass_count / len(optimization_groups),
                },
            },
            "data_instance_mean_metrics": {
                metric: _statistics(
                    [row[metric] for row in data_metrics.values()], len(data_groups))
                for metric in _ENDPOINT_METRICS
            },
            "data_instance_gate": {
                "primary_estimand": (
                    "mean sampling pass rate within initialization; equal-weight mean "
                    "across initializations within data instance; equal-weight mean "
                    "across data instances"
                ),
                "cohort_nested_pass_rate": cohort_nested_pass_rate,
                "data_instance_count": len(data_groups),
                "instances": data_gate_rows,
                "secondary_all_pass_robustness": {
                    "criterion": (
                        "all optimization instances and sampling replicates pass"
                    ),
                    "count": strict_data_pass_count,
                    "rate": strict_data_pass_count / len(data_groups),
                },
            },
        }

    sampling_differences = {metric: [] for metric in _ENDPOINT_METRICS}
    trials = []
    for record in ordered:
        comparison = record["comparison"]
        differences = _endpoint_differences(record)["values"]
        for metric, value in differences.items():
            sampling_differences[metric].append(value)
        trials.append({
            "trial_id": comparison["trial_id"],
            "trial_fingerprint": comparison["trial_fingerprint"],
            "data_instance_sha256": comparison["data_instance_sha256"],
            "paired_instance_sha256": comparison["paired_instance_sha256"],
            "sampling_seed": record["configuration"]["sampling_seed"],
            "methods": {
                method: {"endpoint": _endpoint_values(record, method),
                         "gate": comparison["method_gate"][method]}
                for method in DIRECT_METHODS
            },
            "frames_minus_printed_adafgd": differences,
        })

    optimization_differences = {}
    for identity, group in sorted(optimization_groups.items()):
        optimization_differences[identity] = {
            metric: _mean_or_none([
                _endpoint_differences(record)["values"][metric] for record in group
            ])
            for metric in _ENDPOINT_METRICS
        }
    data_differences = {}
    for identity, group in sorted(data_groups.items()):
        optimization_ids = sorted({
            record["comparison"]["paired_instance_sha256"] for record in group
        })
        data_differences[identity] = {
            metric: _mean_or_none([
                optimization_differences[optimization_id][metric]
                for optimization_id in optimization_ids
            ])
            for metric in _ENDPOINT_METRICS
        }

    data_instances = []
    for identity, group in sorted(data_groups.items()):
        optimization_ids = sorted({
            record["comparison"]["paired_instance_sha256"] for record in group
        })
        grouped_optimization_instances = []
        for optimization_id in optimization_ids:
            replicates = optimization_groups[optimization_id]
            grouped_optimization_instances.append({
                "paired_instance_sha256": optimization_id,
                "initial_factor_sha256": replicates[0]["initial_factor_sha256"],
                "sampling_replicate_count": len(replicates),
                "trial_ids": sorted(
                    record["comparison"]["trial_id"] for record in replicates),
            })
        data_instances.append({
            "data_instance_sha256": identity,
            "data_instance": group[0]["comparison"]["data_instance"],
            "optimization_instance_count": len(grouped_optimization_instances),
            "sampling_replicate_count": len(group),
            "optimization_instances": grouped_optimization_instances,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "format": SUMMARY_FORMAT,
        "protocol": PROTOCOL,
        "cohort_fingerprint": cohort_hashes[0],
        "cohort_spec": cohorts[0],
        "validation_scope": (
            "JSON metadata sidecars only; referenced NPZ archives and factor arrays "
            "are not revalidated by this aggregator"
        ),
        "timing_interpretation": (
            "descriptive fixed-order optimizer timings; not a causal speed comparison"
        ),
        "primary_analysis_unit": "data_instance",
        "gate_primary_estimand": (
            "nested Boolean mean: sampling replicates within initialization, then "
            "initializations within data instance, then data instances within cohort"
        ),
        "data_instance_count": len(data_groups),
        "optimization_instance_count": len(optimization_groups),
        "sampling_trial_count": len(ordered),
        "analysis_unit_note": (
            "Sampling-seed trials are nested within data/initialization optimization "
            "instances, which are nested within physical data instances. Primary metric, "
            "gate, and difference summaries weight data instances equally; initializations "
            "are first weighted equally within data instances."
        ),
        "deterministic_baseline_aggregation": (
            "printed-adaFGD scientific endpoints are collapsed per optimization instance "
            "and must agree across FRAMES sampling replicates; repeated timings are averaged "
            "only as descriptive timing measurements"
        ),
        "data_instances": data_instances,
        "methods": methods,
        "paired_differences": {
            "orientation": "frames_minus_printed_adafgd",
            "primary_analysis_unit": "data_instance",
            "sampling_trial_metrics_descriptive": {
                metric: _statistics(values, len(ordered))
                for metric, values in sampling_differences.items()
            },
            "optimization_instance_mean_metrics": {
                metric: _statistics(
                    [row[metric] for row in optimization_differences.values()],
                    len(optimization_groups))
                for metric in _ENDPOINT_METRICS
            },
            "data_instance_mean_metrics": {
                metric: _statistics(
                    [row[metric] for row in data_differences.values()], len(data_groups))
                for metric in _ENDPOINT_METRICS
            },
        },
        "trials": trials,
    }


def load_and_aggregate_sidecars(paths):
    """Load JSON files and delegate to the pure aggregation function."""
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    return aggregate_sidecars(records)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--row-epochs", type=int, required=True)
    parser.add_argument("--checkpoint-epochs", type=int, required=True)
    parser.add_argument("--tp-weight-per-row", type=float, required=True,
                        help="Alpha in the common mean-loss objective.")
    parser.add_argument("--eta-scale", type=float, required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--channel-seed", type=int, default=0)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--observation-mode", choices=("noiseless", "gaussian", "shots"),
                        default="noiseless")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--shot-seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--spectral-norm-method", choices=("auto", "dense", "matrix-free"),
                        default="auto")
    parser.add_argument("--dense-max-process-dimension", type=int, default=1024)
    parser.add_argument("--spectral-norm-tolerance", type=float, default=1e-10)
    parser.add_argument("--spectral-norm-maxiter", type=int, default=None)
    parser.add_argument("--rho-scale", type=float, default=2.0)
    parser.add_argument("--rho-offset", type=float, default=4.0)
    parser.add_argument("--rho-exponent", type=float, default=0.6)
    parser.add_argument("--step-scale", type=float, default=2.0)
    parser.add_argument("--step-offset", type=float, default=2.0)
    parser.add_argument("--step-exponent", type=float, default=1.0)
    parser.add_argument("--gate-name", default="noiseless_recovery_v1")
    parser.add_argument("--gate-fidelity-min", type=float, default=0.99)
    parser.add_argument("--gate-rmse-max", type=float, default=1e-6)
    parser.add_argument("--gate-normalized-tp-max", type=float, default=1e-3)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--cohort-id", default=None)
    parser.add_argument("--split", choices=("calibration", "heldout", "unspecified"),
                        default="unspecified")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-on-method-failure", action="store_true")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    metadata_path = args.metadata if args.metadata is not None else args.save.with_suffix(".json")
    if args.save.resolve() == metadata_path.resolve():
        parser.error("--save and --metadata must be different paths")
    if args.save.exists() or metadata_path.exists():
        parser.error("refusing to overwrite an existing NPZ or JSON result")
    try:
        data = generate_quiroga_sensing_data(
            args.n_qubits, channel_seed=args.channel_seed,
            observation_mode=args.observation_mode, noise_std=args.noise_std,
            noise_seed=args.noise_seed, shots=args.shots, shot_seed=args.shot_seed)
        if args.subset_size < 0:
            raise ValueError("subset_size must be nonnegative.")
        rows = (None if args.subset_size == 0
                else data.fixed_row_subset(args.subset_size, args.subset_seed))
        rho = PowerSchedule(args.rho_scale, args.rho_offset, args.rho_exponent, cap=1.0)
        step = PowerSchedule(args.step_scale, args.step_offset, args.step_exponent, cap=1.0)
        gate = RecoveryGate(args.gate_name, args.gate_fidelity_min,
                            args.gate_rmse_max, args.gate_normalized_tp_max)
        metadata, payload, traces = run_direct_comparison(
            data, rows=rows, row_epochs=args.row_epochs,
            checkpoint_epochs=args.checkpoint_epochs,
            tp_weight_per_row=args.tp_weight_per_row, eta_scale=args.eta_scale,
            rank=args.rank, initialization_seed=args.initialization_seed,
            sampling_seed=args.sampling_seed, tau=args.tau,
            batch_size=args.batch_size, spectral_norm_method=args.spectral_norm_method,
            dense_max_process_dimension=args.dense_max_process_dimension,
            spectral_norm_tolerance=args.spectral_norm_tolerance,
            spectral_norm_maxiter=args.spectral_norm_maxiter,
            rho_schedule=rho, step_size_schedule=step, gate=gate,
            trial_id=args.trial_id, cohort_id=args.cohort_id, split=args.split,
            expected_commit=args.expected_commit, require_clean=args.require_clean,
            show_progress=not args.quiet)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    metadata["configuration"].update({
        "subset_size": int(args.subset_size), "subset_seed": int(args.subset_seed),
        "channel_seed": int(args.channel_seed),
    })
    save_campaign(args.save, metadata_path, metadata, payload)
    if not args.quiet:
        comparison = metadata["comparison"]
        print(json.dumps({
            "trial_id": comparison["trial_id"],
            "paired_instance_sha256": comparison["paired_instance_sha256"],
            "shared_penalty": comparison["shared_penalty"],
            "budget": comparison["budget"],
            "methods": metadata["methods"],
            "method_gate": comparison["method_gate"],
        }, indent=2, sort_keys=True, allow_nan=False), flush=True)
        print(f"Saved {args.save} and {metadata_path}", flush=True)
    if (args.fail_on_method_failure
            and any(trace.status == "failed" for trace in traces.values())):
        raise SystemExit(2)
    return metadata, payload, traces


if __name__ == "__main__":
    main()
