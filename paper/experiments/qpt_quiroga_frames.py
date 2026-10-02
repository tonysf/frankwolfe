"""FRAMES and adaFGD hooks for the Quiroga-style sensing backend.

``QuirogaSensingObjective`` has the same interface and minibatch RNG usage as
``quantum_process_tomography.QPTMeasurementObjective``: packed-real factor
coordinates, mean half-squared loss over the active rows, and the nonlinear
trace-preserving map ``T(U) = Tr_out(U U^H)`` supplied as FRAMES's composite
map through its current-point Jacobian adjoint.  Unlike the dense runner, the
runner below never forms the ``d**2``-by-``d**2`` process matrix, so its
checkpoint metrics scale with the sensing backend.

For small systems, ``QuirogaSensingData.to_dense_qpt_data`` instead feeds the
existing dense NumPy/JAX runners without modifying them.

``quiroga_adafgd_step`` implements the adaptive update printed in the paper.
For small systems it computes the numerator with a dense Hermitian eigensolve;
at larger sizes it applies the same Hermitian operator matrix-free and uses a
tolerance-controlled Lanczos solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from frank_wolfe import ObjectiveFunction, StochasticFrames

from .quantum_process_tomography import (
    _CheckpointRecorder,
    _checkpoint_smoothing_parameters,
    checkpoint_iterations,
    create_operator_norm_factor_lmo,
    create_trace_preserving_prox,
    make_factor_initial_point,
    pack_factor,
    unpack_factor,
)
from .qpt_quiroga_sensing import (
    QuirogaSensingData,
    _positive_integer,
    choi_output_partial_trace,
    trace_preserving_jacobian_adjoint,
    trace_preserving_loss_and_gradient,
    trace_preserving_residual,
)


def _active_rows_and_targets(data, rows, observations):
    if not isinstance(data, QuirogaSensingData):
        raise TypeError("data must be QuirogaSensingData.")
    rows = np.arange(data.m, dtype=np.int64) if rows is None else data.design.validate_rows(rows)
    if observations is None:
        every_row = rows.size == data.m and np.array_equal(rows, np.arange(data.m))
        targets = data.all_observations() if every_row else data.observations_for_rows(rows)
    else:
        targets = np.asarray(observations)
        if (targets.shape != rows.shape or targets.dtype.kind not in "iuf"
                or not np.all(np.isfinite(targets))):
            raise ValueError("observations must be finite real values matching rows.")
    return rows, np.asarray(targets, dtype=np.float64)


class QuirogaSensingObjective(ObjectiveFunction):
    """Nonconvex factorized loss on the Quiroga-style rows, for StochasticFrames.

    ``rows`` selects the active sensing rows (all ``2 d**3`` by default, or,
    for example, ``data.fixed_row_subset(size, seed)`` for Quiroga's
    underdetermined setting).  ``observations`` overrides their targets.
    Minibatches draw ``rng.integers(0, len(rows), batch_size)`` with
    replacement, exactly like the legacy objective.
    """

    def __init__(self, data, rank=1, batch_size=1, seed=None, rows=None, observations=None):
        super().__init__()
        self.rank = _positive_integer(rank, "rank")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.data = data
        self.rows, self.targets = _active_rows_and_targets(data, rows, observations)
        self.design = data.design
        self.m = self.rows.size
        self._all_rows = self.m == data.m and np.array_equal(self.rows, np.arange(data.m))
        self.rng = np.random.default_rng(seed)
        self.last_batch_indices = None
        self.sampled_measurements = 0

    def unpack(self, vector):
        return unpack_factor(vector, self.data.process_dimension, self.rank)

    def _indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("indices must be a nonempty one-dimensional array.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("measurement indices must be integers.")
        if np.any(indices < 0) or np.any(indices >= self.m):
            raise IndexError("measurement index is out of range.")
        return indices.astype(np.int64, copy=False)

    def sensing_values(self, vector, indices=None):
        factor = self.unpack(vector)
        if indices is None:
            return self.design.full_values(factor)[self.rows]
        return self.design.values(factor, self.rows[self._indices(indices)])

    def gradient_for_indices(self, vector, indices):
        """Return an unbiased packed-real gradient of the mean active loss."""
        indices = self._indices(indices)
        _, gradient = self.design.loss_and_gradient(
            self.unpack(vector), self.rows[indices], self.targets[indices], reduction="mean")
        return pack_factor(gradient)

    def loss_and_gradient(self, vector):
        """Exact mean loss and packed gradient over every active row, O(r d**3)."""
        factor = self.unpack(vector)
        if self._all_rows:
            loss, gradient = self.design.full_loss_and_gradient(factor, self.targets, reduction="mean")
        else:
            residual = self.design.full_values(factor)[self.rows] - self.targets
            loss = 0.5 * np.mean(residual**2)
            weights = np.bincount(self.rows, weights=(2.0 / self.m) * residual, minlength=self.data.m)
            gradient = self.design.full_adjoint(factor, weights)
        return float(loss), pack_factor(gradient)

    def evaluate(self, vector):
        return self.loss_and_gradient(vector)[0]

    def gradient(self, vector):
        return self.loss_and_gradient(vector)[1]

    def stochastic_gradient(self, vector):
        indices = self.rng.integers(0, self.m, size=self.batch_size)
        self.last_batch_indices = indices.copy()
        self.sampled_measurements += self.batch_size
        return self.gradient_for_indices(vector, indices)

    def linear_operator(self, vector):
        return choi_output_partial_trace(self.unpack(vector))

    def linear_operator_adjoint_at(self, vector, matrix):
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != (self.data.d, self.data.d):
            raise ValueError(f"the trace-preserving residual must have shape {(self.data.d, self.data.d)}.")
        return pack_factor(trace_preserving_jacobian_adjoint(self.unpack(vector), matrix))

    def trace_preserving_residual(self, vector):
        return trace_preserving_residual(self.unpack(vector))


@dataclass
class QuirogaFramesResult:
    """Checkpoint metrics of a structured stochastic-FRAMES run (no dense chi)."""

    rank: int
    tau: float
    final_x: np.ndarray
    final_factor: np.ndarray
    checkpoint_steps: np.ndarray
    optimizer_seconds: np.ndarray
    measurement_loss: np.ndarray
    tp_violation: np.ndarray
    smoothed_objective: np.ndarray
    process_fidelity: np.ndarray
    exact_smoothed_gap: np.ndarray
    checkpoint_smoothing_parameters: np.ndarray
    estimated_gaps: np.ndarray
    momentum_weights: np.ndarray
    smoothing_parameters: np.ndarray
    step_sizes: np.ndarray
    cumulative_stochastic_oracles: np.ndarray
    cumulative_sampled_measurements: np.ndarray
    metadata: dict
    algorithm: StochasticFrames


def run_quiroga_sensing_stochastic_frames(
    data,
    *,
    n_steps=1000,
    rank=1,
    tau=10.0,
    batch_size=1,
    initialization_seed=0,
    sampling_seed=0,
    x0=None,
    beta0=10.0,
    rho_schedule=None,
    smoothing_schedule=None,
    step_size_schedule=None,
    metrics_frequency=100,
    show_progress=True,
    rows=None,
    observations=None,
):
    """Run stochastic FRAMES with the same conventions as the dense runner.

    Defaults, initialization, LMO, prox, schedules and checkpoint alignment
    are those of ``run_qpt_stochastic_frames``.  Checkpoint losses and gaps
    are exact over the active rows; the fidelity is NaN without a rank-one
    truth.  Optimizer time excludes the post-hoc checkpoint metrics.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    expected_checkpoints = checkpoint_iterations(n_steps, metrics_frequency)
    objective = QuirogaSensingObjective(data, rank=rank, batch_size=batch_size, seed=sampling_seed,
                                        rows=rows, observations=observations)
    lmo = create_operator_norm_factor_lmo(data.process_dimension, objective.rank, tau)
    prox = create_trace_preserving_prox(data.d)
    x0_provided = x0 is not None
    if x0 is None:
        x0 = make_factor_initial_point(data, rank=objective.rank, seed=initialization_seed)
    x0 = np.asarray(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain finite values.")
    if np.linalg.norm(objective.unpack(x0), ord=2) > tau + 1e-10:
        raise ValueError("x0 must satisfy the operator-norm constraint.")

    recorder = _CheckpointRecorder(n_steps, metrics_frequency)
    algorithm = StochasticFrames(objective, lmo, prox, objective_type="indicator")
    recorder.start()
    algorithm.run(
        x0, beta0=beta0, n_steps=n_steps, show_progress=show_progress,
        rho_schedule=rho_schedule, smoothing_schedule=smoothing_schedule,
        step_size_schedule=step_size_schedule, evaluate_objective=False,
        iterate_callback=recorder, iterate_callback_frequency=metrics_frequency,
    )
    checkpoint_steps = np.asarray(recorder.steps, dtype=int)
    if not np.array_equal(checkpoint_steps, expected_checkpoints):
        raise RuntimeError("the optimizer did not emit the expected checkpoints.")
    checkpoint_beta = _checkpoint_smoothing_parameters(checkpoint_steps, algorithm.smoothing_parameters)

    rank_one_truth = data.truth_factor is not None and data.truth_factor.shape[1] == 1
    rows_out = {name: [] for name in ("loss", "violation", "smoothed", "fidelity", "gap")}
    for vector, beta in zip(recorder.iterates, checkpoint_beta):
        loss, smooth_gradient = objective.loss_and_gradient(vector)
        tp_residual = objective.trace_preserving_residual(vector)
        violation = np.linalg.norm(tp_residual, ord="fro")
        moreau_gradient = objective.linear_operator_adjoint_at(vector, tp_residual) / beta
        combined_gradient = smooth_gradient + moreau_gradient
        atom = lmo(combined_gradient)
        rows_out["loss"].append(loss)
        rows_out["violation"].append(violation)
        rows_out["smoothed"].append(loss + 0.5 * violation**2 / beta)
        rows_out["fidelity"].append(data.fidelity(objective.unpack(vector)) if rank_one_truth else np.nan)
        rows_out["gap"].append(np.dot(combined_gradient, vector - atom))

    final_factor = objective.unpack(algorithm.x)
    metadata = {
        "formulation": "nonconvex_factor_nonlinear_tp_smoothing",
        "nonlinear_composite": "jacobian_adjoint_as_linear",
        "sensing_backend": "quiroga_global_probes_2d_povm", "design": data.design.describe(),
        "observation_mode": data.observation_mode, "active_rows": int(objective.m),
        "all_rows_active": bool(objective._all_rows), "batch_size": int(objective.batch_size),
        "initialization_seed": None if x0_provided else int(initialization_seed),
        "sampling_seed": sampling_seed, "loss_normalization": "mean_half_squared_over_active_rows",
    }
    return QuirogaFramesResult(
        rank=objective.rank, tau=float(tau), final_x=algorithm.x.copy(), final_factor=final_factor,
        checkpoint_steps=checkpoint_steps, optimizer_seconds=np.asarray(recorder.times),
        measurement_loss=np.asarray(rows_out["loss"]), tp_violation=np.asarray(rows_out["violation"]),
        smoothed_objective=np.asarray(rows_out["smoothed"]),
        process_fidelity=np.asarray(rows_out["fidelity"]),
        exact_smoothed_gap=np.asarray(rows_out["gap"]),
        checkpoint_smoothing_parameters=checkpoint_beta.copy(),
        estimated_gaps=algorithm.estimated_gaps.copy(), momentum_weights=algorithm.momentum_weights.copy(),
        smoothing_parameters=algorithm.smoothing_parameters.copy(), step_sizes=algorithm.step_sizes.copy(),
        cumulative_stochastic_oracles=algorithm.num_stochastic_oracles.copy(),
        cumulative_sampled_measurements=np.rint(batch_size * algorithm.num_stochastic_oracles).astype(int),
        metadata=metadata, algorithm=algorithm,
    )


def _adafgd_spectral_norm(design, weights, *, method="auto", dense_max_process_dimension=1024,
                          tolerance=1e-10, maxiter=None, return_diagnostics=False):
    """Numerical spectral norm of the Hermitian adjoint ``A^H(weights)``.

    ``dense`` is the small-system reference.  ``matrix-free`` uses the exact
    :meth:`QuirogaSensingDesign.full_adjoint` matrix-vector action with ARPACK;
    only the eigensolve is tolerance-based.  ``auto`` selects the dense oracle
    up to ``dense_max_process_dimension`` and the matrix-free path above it.
    """
    if method not in ("auto", "dense", "matrix-free"):
        raise ValueError("spectral_norm_method must be 'auto', 'dense', or 'matrix-free'.")
    dense_max_process_dimension = _positive_integer(
        dense_max_process_dimension, "dense_max_process_dimension")
    if (isinstance(tolerance, (bool, np.bool_)) or np.ndim(tolerance) != 0
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("spectral_norm_tolerance must be finite and nonnegative.")
    tolerance = float(tolerance)
    if maxiter is not None:
        maxiter = _positive_integer(maxiter, "spectral_norm_maxiter")

    selected = method
    if selected == "auto":
        selected = ("dense" if design.process_dimension <= dense_max_process_dimension
                    else "matrix-free")
    began = perf_counter()
    if selected == "dense":
        matrix = design.dense_adjoint(
            weights, max_process_dimension=dense_max_process_dimension)
        value = float(np.max(np.abs(np.linalg.eigvalsh(matrix))))
        diagnostics = {
            "seconds": perf_counter() - began,
            "operator_calls": 1,
            "adjoint_column_equivalent": design.process_dimension,
        }
        return ((value, selected, diagnostics) if return_diagnostics
                else (value, selected))

    size = design.process_dimension
    weights = np.asarray(weights, dtype=np.float64)
    operator_calls = 0

    def matvec(vector):
        nonlocal operator_calls
        operator_calls += 1
        column = np.asarray(vector, dtype=np.complex128).reshape(size, 1)
        return np.asarray(design.full_adjoint(column, weights))[:, 0]

    operator = LinearOperator(
        (size, size), matvec=matvec, rmatvec=matvec, dtype=np.complex128)
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    v0 /= np.linalg.norm(v0)
    try:
        eigenvalue = eigsh(
            operator, k=1, which="LM", tol=tolerance, maxiter=maxiter,
            v0=v0, return_eigenvectors=False,
        )[0]
    except ArpackNoConvergence as error:
        raise RuntimeError(
            "Matrix-free spectral-norm eigensolve did not converge; increase "
            "spectral_norm_maxiter or relax spectral_norm_tolerance."
        ) from error
    value = float(abs(eigenvalue))
    if not np.isfinite(value):
        raise FloatingPointError("Matrix-free spectral-norm eigensolve returned a nonfinite value.")
    diagnostics = {
        "seconds": perf_counter() - began,
        "operator_calls": operator_calls,
        "adjoint_column_equivalent": operator_calls,
    }
    return ((value, selected, diagnostics) if return_diagnostics
            else (value, selected))


def quiroga_adafgd_step(
    data,
    factor,
    *,
    eta_scale,
    tp_weight,
    rows=None,
    observations=None,
    spectral_norm_method="auto",
    dense_max_process_dimension=1024,
    spectral_norm_tolerance=1e-10,
    spectral_norm_maxiter=None,
):
    """One printed-rule adaFGD update of Quiroga and Kyrillidis.

    ``U+ = U - eta (A^H(A(UU^H) - f) U + lambda grad_chi H(UU^H) U)`` with the
    summed half-squared loss, ``H = ||Tr_out(UU^H) - I||_F**2`` (not halved)
    and ``eta = eta_scale ||A^H(A(UU^H) - f)||_2 / ||A(UU^H)||_2``.  The paper
    reports neither ``eta_scale`` nor ``lambda = tp_weight``.

    This is the rule printed in arXiv:2312.01311, not the exact line-search
    rule in its cited reference [47].  It is not scale-covariant and needs
    calibration for each sensing design.

    ``spectral_norm_method='auto'`` uses a dense small-system oracle up to
    ``dense_max_process_dimension`` and a matrix-free, tolerance-controlled
    Hermitian eigensolve above it.  The matrix-vector action and both factor
    terms are exact structured operators.  Pass ``observations`` when looping
    so targets are not regenerated at every update.
    """
    for value, name in ((eta_scale, "eta_scale"), (tp_weight, "tp_weight")):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    rows, targets = _active_rows_and_targets(data, rows, observations)
    factor = np.asarray(factor, dtype=np.complex128)
    design = data.design
    predicted = design.full_values(factor)[rows]
    residual = predicted - targets
    weights = np.bincount(rows, weights=residual, minlength=data.m)
    numerator, spectral_norm_method_used, spectral_norm_diagnostics = _adafgd_spectral_norm(
        design, weights, method=spectral_norm_method,
        dense_max_process_dimension=dense_max_process_dimension,
        tolerance=spectral_norm_tolerance, maxiter=spectral_norm_maxiter,
        return_diagnostics=True)
    denominator = float(np.linalg.norm(predicted))
    denominator_floor = np.finfo(np.float64).eps * max(
        1.0, float(np.linalg.norm(factor, ord="fro") ** 2))
    if not np.isfinite(denominator) or denominator <= denominator_floor:
        raise FloatingPointError(
            "The adaFGD prediction-norm denominator is zero or numerically negligible; "
            "the printed adaptive step is undefined for this iterate and active row set.")
    eta = eta_scale * numerator / denominator
    measurement_term = design.full_adjoint(factor, weights)
    tp_half_loss, tp_term = trace_preserving_loss_and_gradient(factor)
    updated = factor - eta * (measurement_term + tp_weight * tp_term)
    return updated, {
        "eta": eta, "spectral_norm_numerator": numerator,
        "spectral_norm_method": spectral_norm_method_used,
        "spectral_norm_tolerance": float(spectral_norm_tolerance),
        "spectral_norm_seconds": spectral_norm_diagnostics["seconds"],
        "spectral_norm_operator_calls": spectral_norm_diagnostics["operator_calls"],
        "spectral_norm_adjoint_column_equivalent": spectral_norm_diagnostics[
            "adjoint_column_equivalent"],
        "prediction_norm_denominator": denominator,
        "measurement_loss_sum": float(0.5 * np.sum(residual**2)),
        "tp_penalty_unhalved": float(2.0 * tp_half_loss),
    }
