"""Factorized QPT_BFW compatibility runner for stochastic FRAMES.

This module keeps QPT_BFW's original decision variable ``U``, process matrix
``chi = U U^H``, operator-norm constraint, and smooth trace-preservation
penalty.  It replaces the deterministic full-measurement gradient with a
uniform measurement minibatch and runs :class:`StochasticFrames`.

Because the original objective is entirely smooth, the composite term is
``g = 0`` and its proximal map is the identity.  The smoothing schedule is
still accepted and recorded for API parity, but has no effect on this
compatibility formulation.  Use ``quantum_process_tomography.py`` for the
nonconvex factor-space variant in which the smoothing schedule actively
enforces trace preservation through the nonlinear map's Jacobian adjoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from frank_wolfe import ObjectiveFunction, StochasticFrames
from paper.experiments.quantum_process_tomography import (
    PowerSchedule,
    QPTData,
    Schedule,
    checkpoint_iterations,
    process_fidelity_proxy,
)


def pack_factor(matrix):
    """Pack a complex factor into its real and imaginary coordinates."""

    matrix = np.asarray(matrix)
    return np.concatenate([matrix.real.ravel(), matrix.imag.ravel()])


def unpack_factor(vector, process_dimension, rank):
    """Inverse of :func:`pack_factor` with strict shape validation."""

    vector = np.asarray(vector, dtype=float)
    expected_size = 2 * process_dimension * rank
    if vector.ndim != 1 or vector.size != expected_size:
        raise ValueError(
            f"the packed factor must have shape ({expected_size},); got "
            f"{vector.shape}."
        )
    half = expected_size // 2
    return vector[:half].reshape(process_dimension, rank) + 1j * vector[
        half:
    ].reshape(process_dimension, rank)


def identity_prox(value, beta):
    """Proximal map of the zero function; ``beta`` is intentionally inert."""

    del beta
    return np.asarray(value)


class QPTFactorObjective(ObjectiveFunction):
    """Original smooth QPT_BFW objective with an unbiased minibatch oracle."""

    def __init__(self, data, rank=1, lam=0.05, batch_size=1, seed=None):
        super().__init__()
        if not isinstance(rank, (int, np.integer)) or rank <= 0:
            raise ValueError("rank must be a positive integer.")
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lam must be a nonnegative finite number.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.data = data
        self.rank = int(rank)
        self.lam = float(lam)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.last_batch_indices = None
        self.sampled_measurements = 0

    def unpack(self, vector):
        return unpack_factor(vector, self.data.process_dimension, self.rank)

    def process_matrix(self, vector):
        factor = self.unpack(vector)
        return factor @ factor.conj().T

    def _sensing_values(self, factor, indices=None):
        tensors = (
            self.data.D_tensors
            if indices is None
            else self.data.D_tensors[indices]
        )
        chi = factor @ factor.conj().T
        return np.einsum(
            "mab,ab->m", tensors.conj(), chi, optimize=True
        ).real

    def trace_preserving_residual(self, factor):
        chi = factor @ factor.conj().T
        mapped = np.einsum(
            "nm,nmij->ij", chi, self.data.B_tensors, optimize=True
        )
        return mapped - np.eye(self.data.d, dtype=np.complex128)

    def _tp_wirtinger_gradient(self, factor, residual=None):
        if residual is None:
            residual = self.trace_preserving_residual(factor)
        chi_gradient = 2.0 * np.einsum(
            "nmij,ij->nm",
            self.data.B_tensors.conj(),
            residual,
            optimize=True,
        )
        return chi_gradient @ factor

    def loss_components(self, vector):
        factor = self.unpack(vector)
        measurement_residual = (
            self._sensing_values(factor) - self.data.f_vector
        )
        measurement_loss = 0.5 * np.mean(measurement_residual**2)
        tp_residual = self.trace_preserving_residual(factor)
        tp_penalty = np.linalg.norm(tp_residual, ord="fro") ** 2
        return float(measurement_loss), float(tp_penalty)

    def evaluate(self, vector):
        measurement_loss, tp_penalty = self.loss_components(vector)
        return measurement_loss + self.lam * tp_penalty

    def _gradient_for_indices(self, vector, indices):
        factor = self.unpack(vector)
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("indices must be a nonempty one-dimensional array.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("measurement indices must be integers.")
        indices = indices.astype(int, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.data.m):
            raise IndexError("measurement index is out of range.")

        sensing = self._sensing_values(factor, indices=indices)
        residual = sensing - self.data.f_vector[indices]
        chi_gradient = np.einsum(
            "m,mab->ab",
            residual,
            self.data.D_tensors[indices],
            optimize=True,
        ) / indices.size
        measurement_wirtinger = chi_gradient @ factor
        tp_wirtinger = self._tp_wirtinger_gradient(factor)

        # QPT_BFW stores the Wirtinger derivative. Multiplying by two gives
        # the true gradient in the packed real coordinates. This common scale
        # leaves every LMO atom and optimizer trajectory unchanged.
        return 2.0 * pack_factor(
            measurement_wirtinger + self.lam * tp_wirtinger
        )

    def gradient_for_indices(self, vector, indices):
        return self._gradient_for_indices(vector, indices)

    def gradient(self, vector):
        return self._gradient_for_indices(vector, np.arange(self.data.m))

    def stochastic_gradient(self, vector):
        indices = self.rng.integers(
            0,
            self.data.m,
            size=self.batch_size,
        )
        self.last_batch_indices = indices.copy()
        self.sampled_measurements += self.batch_size
        return self._gradient_for_indices(vector, indices)

    def linear_operator(self, vector):
        return np.asarray(vector)

    def linear_operator_adjoint(self, vector):
        return np.asarray(vector)


def create_operator_norm_factor_lmo(process_dimension, rank, tau):
    """Exact thin-SVD LMO for ``{U: ||U||_op <= tau}``."""

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number.")

    def lmo(packed_gradient):
        gradient = unpack_factor(packed_gradient, process_dimension, rank)
        if np.linalg.norm(gradient, ord="fro") == 0:
            return np.zeros_like(packed_gradient, dtype=float)
        left, _, right_h = np.linalg.svd(gradient, full_matrices=False)
        return pack_factor(-tau * (left @ right_h))

    return lmo


def make_factor_initial_point(data, rank=1, seed=0):
    """Match QPT_BFW's trace-normalized random complex initialization."""

    rng = np.random.default_rng(seed)
    factor = rng.uniform(0.0, 1.0, (data.process_dimension, rank))
    factor = factor + 1j * rng.uniform(
        0.0, 1.0, (data.process_dimension, rank)
    )
    trace = np.linalg.norm(factor, ord="fro") ** 2
    factor *= np.sqrt(data.d / trace)
    return pack_factor(factor)


class _FactorCheckpointRecorder:
    def __init__(self, n_steps, frequency):
        self.requested_steps = set(
            checkpoint_iterations(n_steps, frequency).tolist()
        )
        self.steps = []
        self.seconds = []
        self.iterates = []
        self.started_at = None

    def start(self):
        self.started_at = perf_counter()

    def __call__(self, completed_steps, vector):
        if completed_steps not in self.requested_steps:
            return
        if self.started_at is None:
            raise RuntimeError("the checkpoint recorder clock was not started.")
        self.steps.append(completed_steps)
        self.seconds.append(
            0.0
            if completed_steps == 0
            else perf_counter() - self.started_at
        )
        self.iterates.append(np.asarray(vector))


@dataclass
class QPTFactorExperimentResult:
    rank: int
    tau: float
    lam: float
    batch_size: int
    initialization_seed: int
    sampling_seed: int
    final_x: np.ndarray
    final_factor: np.ndarray
    final_chi: np.ndarray
    checkpoint_steps: np.ndarray
    optimizer_seconds: np.ndarray
    measurement_loss: np.ndarray
    tp_violation: np.ndarray
    objective_value: np.ndarray
    process_fidelity_proxy: np.ndarray
    exact_fw_gap: np.ndarray
    qpt_bfw_exact_gap: np.ndarray
    estimated_gaps: np.ndarray
    qpt_bfw_estimated_gaps: np.ndarray
    momentum_weights: np.ndarray
    smoothing_parameters: np.ndarray
    step_sizes: np.ndarray
    cumulative_stochastic_oracles: np.ndarray
    cumulative_sampled_measurements: np.ndarray
    algorithm: StochasticFrames


def run_qpt_factor_stochastic_frames(
    data,
    *,
    n_steps=1000,
    rank=1,
    tau=10.0,
    lam=0.05,
    batch_size=1,
    initialization_seed=0,
    sampling_seed=0,
    x0=None,
    beta0=1.0,
    rho_schedule: Schedule = None,
    smoothing_schedule: Schedule = None,
    step_size_schedule: Schedule = None,
    metrics_frequency=100,
    show_progress=True,
):
    """Run momentum stochastic FW on the original factorized QPT objective.

    ``smoothing_schedule`` is forwarded and recorded but does not affect the
    iterates because the zero composite term has an identity proximal map.
    Full objective, fidelity, and exact-gradient gaps are evaluated after the
    timed run at sparse checkpoints.  Timings include progress reporting and
    the sparse defensive checkpoint copies, but exclude full post-hoc metric
    passes.
    """

    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    expected_checkpoints = checkpoint_iterations(n_steps, metrics_frequency)
    objective = QPTFactorObjective(
        data,
        rank=rank,
        lam=lam,
        batch_size=batch_size,
        seed=sampling_seed,
    )
    lmo = create_operator_norm_factor_lmo(
        data.process_dimension, objective.rank, tau
    )
    if x0 is None:
        x0 = make_factor_initial_point(
            data, rank=objective.rank, seed=initialization_seed
        )
    x0 = np.asarray(x0, dtype=float)
    factor0 = objective.unpack(x0)
    if not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain finite values.")
    if np.linalg.norm(factor0, ord=2) > tau + 1e-10:
        raise ValueError("x0 must satisfy the operator-norm constraint.")

    recorder = _FactorCheckpointRecorder(n_steps, metrics_frequency)
    algorithm = StochasticFrames(
        objective,
        lmo,
        identity_prox,
        objective_type="indicator",
    )
    recorder.start()
    algorithm.run(
        x0,
        beta0=beta0,
        n_steps=n_steps,
        show_progress=show_progress,
        rho_schedule=rho_schedule,
        smoothing_schedule=smoothing_schedule,
        step_size_schedule=step_size_schedule,
        evaluate_objective=False,
        iterate_callback=recorder,
        iterate_callback_frequency=metrics_frequency,
    )

    checkpoint_steps = np.asarray(recorder.steps, dtype=int)
    if not np.array_equal(checkpoint_steps, expected_checkpoints):
        raise RuntimeError("the optimizer did not emit the expected checkpoints.")

    measurement_loss = []
    tp_violation = []
    objective_value = []
    fidelity_proxy = []
    exact_fw_gap = []
    for vector in recorder.iterates:
        loss, penalty = objective.loss_components(vector)
        gradient = objective.gradient(vector)
        atom = lmo(gradient)
        chi = objective.process_matrix(vector)
        measurement_loss.append(loss)
        tp_violation.append(np.sqrt(penalty))
        objective_value.append(loss + objective.lam * penalty)
        fidelity_proxy.append(
            process_fidelity_proxy(chi, data.chi_star, data.d)
        )
        exact_fw_gap.append(np.dot(gradient, vector - atom))

    final_factor = objective.unpack(algorithm.x)
    exact_fw_gap = np.asarray(exact_fw_gap)
    return QPTFactorExperimentResult(
        rank=objective.rank,
        tau=float(tau),
        lam=objective.lam,
        batch_size=objective.batch_size,
        initialization_seed=int(initialization_seed),
        sampling_seed=int(sampling_seed),
        final_x=algorithm.x.copy(),
        final_factor=final_factor,
        final_chi=final_factor @ final_factor.conj().T,
        checkpoint_steps=checkpoint_steps,
        optimizer_seconds=np.asarray(recorder.seconds),
        measurement_loss=np.asarray(measurement_loss),
        tp_violation=np.asarray(tp_violation),
        objective_value=np.asarray(objective_value),
        process_fidelity_proxy=np.asarray(fidelity_proxy),
        exact_fw_gap=exact_fw_gap,
        # Upstream packs an un-doubled Wirtinger derivative. Its reported
        # gaps are therefore half the true packed-real directional gap.
        qpt_bfw_exact_gap=0.5 * exact_fw_gap,
        estimated_gaps=algorithm.estimated_gaps.copy(),
        qpt_bfw_estimated_gaps=0.5 * algorithm.estimated_gaps,
        momentum_weights=algorithm.momentum_weights.copy(),
        smoothing_parameters=algorithm.smoothing_parameters.copy(),
        step_sizes=algorithm.step_sizes.copy(),
        cumulative_stochastic_oracles=(
            algorithm.num_stochastic_oracles.copy()
        ),
        cumulative_sampled_measurements=(
            np.rint(batch_size * algorithm.num_stochastic_oracles).astype(int)
        ),
        algorithm=algorithm,
    )


def save_qpt_factor_result(path, result, metadata=None):
    payload = {
        "meta_rank": np.asarray(result.rank),
        "meta_tau": np.asarray(result.tau),
        "meta_lam": np.asarray(result.lam),
        "meta_batch_size": np.asarray(result.batch_size),
        "meta_initialization_seed": np.asarray(result.initialization_seed),
        "meta_sampling_seed": np.asarray(result.sampling_seed),
        "final_x": result.final_x,
        "final_factor": result.final_factor,
        "final_chi": result.final_chi,
        "checkpoint_steps": result.checkpoint_steps,
        "optimizer_seconds": result.optimizer_seconds,
        "measurement_loss": result.measurement_loss,
        "tp_violation": result.tp_violation,
        "objective_value": result.objective_value,
        "process_fidelity_proxy": result.process_fidelity_proxy,
        "exact_fw_gap": result.exact_fw_gap,
        "qpt_bfw_exact_gap": result.qpt_bfw_exact_gap,
        "estimated_gaps": result.estimated_gaps,
        "qpt_bfw_estimated_gaps": result.qpt_bfw_estimated_gaps,
        "momentum_weights": result.momentum_weights,
        "smoothing_parameters": result.smoothing_parameters,
        "step_sizes": result.step_sizes,
        "cumulative_stochastic_oracles": (
            result.cumulative_stochastic_oracles
        ),
        "cumulative_sampled_measurements": (
            result.cumulative_sampled_measurements
        ),
    }
    if metadata:
        payload.update(
            {f"meta_{key}": np.asarray(value) for key, value in metadata.items()}
        )
    np.savez_compressed(path, **payload)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the momentum stochastic-FW compatibility baseline on the "
            "original factorized QPT_BFW objective."
        )
    )
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--rho-scale", type=float, default=4.0)
    parser.add_argument("--rho-offset", type=float, default=8.0)
    parser.add_argument("--rho-exponent", type=float, default=2.0 / 3.0)
    parser.add_argument("--smoothing-scale", type=float, default=1.0)
    parser.add_argument("--smoothing-offset", type=float, default=1.0)
    parser.add_argument("--smoothing-exponent", type=float, default=0.25)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--step-offset", type=float, default=1.0)
    parser.add_argument("--step-exponent", type=float, default=0.5)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    data = QPTData.from_hdf5(args.h5)
    rho_schedule = PowerSchedule(
        args.rho_scale, args.rho_offset, args.rho_exponent, cap=1.0
    )
    smoothing_schedule = PowerSchedule(
        args.smoothing_scale,
        args.smoothing_offset,
        args.smoothing_exponent,
    )
    step_size_schedule = PowerSchedule(
        args.step_scale, args.step_offset, args.step_exponent, cap=1.0
    )
    result = run_qpt_factor_stochastic_frames(
        data,
        n_steps=args.steps,
        rank=args.rank,
        tau=args.tau,
        lam=args.lam,
        batch_size=args.batch_size,
        initialization_seed=args.initialization_seed,
        sampling_seed=args.sampling_seed,
        rho_schedule=rho_schedule,
        smoothing_schedule=smoothing_schedule,
        step_size_schedule=step_size_schedule,
        metrics_frequency=args.metrics_every,
        show_progress=not args.quiet,
    )
    print(
        f"Final objective={result.objective_value[-1]:.6e}, "
        f"TP violation={result.tp_violation[-1]:.6e}, "
        f"exact FW gap={result.exact_fw_gap[-1]:.6e}"
    )
    print(
        "Note: smoothing parameters are recorded but inert because g=0 in "
        "the original factor formulation."
    )
    if args.save is not None:
        save_qpt_factor_result(
            args.save,
            result,
            metadata={
                "formulation": "qpt_bfw_factor",
                "source_h5": str(args.h5.resolve()),
                "steps": args.steps,
                "rank": args.rank,
                "tau": args.tau,
                "lam": args.lam,
                "batch_size": args.batch_size,
                "metrics_every": args.metrics_every,
                "initialization_seed": args.initialization_seed,
                "sampling_seed": args.sampling_seed,
                "rho_scale": args.rho_scale,
                "rho_offset": args.rho_offset,
                "rho_exponent": args.rho_exponent,
                "smoothing_scale": args.smoothing_scale,
                "smoothing_offset": args.smoothing_offset,
                "smoothing_exponent": args.smoothing_exponent,
                "step_scale": args.step_scale,
                "step_offset": args.step_offset,
                "step_exponent": args.step_exponent,
            },
        )
        print(f"Saved results to {args.save}")
    return result


if __name__ == "__main__":
    main()
