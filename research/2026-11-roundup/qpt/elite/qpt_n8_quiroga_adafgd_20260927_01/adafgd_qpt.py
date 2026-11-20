"""Structured rank-one QPT baseline following Quiroga-style adaFGD.

The finite measurement set is sampled once and reused.  The objective is the
factorized measurement loss plus a fixed TP penalty.  At each iteration the
adaptive learning rate follows a scalable factor-space proxy for the
published proportional rule

    eta_k = eta_scale * ||factor_measurement_gradient(U_k)||_2
                        / ||A(U_k U_k^H)||_2.

The paper's numerator is the spectral norm of the matrix-space adjoint
residual.  Materializing that 65536-by-65536 matrix is impossible here, so its
action on the rank-one factor is used and the approximation is recorded in the
result metadata.  The paper does not specify ``eta_scale`` or the fixed TP
weight; those are the only quantities exposed for seed-0 calibration.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from paper.experiments import quantum_process_tomography_structured_jax as qpt
from paper.experiments.quantum_process_tomography import PowerSchedule
from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.qpt_structured_operators import trace_preserving_loss_and_gradient


def _build_adafgd_scan(jax, jnp, tau, noiseless, rank_one=False, product_state=False, eta_multiplier=1.0):
    values_fn, gradient_fn = qpt._measurement_functions(rank_one, product_state)

    def scan(state, start, symbols, observations, rho, beta, eta_scale, bank, basis, truth):
        del rho

        def step(carry, inputs):
            factor, _ = carry
            iteration, batch, targets, fixed_penalty_beta, scale = inputs
            del iteration
            predicted = values_fn(factor, batch, bank, xp=jnp)
            if noiseless:
                targets = values_fn(truth, batch, bank, xp=jnp) + targets
            _, measurement_gradient = gradient_fn(factor, batch, targets, bank, xp=jnp)
            _, tp_gradient = trace_preserving_loss_and_gradient(factor, basis, xp=jnp)
            combined = measurement_gradient + tp_gradient / fixed_penalty_beta
            tiny = jnp.finfo(factor.real.dtype).tiny
            adaptive_eta = eta_multiplier * scale * jnp.linalg.norm(measurement_gradient) / jnp.maximum(
                jnp.linalg.norm(predicted), tiny
            )
            next_factor = factor - adaptive_eta * combined
            return (next_factor, combined), adaptive_eta

        iterations = start + jnp.arange(symbols.shape[0], dtype=jnp.int32)
        return jax.lax.scan(
            step, state, (iterations, symbols, observations, beta, eta_scale)
        )

    return jax.jit(scan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--metrics-every", type=int, default=1000)
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--metric-seed", type=int, default=12345)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--eta-scale", type=float, required=True)
    parser.add_argument("--tp-weight", type=float, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.eta_scale) or args.eta_scale <= 0:
        parser.error("--eta-scale must be finite and positive")
    if not np.isfinite(args.tp_weight) or args.tp_weight <= 0:
        parser.error("--tp-weight must be finite and positive")

    data = StructuredQPTData.load_npz(args.data)
    from paper.experiments import qpt_batch_preparation
    fixed_rng = np.random.default_rng(args.sampling_seed)
    fixed_symbols, fixed_noise = qpt_batch_preparation.prepare_on_demand_batch(
        data, fixed_rng, args.batch_size
    )
    fixed_symbols = np.ascontiguousarray(fixed_symbols)
    fixed_noise = np.ascontiguousarray(fixed_noise)
    fixed_digest = hashlib.sha256(fixed_symbols.tobytes()).hexdigest()
    noise_digest = hashlib.sha256(fixed_noise.tobytes()).hexdigest()
    original_prepare = qpt_batch_preparation.prepare_on_demand_batch

    def fixed_prepare(batch_data, rng, count):
        del rng
        if batch_data is not data or count % args.batch_size:
            raise ValueError("fixed adaFGD preparation requires whole fixed batches")
        repeats = count // args.batch_size
        if repeats == 1:
            return fixed_symbols.copy(), fixed_noise.copy()
        return np.tile(fixed_symbols, (repeats, 1)), np.tile(fixed_noise, repeats)

    qpt_batch_preparation.prepare_on_demand_batch = fixed_prepare
    original_builder = qpt._build_scan
    def adafgd_builder(jax, jnp, tau, noiseless, rank_one=False, product_state=False):
        return _build_adafgd_scan(
            jax, jnp, tau, noiseless, rank_one, product_state, args.eta_scale
        )

    qpt._build_scan = adafgd_builder
    constant_beta = PowerSchedule(1.0 / args.tp_weight, 1.0, 0.0, cap=None)
    constant_scale = PowerSchedule(1.0, 1.0, 0.0, cap=1.0)
    unused_rho = PowerSchedule(1.0, 1.0, 0.0, cap=1.0)
    try:
        result = qpt.run_qpt_structured_jax(
            data, n_steps=args.steps, rank=1, tau=20.0,
            batch_size=args.batch_size, initialization_seed=args.initialization_seed,
            sampling_seed=args.sampling_seed, beta0=1.0 / args.tp_weight,
            rho_schedule=unused_rho, smoothing_schedule=constant_beta,
            step_size_schedule=constant_scale, chunk_steps=1,
            metrics_frequency=args.metrics_every, metric_mode="sampled",
            metric_samples=args.metric_samples, metric_batch_size=args.metric_batch_size,
            metric_seed=args.metric_seed, device="gpu", precision="64",
            show_progress=not args.quiet, measurement_backend="product-state",
            prefetch=True,
        )
    finally:
        qpt._build_scan = original_builder
        qpt_batch_preparation.prepare_on_demand_batch = original_prepare

    result.metadata.update(
        optimizer="quiroga_adafgd_fixed_tp",
        update_rule="U_next=U-eta_k*(measurement_gradient+tp_weight*tp_gradient)",
        adaptive_step_rule_published="eta proportional to ||A^dagger(A(UU^H)-f)||_2/||A(UU^H)||_2",
        adaptive_step_implementation="eta_scale*||factor_measurement_gradient||_F/||A(UU^H)||_2",
        adaptive_step_approximation="matrix-space numerator replaced by its rank-one factor action; no dense process matrix is materialized",
        eta_scale=float(args.eta_scale), tp_weight=float(args.tp_weight),
        measurement_set="fixed_seeded_subset_reused_every_iteration",
        fixed_measurement_symbols_sha256=fixed_digest,
        fixed_measurement_noise_sha256=noise_digest,
        fixed_measurement_count=int(args.batch_size), projection="none",
        estimated_gaps_field_semantics="realized_adaptive_learning_rate",
        literature_note="Quiroga and Kyrillidis adaFGD; unspecified constants calibrated on seed 0",
    )
    qpt.save_structured_result(args.save, result)
    rates = result.estimated_gaps
    print(
        f"Saved {args.save}; fidelity={result.process_fidelity_proxy[-1]:.12g}; "
        f"loss={result.measurement_loss[-1]:.12g}; tp={result.tp_violation[-1]:.12g}; "
        f"eta_first={rates[0]:.12g}; eta_last={rates[-1]:.12g}", flush=True
    )


if __name__ == "__main__":
    main()
