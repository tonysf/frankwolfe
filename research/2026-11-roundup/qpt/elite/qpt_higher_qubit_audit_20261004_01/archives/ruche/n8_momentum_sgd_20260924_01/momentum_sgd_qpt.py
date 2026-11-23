"""Projected momentum-SGD baseline for the frozen structured QPT runner.

The baseline deliberately reuses the production runner's data sampling,
measurement gradients, exact TP penalty, metrics, checkpointing, and timing.
Only the factor update is replaced:

    m_k = (1-rho_k) m_{k-1} + rho_k g_k
    U_{k+1} = projection_{||.||_2 <= tau}(U_k - eta_k(m_k + grad TP/beta_k)).

This campaign is rank one, where the spectral-norm projection is exactly a
Euclidean rescaling of the factor vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paper.experiments import quantum_process_tomography_structured_jax as qpt
from paper.experiments.quantum_process_tomography import PowerSchedule
from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.qpt_structured_operators import trace_preserving_loss_and_gradient


def _build_projected_momentum_scan(
    jax, jnp, tau, noiseless, rank_one=False, product_state=False
):
    """Drop-in replacement for the runner's FRAMES scan builder."""
    values_fn, gradient_fn = qpt._measurement_functions(rank_one, product_state)

    def scan(state, start, symbols, observations, rho, beta, eta, bank, basis, truth):

        def step(carry, inputs):
            factor, previous = carry
            iteration, batch, targets, weight, smoothing, learning_rate = inputs
            if noiseless:
                targets = values_fn(truth, batch, bank, xp=jnp) + targets
            _, sampled = gradient_fn(factor, batch, targets, bank, xp=jnp)
            momentum = jax.lax.cond(
                iteration == 0,
                lambda: sampled,
                lambda: (1.0 - weight) * previous + weight * sampled,
            )
            _, tp_gradient = trace_preserving_loss_and_gradient(factor, basis, xp=jnp)
            combined = momentum + tp_gradient / smoothing
            candidate = factor - learning_rate * combined
            norm = jnp.linalg.norm(candidate)
            scale = jnp.minimum(
                1.0, tau / jnp.maximum(norm, jnp.finfo(candidate.real.dtype).tiny)
            )
            next_factor = candidate * scale

            # Retain the same FW-gap diagnostic as FRAMES. It is evaluated at
            # the pre-update point and is independent of the optimizer choice.
            atom = qpt._lmo(jnp, combined, tau)
            gap = jnp.vdot(combined, factor - atom).real
            return (next_factor, momentum), gap

        iterations = start + jnp.arange(symbols.shape[0], dtype=jnp.int32)
        return jax.lax.scan(
            step, state, (iterations, symbols, observations, rho, beta, eta)
        )

    return jax.jit(scan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--chunk-steps", type=int, default=1)
    parser.add_argument("--metrics-every", type=int, default=1000)
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--metric-seed", type=int, default=12345)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--rho-scale", type=float, default=0.6)
    parser.add_argument("--rho-offset", type=float, default=4.0)
    parser.add_argument("--rho-exponent", type=float, default=0.5)
    parser.add_argument("--smoothing-scale", type=float, default=1.0e7)
    parser.add_argument("--smoothing-offset", type=float, default=1.0)
    parser.add_argument("--smoothing-exponent", type=float, default=0.25)
    parser.add_argument("--eta-scale", type=float, required=True)
    parser.add_argument("--eta-offset", type=float, default=10.0)
    parser.add_argument("--eta-exponent", type=float, default=0.75)
    parser.add_argument("--restart-path", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.eta_scale <= 0 or not np.isfinite(args.eta_scale):
        parser.error("--eta-scale must be finite and positive")

    data = StructuredQPTData.load_npz(args.data)
    schedules = {
        "rho": PowerSchedule(
            args.rho_scale, args.rho_offset, args.rho_exponent, cap=1.0
        ),
        "smoothing": PowerSchedule(
            args.smoothing_scale,
            args.smoothing_offset,
            args.smoothing_exponent,
            cap=None,
        ),
        "eta": PowerSchedule(
            args.eta_scale, args.eta_offset, args.eta_exponent, cap=None
        ),
    }

    original_builder = qpt._build_scan
    qpt._build_scan = _build_projected_momentum_scan
    try:
        result = qpt.run_qpt_structured_jax(
            data,
            n_steps=args.steps,
            rank=1,
            tau=args.tau,
            batch_size=args.batch_size,
            initialization_seed=args.initialization_seed,
            sampling_seed=args.sampling_seed,
            beta0=args.smoothing_scale,
            rho_schedule=schedules["rho"],
            smoothing_schedule=schedules["smoothing"],
            step_size_schedule=schedules["eta"],
            chunk_steps=args.chunk_steps,
            metrics_frequency=args.metrics_every,
            metric_mode="sampled",
            metric_samples=args.metric_samples,
            metric_batch_size=args.metric_batch_size,
            metric_seed=args.metric_seed,
            device="gpu",
            precision="64",
            show_progress=not args.quiet,
            measurement_backend="product-state",
            restart_path=args.restart_path,
            resume=args.resume,
            prefetch=True,
        )
    finally:
        qpt._build_scan = original_builder

    result.metadata.update(
        optimizer="projected_momentum_sgd",
        update_rule=(
            "m=(1-rho)*m+rho*sample_gradient; "
            "U=project_operator_norm_ball(U-eta*(m+tp_gradient/beta))"
        ),
        projection="rank1_spectral_norm_ball_via_factor_l2_rescaling",
        eta_scale=float(args.eta_scale),
        eta_offset=float(args.eta_offset),
        eta_exponent=float(args.eta_exponent),
        eta_cap=None,
        diagnostic_gap="Frank-Wolfe gap of the common smoothed objective",
    )
    qpt.save_structured_result(args.save, result)
    print(
        f"Saved {args.save}; optimizer={result.total_seconds:.3f}s; "
        f"final_fidelity={result.process_fidelity_proxy[-1]:.12g}; "
        f"final_loss={result.measurement_loss[-1]:.12g}; "
        f"final_tp={result.tp_violation[-1]:.12g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
