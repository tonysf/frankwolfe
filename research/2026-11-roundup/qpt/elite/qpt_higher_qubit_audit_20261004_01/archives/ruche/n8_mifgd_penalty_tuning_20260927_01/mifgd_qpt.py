"""Stochastic MiFGD/FGD baseline for rank-one structured QPT.

This adapts the published MiFGD two-sequence recursion to the common QPT
minibatch oracle and the fixed-penalty QPT objective

    measurement_loss(U) + tp_weight * 0.5 ||T(UU^H)-I||_F^2.

No TP smoothing schedule or norm-ball projection is used.  ``mu=0`` is FGD;
``mu>0`` is the Nesterov-style MiFGD update.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paper.experiments import quantum_process_tomography_structured_jax as qpt
from paper.experiments.quantum_process_tomography import PowerSchedule
from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.qpt_structured_operators import trace_preserving_loss_and_gradient


def make_mifgd_scan(momentum):
    momentum = float(momentum)

    def build(jax, jnp, tau, noiseless, rank_one=False, product_state=False):
        values_fn, gradient_fn = qpt._measurement_functions(rank_one, product_state)

        def scan(state, start, symbols, observations, rho, beta, eta, bank, basis, truth):
            del rho

            def step(carry, inputs):
                factor, previous_factor = carry
                iteration, batch, targets, smoothing, learning_rate = inputs
                extrapolated = jax.lax.cond(
                    iteration == 0,
                    lambda: factor,
                    lambda: factor + momentum * (factor - previous_factor),
                )
                if noiseless:
                    targets = values_fn(truth, batch, bank, xp=jnp) + targets
                _, measurement_gradient = gradient_fn(
                    extrapolated, batch, targets, bank, xp=jnp
                )
                _, tp_gradient = trace_preserving_loss_and_gradient(
                    extrapolated, basis, xp=jnp
                )
                combined = measurement_gradient + tp_gradient / smoothing
                next_factor = extrapolated - learning_rate * combined
                atom = qpt._lmo(jnp, combined, tau)
                gap = jnp.vdot(combined, extrapolated - atom).real
                return (next_factor, factor), gap

            iterations = start + jnp.arange(symbols.shape[0], dtype=jnp.int32)
            return jax.lax.scan(
                step,
                state,
                (iterations, symbols, observations, beta, eta),
            )

        return jax.jit(scan)

    return build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--chunk-steps", type=int, default=1)
    parser.add_argument("--metrics-every", type=int, default=1000)
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--metric-seed", type=int, default=12345)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--mu", type=float, default=0.75)
    parser.add_argument("--tp-weight", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.eta) or args.eta <= 0:
        parser.error("--eta must be finite and positive")
    if not np.isfinite(args.mu) or not 0 <= args.mu < 1:
        parser.error("--mu must be in [0, 1)")
    if not np.isfinite(args.tp_weight) or args.tp_weight <= 0:
        parser.error("--tp-weight must be finite and positive")

    data = StructuredQPTData.load_npz(args.data)
    constant_eta = PowerSchedule(args.eta, 1.0, 0.0, cap=None)
    constant_beta = PowerSchedule(1.0 / args.tp_weight, 1.0, 0.0, cap=None)
    unused_rho = PowerSchedule(1.0, 1.0, 0.0, cap=1.0)

    original_builder = qpt._build_scan
    qpt._build_scan = make_mifgd_scan(args.mu)
    try:
        result = qpt.run_qpt_structured_jax(
            data,
            n_steps=args.steps,
            rank=1,
            tau=20.0,
            batch_size=args.batch_size,
            initialization_seed=args.initialization_seed,
            sampling_seed=args.sampling_seed,
            beta0=1.0 / args.tp_weight,
            rho_schedule=unused_rho,
            smoothing_schedule=constant_beta,
            step_size_schedule=constant_eta,
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
            prefetch=True,
        )
    finally:
        qpt._build_scan = original_builder

    method = "fgd_fixed_tp" if args.mu == 0 else "mifgd_fixed_tp"
    result.metadata.update(
        optimizer=method,
        update_rule=(
            "Z=U+mu*(U-U_previous); "
            "U_next=Z-eta*(sample_gradient(Z)+tp_weight*tp_gradient(Z))"
        ),
        objective=(
            "sampled_measurement_loss + tp_weight * "
            "0.5*||T(UU^H)-I||_F^2"
        ),
        tp_weight=float(args.tp_weight),
        eta=float(args.eta),
        eta_schedule="constant",
        mu=float(args.mu),
        projection="none",
        gradient_estimate_field_semantics="previous_factor_for_two_sequence_update",
        literature_note="MiFGD recursion; fixed TP penalty adapted from QPT FGD",
    )
    qpt.save_structured_result(args.save, result)
    print(
        f"Saved {args.save}; method={method}; optimizer={result.total_seconds:.3f}s; "
        f"final_fidelity={result.process_fidelity_proxy[-1]:.12g}; "
        f"final_loss={result.measurement_loss[-1]:.12g}; "
        f"final_tp={result.tp_violation[-1]:.12g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
