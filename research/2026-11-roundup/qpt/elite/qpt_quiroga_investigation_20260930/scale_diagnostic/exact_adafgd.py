"""Exact-matrix-norm Quiroga adaFGD on a fixed QPT subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from paper.experiments import quantum_process_tomography_structured_jax as qpt
from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.qpt_structured_operators import (
    product_measurement_vectors,
    rank_one_measurement_vectors,
    trace_preserving_loss_and_gradient,
)
from paper.experiments.quantum_process_tomography_jax import (
    _configure_jax_precision,
    _require_jax,
    select_jax_device,
)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--save", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--subset-size", type=int, default=8192)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--initialization-seed", type=int, default=0)
    p.add_argument("--initialization", choices=("trace_normalized", "raw_uniform"), default="trace_normalized")
    p.add_argument("--eta-scale", type=float, required=True)
    p.add_argument("--tp-weight", type=float, required=True)
    p.add_argument("--metrics-every", type=int, default=100)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    if a.steps < 1 or a.subset_size < 1 or a.metrics_every < 1:
        p.error("steps, subset-size, and metrics-every must be positive")
    if not np.isfinite(a.eta_scale) or a.eta_scale <= 0:
        p.error("eta-scale must be finite and positive")
    if not np.isfinite(a.tp_weight) or a.tp_weight <= 0:
        p.error("tp-weight must be finite and positive")

    setup_started = perf_counter()
    data = StructuredQPTData.load_npz(a.data)
    if data.observation_mode != "stored":
        raise ValueError("this exact comparison requires a stored dataset")
    if a.subset_size > data.m:
        raise ValueError("subset-size exceeds the measurement population")
    rng = np.random.default_rng(a.subset_seed)
    indices = rng.choice(data.m, size=a.subset_size, replace=False)
    symbols = data.indices_to_symbols(indices).astype(np.int32)
    observations = np.asarray(data.observations_for_symbols(symbols), dtype=np.float64)
    local_vectors = rank_one_measurement_vectors(data.local_measurements)
    if local_vectors is None:
        raise ValueError("exact adaFGD requires the verified rank-one sensing bank")
    vectors = np.asarray(product_measurement_vectors(symbols, local_vectors), dtype=np.complex128)
    if a.initialization == "trace_normalized":
        x0 = qpt.make_factor_initial_point(data, 1, a.initialization_seed)
        factor0 = qpt.unpack_factor(np.asarray(x0), data.process_dimension, 1)
    else:
        init_rng = np.random.default_rng(a.initialization_seed)
        factor0 = init_rng.uniform(0.0, 1.0, (data.process_dimension, 1))
        factor0 = factor0 + 1j * init_rng.uniform(0.0, 1.0, (data.process_dimension, 1))

    jax = _require_jax()
    _configure_jax_precision(jax, "64")
    import jax.numpy as jnp

    device = select_jax_device("gpu", 0)
    with jax.default_device(device):
        factor = jax.device_put(factor0, device)
        h = jax.device_put(vectors, device)
        targets = jax.device_put(observations, device)
        basis = jax.device_put(data.local_basis, device)
        truth = jax.device_put(data.truth_factor, device)
        jax.block_until_ready((factor, h, targets, basis, truth))

        def quantities(current):
            amplitudes = jnp.conj(h) @ current
            predicted = jnp.sum(jnp.real(jnp.conj(amplitudes) * amplitudes), axis=1)
            residual = predicted - targets
            # Quiroga Eq. (10) uses a sum of squared residuals.
            matrix_gradient = h.T @ (residual[:, None] * jnp.conj(h))
            matrix_gradient = 0.5 * (matrix_gradient + jnp.conj(matrix_gradient.T))
            eigenvalues = jnp.linalg.eigvalsh(matrix_gradient)
            numerator = jnp.max(jnp.abs(eigenvalues))
            denominator = jnp.linalg.norm(predicted)
            tiny = jnp.finfo(predicted.dtype).tiny
            eta = a.eta_scale * numerator / jnp.maximum(denominator, tiny)
            # The published update uses the matrix-space gradient times U.
            measurement_gradient = matrix_gradient @ current
            tp_loss, tp_gradient = trace_preserving_loss_and_gradient(current, basis, xp=jnp)
            measurement_loss = 0.5 * jnp.sum(residual**2)
            return (measurement_loss, tp_loss, measurement_gradient, tp_gradient,
                    numerator, denominator, eta)

        def one_step(current, _):
            _, _, measurement_gradient, tp_gradient, numerator, denominator, eta = quantities(current)
            following = current - eta * (measurement_gradient + a.tp_weight * tp_gradient)
            return following, jnp.stack((eta, numerator, denominator))

        def make_advance(length):
            @jax.jit
            def advance(current):
                return jax.lax.scan(one_step, current, None, length=length)
            return advance

        @jax.jit
        def metrics(current):
            measurement_loss, tp_loss, _, _, numerator, denominator, eta = quantities(current)
            overlap = jnp.sum(jnp.abs(jnp.conj(truth.T) @ current) ** 2)
            trace = jnp.sum(jnp.abs(current) ** 2)
            fidelity = overlap / (data.d * trace)
            return jnp.stack((measurement_loss, jnp.sqrt(2.0 * tp_loss),
                              measurement_loss + a.tp_weight * (2.0 * tp_loss),
                              fidelity, numerator, denominator, eta))

        checkpoints = np.unique(np.r_[0, np.arange(a.metrics_every, a.steps + 1, a.metrics_every), a.steps])
        rows, times, all_diagnostics, compiled = [], [], [], {}
        optimizer_seconds = 0.0
        setup_seconds = perf_counter() - setup_started
        for start, stop in zip(checkpoints[:-1], checkpoints[1:]):
            if not rows:
                rows.append(np.asarray(jax.device_get(metrics(factor))))
                times.append(optimizer_seconds)
            length = int(stop - start)
            if length not in compiled:
                compiled[length] = make_advance(length)
            began = perf_counter()
            factor, diagnostics = compiled[length](factor)
            jax.block_until_ready((factor, diagnostics))
            optimizer_seconds += perf_counter() - began
            host_diagnostics = np.asarray(jax.device_get(diagnostics))
            if not np.all(np.isfinite(host_diagnostics)):
                raise FloatingPointError("nonfinite exact adaFGD diagnostics")
            all_diagnostics.append(host_diagnostics)
            row = np.asarray(jax.device_get(metrics(factor)))
            if not np.all(np.isfinite(row)):
                raise FloatingPointError("nonfinite exact adaFGD checkpoint")
            rows.append(row)
            times.append(optimizer_seconds)
            if not a.quiet:
                print(f"step {stop}/{a.steps}: fidelity={row[3]:.9g}, TP={row[1]:.6g}, eta={row[6]:.6g}", flush=True)
        final_factor = np.asarray(jax.device_get(factor))

    values = np.asarray(rows)
    diagnostics = np.concatenate(all_diagnostics, axis=0)
    metadata = dict(
        schema_version=1, format="qpt_quiroga_exact_result",
        optimizer="quiroga_adafgd_exact_matrix_norm", n_qubits=int(data.n_qubits),
        n_steps=int(a.steps), requested_n_steps=int(a.steps), rank=1,
        initialization_seed=int(a.initialization_seed), sampling_seed=int(a.subset_seed),
        subset_seed=int(a.subset_seed), subset_size=int(a.subset_size),
        subset_without_replacement=True, subset_indices_sha256=digest(indices),
        subset_symbols_sha256=digest(symbols), subset_observations_sha256=digest(observations),
        initial_factor_sha256=digest(factor0), initialization=a.initialization,
        initial_factor_norm=float(np.linalg.norm(factor0)), eta_scale=float(a.eta_scale),
        tp_weight=float(a.tp_weight), measurement_loss_normalization="sum_half_squared_as_published",
        adaptive_step_rule="eta_scale*||A^dagger(A(UU^H)-f)||_2/||A(UU^H)||_2",
        adaptive_numerator="exact Hermitian spectral norm via eigvalsh",
        update_gradient="A^dagger(A(UU^H)-f)@U + lambda*grad_chi_H@U",
        tp_penalty="exact ||sum K^H K-I||_F^2; source half-loss factor gradient equals published grad_chi_H@U", projection="none",
        device=str(device), precision="64", optimizer_seconds=float(optimizer_seconds),
        setup_seconds=float(setup_seconds), data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
    )
    save(a.save, dict(
        final_factor=final_factor,
        checkpoint_steps=checkpoints.astype(np.int64),
        optimizer_seconds=np.asarray(times),
        measurement_loss=values[:, 0], tp_violation=values[:, 1],
        smoothed_objective=values[:, 2], process_fidelity_proxy=values[:, 3],
        matrix_gradient_spectral_norm=values[:, 4], prediction_norm=values[:, 5],
        checkpoint_adaptive_step=values[:, 6], estimated_gaps=diagnostics[:, 0],
        step_matrix_gradient_spectral_norm=diagnostics[:, 1],
        step_prediction_norm=diagnostics[:, 2],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    ))
    print(f"Saved {a.save}; fidelity={values[-1,3]:.12g}; TP={values[-1,1]:.12g}; eta={values[-1,6]:.12g}", flush=True)


if __name__ == "__main__":
    main()
