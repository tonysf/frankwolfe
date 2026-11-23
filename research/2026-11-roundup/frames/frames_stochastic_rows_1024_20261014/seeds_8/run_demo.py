#!/usr/bin/env python3
"""Nonconvex stochastic FRAMES with uniform scalar-row least squares.

Requirements: numpy, matplotlib. Run: python run_demo.py
f(x)=||A x-B||^2-(5/8)(u^T x)^2. Each update samples exactly one row.
The data matrix is fixed; no extra gradient noise is injected. Random
initialization and sampled rows use separate streams. The N^(-1/3) and
N^(-1/4) lines are slope guides; all schedules and prefactors are unchanged.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = Path("/home/tony/gw_benchmark_experiment_20260927/provenance/manuscript.txt")
DIM, STEPS = 1024, 131072
LAST_WINDOW = 10000
REFERENCE_ANCHOR = 50000
ZOOM_START = 1000
SEEDS = list(range(8))
DATA_SEED = 20261014
RHO = 5 / 4
CENTER = np.r_[np.full(64, 1/32), np.full(64, -3/128), np.full(896, 1/128)]
U = np.r_[np.where(np.arange(64) % 2 == 0, 1., -1.) / 8, np.zeros(960)]
M = np.sqrt(311) / 16
HADAMARD = np.ones((1, 1))
while HADAMARD.shape[0] < DIM:
    HADAMARD = np.block([[HADAMARD, HADAMARD], [HADAMARD, -HADAMARD]])
COLUMN_SIGNS = 2*np.random.default_rng(DATA_SEED).integers(0, 2, DIM)-1
A = HADAMARD * COLUMN_SIGNS[None, :] / np.sqrt(2*DIM)
B = A @ CENTER
del HADAMARD


def initial_points(seeds):
    """Uniform random negative vertices, independent of oracle RNG streams."""
    indices = np.array([np.random.default_rng(np.random.SeedSequence(
        [20261014, int(seed), 1024])).integers(DIM) for seed in seeds])
    points = np.zeros((len(seeds), DIM))
    points[np.arange(len(seeds)), indices] = -1.
    return points, indices


def gradient(x):
    """Exact full gradient, using A^T A=I/2; only for certificates/checks."""
    return x - CENTER - RHO * np.sum(x * U, axis=-1, keepdims=True) * U


def loss(x):
    return .5 * np.sum((x-CENTER)**2, axis=-1) - RHO/2 * np.sum(x*U, axis=-1)**2


def row_oracle(x, indices):
    """One uniformly sampled scalar residual per x; factor DIM is essential."""
    rows = A[indices]
    residuals = np.sum(rows*x, axis=-1)-B[indices]
    return (2*DIM*residuals[..., None]*rows
            - RHO*np.sum(x*U, axis=-1, keepdims=True)*U)


def certificates(x, beta, grad=None):
    if grad is None:
        grad = gradient(x)
    negative = np.minimum(x, 0.)
    h = grad + negative / beta
    gap = np.sum(h*x, axis=-1) + np.max(np.abs(h), axis=-1)
    original = np.sum(grad*x, axis=-1) + np.maximum(0., -np.min(grad, axis=-1))
    feasibility = np.linalg.norm(negative, axis=-1)
    return gap, original, feasibility


def step(x, momentum, beta, gamma):
    h = momentum + np.minimum(x, 0.) / beta
    j = np.argmax(np.abs(h), axis=1)
    signs = np.sign(h[np.arange(len(x)), j])
    result = (1-gamma) * x
    result[np.arange(len(x)), j] -= gamma * signs
    return result


def analytic_stationary_points():
    saddle = np.r_[np.full(64, 1/64), np.zeros(960)]
    plus, minus = np.zeros(DIM), np.zeros(DIM)
    plus[U > 0] = 1/32
    minus[U < 0] = 1/32
    return saddle, plus, minus


def smoothed_stationary_point(beta, sign=1):
    """An exact smoothed stationary branch; no global-minimum claim needed."""
    w = beta / (256 * (1 + 7*beta/4))
    result = np.zeros(DIM)
    result[sign*U > 0] = 1/32 - 2*w
    result[64:128] = -w
    return result


def validate_math():
    np.testing.assert_allclose(U @ U, 1.)
    np.testing.assert_allclose(U @ CENTER, 0., atol=1e-16)
    np.testing.assert_allclose(np.sum(A*A, axis=1), .5, atol=1e-14)
    # The Hessian acts as -1/4 along U, and as identity perpendicular to U.
    hessian_u = U - RHO * np.dot(U, U) * U
    np.testing.assert_allclose(hessian_u, -U/4, atol=1e-16)
    saddle, plus, minus = analytic_stationary_points()
    np.testing.assert_allclose(loss(saddle), 27/512)
    np.testing.assert_allclose(loss(np.stack([plus, minus])), 13/256)
    for t in [-.1, -.05, 0., .05, .1]:
        z = saddle + t*U
        assert z.min() >= 0 and np.linalg.norm(z, 1) <= 1+1e-14
        np.testing.assert_allclose(loss(z), 27/512-t*t/8, atol=1e-15)
    for point in [saddle, plus, minus]:
        _, original, delta = certificates(point[None, :], 1.)
        np.testing.assert_allclose(original, 0., atol=1e-15)
        np.testing.assert_allclose(delta, 0., atol=1e-15)
    # The orthant is essential: each original minimum has a descent direction
    # towards a negative vertex of the second block when D is removed.
    direction = -plus.copy()
    direction[64] -= 1
    np.testing.assert_allclose(gradient(plus) @ direction, -1/256, atol=1e-15)
    rng = np.random.default_rng(1024)
    x = rng.normal(size=(100, DIM))
    x *= rng.uniform(size=(100, 1)) / np.sum(np.abs(x), axis=1, keepdims=True)
    # Include vertices and nonconvex stationary points, not only small random x.
    x = np.vstack([x, plus, minus, saddle, np.eye(1, DIM)[0], -np.eye(1, DIM)[0]])
    grad = gradient(x)
    # Full data-matrix calculation independently validates the cheap exact loss.
    residuals = x @ A.T-B
    direct_loss = np.sum(residuals**2, axis=1)-RHO/2*np.sum(x*U, axis=1)**2
    np.testing.assert_allclose(direct_loss, loss(x), atol=2e-14)
    direct_gradient = 2*residuals @ A-RHO*np.sum(x*U, axis=1, keepdims=True)*U
    np.testing.assert_allclose(direct_gradient, grad, atol=2e-14)
    # Enumerate every single-row oracle at representative points. This checks
    # exact expectations, not an approximate Monte Carlo estimate of the mean.
    for point in [x[0], saddle, plus, minus, x[-1]]:
        all_samples = row_oracle(point, np.arange(DIM))
        mean = gradient(point)
        np.testing.assert_allclose(all_samples.mean(axis=0), mean, atol=2e-14)
        variance = np.mean(np.sum((all_samples-mean)**2, axis=1))
        exact_variance = (DIM-1)*np.sum((point-CENTER)**2)
        np.testing.assert_allclose(variance, exact_variance, atol=1e-10, rtol=1e-13)
        assert variance <= (DIM-1)*311/256+1e-10
        assert np.mean(np.sum(all_samples**2, axis=1)) <= 1244+1e-10
        component_values = DIM*(A @ point-B)**2-RHO/2*np.dot(point, U)**2
        np.testing.assert_allclose(component_values.mean(), loss(point), atol=1e-14)
    # This variance is nonzero at either constrained global minimum.
    np.testing.assert_allclose((DIM-1)*np.sum((plus-CENTER)**2), 31713/256)
    assert np.all(np.linalg.norm(grad, axis=1) <= M+1e-14)
    assert np.all(np.sum(np.maximum(x, 0), axis=1) <= 1+1e-14)
    # Independent directional finite difference catches sign/factor mistakes.
    d = rng.normal(size=x.shape)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    eps = 1e-5
    fd = (loss(x+eps*d)-loss(x-eps*d))/(2*eps)
    np.testing.assert_allclose(fd, np.sum(grad*d, axis=1), atol=1e-10, rtol=1e-7)
    for beta in [1., .1, .01]:
        xb = smoothed_stationary_point(beta)
        g, h, delta = certificates(xb[None, :], beta)
        np.testing.assert_allclose(np.linalg.norm(xb, 1), 1., atol=1e-15)
        np.testing.assert_allclose(g, 0., atol=1e-15)
        np.testing.assert_allclose(delta, beta/(32*(1+7*beta/4)), atol=1e-15)
        np.testing.assert_allclose(h, -beta/(1024*(1+7*beta/4)**2), atol=1e-15)
        penalty_grad = grad + np.minimum(x, 0)/beta
        vertex = step(np.zeros_like(x), penalty_grad, beta, 1.)
        np.testing.assert_allclose(np.sum(penalty_grad*vertex, axis=1), -np.max(np.abs(penalty_grad), axis=1))
        g, h, delta = certificates(x, beta)
        assert np.all(g >= -1e-13)
        assert np.all(delta <= M*beta+np.sqrt(beta*np.maximum(g, 0))+1e-13)
        assert np.all(np.abs(h) <= g+M*delta+1e-13)
    return ["Feasible negative curvature and analytic stationary-point KKT checks",
            "Orthant changes the stationary points relative to the l1 ball alone",
            "Directional finite-difference gradient check",
            "Direct Ax-B loss and full-gradient agreement with the orthogonal-design formula",
            "Exact all-row oracle expectation, variance, and second-moment checks",
            "Finite-sum normalization and nonzero row-sampling variance at a constrained minimum",
            "Exact smoothed stationary branch: zero gap, nonzero feasibility, signed original gap",
            "Exact l1 LMO and orthant projection; pointwise transfer inequalities"]


def plot_certificates(values, final_x):
    n = np.arange(1, STEPS+1)
    gap, signed, delta = values
    # Recorded delta contains x_0,...,x_(N-1). Align feasibility with completed
    # updates 1,...,N, using the stored final iterate for the final value.
    post_update_delta = np.concatenate([
        delta[1:], np.linalg.norm(np.minimum(final_x, 0.), axis=1)[None, :]], axis=0)
    series = [np.cumsum(gap, axis=0)/n[:, None],
              np.cumsum(np.abs(signed), axis=0)/n[:, None], post_update_delta]
    titles = ["Cumulative smoothed gap", "Cumulative original gap magnitude",
              "Per-iteration feasibility"]
    labels = [r"$N^{-1}\sum_{k<N}G_k$", r"$N^{-1}\sum_{k<N}|H_k|$",
              r"$\mathrm{dist}_D(x_N)$"]
    display = np.unique(np.r_[np.arange(1, 100),
                              np.geomspace(100, STEPS, 4000).astype(int),
                              ZOOM_START, REFERENCE_ANCHOR])-1
    colors = plt.cm.viridis(np.linspace(.08, .92, len(SEEDS)))
    for zoom in [False, True]:
        shown_indices = display[n[display] >= ZOOM_START] if zoom else display
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for axis, curve, title, label in zip(axes, series, titles, labels):
            mean = curve.mean(axis=1)
            se = curve.std(axis=1, ddof=1)/np.sqrt(len(SEEDS))
            for s in range(len(SEEDS)):
                shown = np.where(curve[shown_indices, s] > 0, curve[shown_indices, s], np.nan)
                axis.loglog(n[shown_indices], shown, color=colors[s], lw=.8, alpha=.65)
            axis.fill_between(n[shown_indices], np.maximum(mean[shown_indices]-2*se[shown_indices], 1e-15),
                              mean[shown_indices]+2*se[shown_indices],
                              where=mean[shown_indices] > 0, color="black", alpha=.15)
            axis.loglog(n[shown_indices], np.where(mean[shown_indices] > 0, mean[shown_indices], np.nan),
                        color="black", lw=2, label="8-seed mean; band ±2 SE")
            if not zoom and mean[0] > 0:
                axis.plot(n[0], mean[0], ".", color="black", ms=4)
            if zoom:
                for exponent, style, color, reference_label in [
                        (-1/3, "--", "#d95f02", r"$N^{-1/3}$ reference"),
                        (-1/4, "-.", "#ff0000", r"$N^{-1/4}$ reference")]:
                    reference = mean[REFERENCE_ANCHOR-1]*(n[shown_indices]/REFERENCE_ANCHOR)**exponent
                    axis.loglog(n[shown_indices], reference, style, color=color, lw=1.8,
                                label=reference_label)
            axis.set(xlabel="Sampled rows / updates N", ylabel=label, title=title,
                     xlim=(ZOOM_START if zoom else 1, STEPS))
            axis.grid(True, which="major", alpha=.15)
        axes[0].legend(fontsize=8,
                       title=f"Anchored at N = {REFERENCE_ANCHOR:,}" if zoom else None,
                       title_fontsize=8)
        fig.suptitle("From N = 1,000: reference slopes anchored at N = 50,000" if zoom
                     else "Nonconvex 1024D least squares: full run", fontsize=13)
        filename = "gaps_feasibility_zoom" if zoom else "gaps_feasibility"
        fig.savefig(OUT/f"{filename}.png", dpi=180)
        fig.savefig(OUT/f"{filename}.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
    tail = slice(STEPS-LAST_WINDOW, STEPS)
    for axis, raw, title, label in zip(axes, [gap, signed, delta],
                                      ["True smoothed gap", "Signed original gap", "Feasibility violation"],
                                      [r"$G_k$", r"$H_k$", r"$\mathrm{dist}_D(x_k)$"]):
        for s in range(len(SEEDS)):
            axis.plot(n[tail]-1, raw[tail, s], color=colors[s], lw=.6, alpha=.6)
        axis.set(xlabel="Iteration k", ylabel=label, title=title,
                 xlim=(STEPS-LAST_WINDOW, STEPS-1))
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(True, alpha=.15)
    fig.suptitle(f"Instantaneous certificates: last {LAST_WINDOW:,} iterations", fontsize=13)
    fig.savefig(OUT/"instantaneous_certificates.png", dpi=180)
    fig.savefig(OUT/"instantaneous_certificates.pdf")
    plt.close(fig)


def main():
    started = time.perf_counter()
    checks = validate_math()
    n = np.arange(1, STEPS+1)
    beta, alpha, gamma = n**(-.25), n**(-.5), n**(-.75)
    row_indices = np.stack([np.random.default_rng(s).integers(DIM, size=STEPS) for s in SEEDS], axis=1)
    values = np.empty((3, STEPS, len(SEEDS)))
    initial_x, initial_coordinates = initial_points(SEEDS)
    x = initial_x.copy()
    momentum = np.zeros_like(x)
    max_norm = float(np.max(np.sum(np.abs(initial_x), axis=1)))
    for k in range(STEPS):
        grad = gradient(x)
        sample = row_oracle(x, row_indices[k])
        momentum = sample if k == 0 else (1-alpha[k])*momentum+alpha[k]*sample
        values[:, k] = certificates(x, beta[k], grad)
        x = step(x, momentum, beta[k], gamma[k])
        max_norm = max(max_norm, float(np.max(np.sum(np.abs(x), axis=1))))
    optimizer_seconds = time.perf_counter()-started
    gap, signed, delta = values
    np.testing.assert_allclose(delta[0], 1., atol=1e-15)
    assert max_norm <= 1+1e-12
    assert gap.min() >= -1e-12
    assert np.all(delta <= M*beta[:, None]+np.sqrt(beta[:, None]*np.maximum(gap, 0))+1e-12)
    assert np.all(np.abs(signed) <= gap+M*delta+1e-12)
    avg_gap = np.cumsum(gap, axis=0)/n[:, None]
    avg_delta = np.cumsum(delta, axis=0)/n[:, None]
    avg_magnitude = np.cumsum(np.abs(signed), axis=0)/n[:, None]
    avg_beta = np.cumsum(beta)/n
    assert np.all(avg_delta.mean(axis=1) <= M*avg_beta+np.sqrt(avg_beta*avg_gap.mean(axis=1))+1e-12)
    assert np.all(avg_magnitude.mean(axis=1) <= avg_gap.mean(axis=1)+M*avg_delta.mean(axis=1)+1e-12)
    checks += ["All iterates remain in C and all true smoothed gaps are nonnegative",
               "Pointwise and cumulative transfer inequalities hold for all recorded iterates",
               "Random starts are in C and have orthant violation exactly one"]
    table = []
    for horizon in [64, 256, 1024, 4096, 16384, 32768, 65536, 131072]:
        index = horizon-1
        table.append({"updates": horizon,
                      "mean_cumulative_true_gap": float(avg_gap[index].mean()),
                      "seed_standard_error_true_gap": float(avg_gap[index].std(ddof=1)/np.sqrt(len(SEEDS))),
                      "mean_cumulative_abs_original_gap": float(avg_magnitude[index].mean()),
                      "mean_cumulative_feasibility": float(avg_delta[index].mean())})
    with (OUT/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    np.savez_compressed(OUT/"results.npz", A=A, B=B, target=CENTER, u=U, rho=RHO,
                        data_seed=DATA_SEED, column_signs=COLUMN_SIGNS.astype(np.int8), seeds=SEEDS,
                        sampled_rows=row_indices.astype(np.int16), beta=beta, alpha=alpha, gamma=gamma,
                        true_smoothed_gap=gap, signed_original_gap=signed, feasibility=delta,
                        initial_x=initial_x, initial_coordinates=initial_coordinates,
                        final_x=x)
    metadata = {"dimension": DIM, "steps": STEPS, "seeds": SEEDS,
                "problem": "min_{||x||_1<=1} ||A x-B||^2 - (5/8)*(u^T x)^2 + indicator_R_+^1024(x)",
                "data": {"matrix_shape": [DIM, DIM], "matrix": "A=H_1024 diag(c)/sqrt(2048), H is the Sylvester Hadamard matrix",
                         "target": "B=A a; a_j=1/32 on first64, -3/128 on next64, 1/128 otherwise",
                         "column_signs": "c=2*default_rng(20261014).integers(0,2,1024)-1; sampled once, then fixed",
                         "data_seed": DATA_SEED, "identity": "A^T A=I/2; row norm squared=1/2"},
                "u": "Alternating +1/8,-1/8 on first64 entries, zero elsewhere",
                "nonconvexity": "Hessian I-(5/4)uu^T has eigenvalues -1/4 and 1; feasible negative curvature at the uniform first64 stationary point",
                "source_manuscript": str(SOURCE),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "source_theorem": "Proposition 2.11 and pointwise Lemma 2.2 transfer bounds",
                "schedules": {"gamma": "(k+1)^(-3/4)", "beta": "(k+1)^(-1/4)", "alpha": "(k+1)^(-1/2)"},
                "prefactors": "All one; no tuning, no warm start, no schedule shift",
                "initialization": {"distribution": "x0=-e_J, J uniform on all1024 coordinates; m0=first sampled gradient",
                                   "initialization_rng": "default_rng(SeedSequence([20261014, seed, 1024]))",
                                   "oracle_rng": f"default_rng(seed).integers(1024,size={STEPS})",
                                   "independence": "Initialization and subsequent oracle sampling use separate random streams",
                                   "initial_coordinates_zero_based": initial_coordinates.tolist(),
                                   "feasibility": "||x0||_1=1, so x0 is in C; dist_D(x0)=1 is allowed",
                                   "theorem_application": "Condition on the realized x0. The same uniform constants also bound averages over initialization randomness."},
                "oracle": {"sample": "q_i(x)=2*1024*(A_i x-B_i)*A_i^T-(5/4)*(u^T x)u, i uniform among rows",
                           "finite_sum": "f_i=1024*(A_i x-B_i)^2-(5/8)*(u^T x)^2; f=(1/1024) sum_i f_i",
                           "noise": "Only IID uniform row indices; no additive noise, no block selection",
                           "conditional_mean": "grad f(x)=x-a-(5/4)*(u^T x)u",
                           "conditional_variance": "1023*||x-a||^2",
                           "variance_at_each_original_minimum": 31713/256},
                "constants": {"global_gradient_lipschitz": 1., "component_gradient_lipschitz_bound": 1024.,
                              "oracle_variance_bound": (DIM-1)*311/256,
                              "gradient_bound_on_C": float(M), "oracle_second_moment_bound": 1244.,
                              "diameter_C": 2., "operator_norm_T": 1., "regularity_kappa": 1.,
                              "objective_lower_bound_on_C": -5/512, "objective_upper_bound_on_C": 311/512},
                "gap_definition": "True gradient of f + ||x_-||^2/(2beta), LMO over l1ball; never use noisy momentum as the reporting gradient",
                "original_gap_definition": "H=<grad f(x),x>+max(0,-min grad f(x)); signed off D; plot average absolute magnitude",
                "averaging": "G and |H|: cumulative average of every pre-update x_0 through x_(N-1), then mean across seeds. Feasibility: delta_N at x_N after N updates, mean across seeds only; no time average.",
                "feasibility_indexing": "Long-range panels use stored feasibility[1:] for x_1 through x_(STEPS-1), followed by ||min(final_x,0)||_2 for x_STEPS. Raw stored feasibility and instantaneous plots retain their original x_k indexing.",
                "log_plot_zero_handling": "Exactly zero entries, if any, are omitted, never replaced by positive data",
                "plotted_references": {"exponents": [-1/3, -1/4],
                                       "meaning": "Comparison slopes, not numerical upper bounds",
                                       "colors": {"N^(-1/3)": "#d95f02", "N^(-1/4)": "#ff0000"},
                                       "anchor": f"Both guides pass through each mean curve at N={REFERENCE_ANCHOR}",
                                       "shown_in": f"Zoomed panels only, N>={ZOOM_START}; full-range panels have no reference lines"},
                "long_range_display": "Fixed logarithmic display grid; all iterates used in gap statistics; no time smoothing of feasibility. Zoom crops the original curves without restarting cumulative averages.",
                "instantaneous_plot": {"window": LAST_WINDOW,
                                       "first_iteration": STEPS-LAST_WINDOW,
                                       "last_iteration": STEPS-1,
                                       "subsampling": "None; every recorded iterate in the window is plotted"},
                "guarantees": {"literal_equation65": "N^-1 sum E G_k=O(N^-1/4)",
                               "output_rule": "Conditional on each fixed x0, k_n(x0) in argmin_{ceil(n/2)<=k<=n} E[G_k|x0]; expectation over oracle draws, not a pathwise argmin",
                               "cumulative_original_diagnostics": "Derived from pathwise transfers: avg_delta<=M avg_beta+sqrt(avg_beta avg_G), avg_abs_H<=avg_G+M avg_delta; hence expectation O(N^-1/4)",
                               "long_horizon_output": "Cumulative gaps and per-iteration feasibility shown; the feasibility trace is an empirical diagnostic, not the theorem's expectation-selected output",
                               "output_computation": "The expectation-defined index is left formal. Binary-tree checks from the old oracle are inapplicable: row histories branch1024ways. Report cumulative Eq65, not a per-run minimum or last iterate substitute.",
                               "limitations": "No monotonicity or last-iterate rate claimed. Empirical finite-seed trends do not prove the theorem."},
                "checks": checks, "summary": table,
                "optimizer_seconds_including_certificates": optimizer_seconds,
                "python_executable": sys.executable, "numpy_version": np.__version__,
                "matplotlib_version": matplotlib.__version__,
                "draft_caveats": ["Use positive penalty gradient (x-proj_D(x))/beta; Algorithm1 printed sign is opposite",
                                  "Use fixed-schedule Proposition2.11, not adaptive Algorithm1",
                                  "Raw oracle unbiasedness and bounded variance are explicit here; draft leaves hypotheses incomplete",
                                  "Common Lf=5/4 safely bounds both smoothness and gradient magnitude in the draft's notation"]}
    plot_certificates(values, x)
    metadata["total_seconds"] = time.perf_counter()-started
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"optimizer_seconds": optimizer_seconds, "total_seconds": metadata["total_seconds"],
                      "initial_coordinates_zero_based": initial_coordinates.tolist(),
                      "summary": table, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
