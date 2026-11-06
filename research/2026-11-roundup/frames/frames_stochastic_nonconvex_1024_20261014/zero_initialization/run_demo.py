#!/usr/bin/env python3
"""Nonconvex stochastic FRAMES: 1024D, gaps and feasibility only.

Requirements: numpy, matplotlib. Run: python run_demo.py
Each stochastic component is a least-squares block plus the same concave
rank-one quadratic. The raw gradient oracle is conditionally unbiased;
the momentum estimate generally is not. All schedule prefactors are one.
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
DIM, STEPS, EXACT_HORIZON = 1024, 32768, 10
SEEDS = list(range(8))
RHO = 5 / 4
A = np.r_[np.full(64, 1/32), np.full(64, -3/128), np.full(896, 1/128)]
B = np.where(np.arange(DIM) % 2 == 0, 1., -1.) / 64
U = np.r_[np.where(np.arange(64) % 2 == 0, 1., -1.) / 8, np.zeros(960)]
M = np.sqrt(311) / 16


def gradient(x):
    return x - A - RHO * np.sum(x * U, axis=-1, keepdims=True) * U


def loss(x):
    return .5 * np.sum((x-A)**2, axis=-1) + 1/8 - RHO/2 * np.sum(x*U, axis=-1)**2


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
    np.testing.assert_allclose(U @ A, 0., atol=1e-16)
    np.testing.assert_allclose(U @ B, 1/8)
    np.testing.assert_allclose(B @ B, .25)
    # The Hessian acts as -1/4 along U, and as identity perpendicular to U.
    hessian_u = U - RHO * np.dot(U, U) * U
    np.testing.assert_allclose(hessian_u, -U/4, atol=1e-16)
    saddle, plus, minus = analytic_stationary_points()
    np.testing.assert_allclose(loss(saddle), 91/512)
    np.testing.assert_allclose(loss(np.stack([plus, minus])), 45/256)
    for t in [-.1, -.05, 0., .05, .1]:
        z = saddle + t*U
        assert z.min() >= 0 and np.linalg.norm(z, 1) <= 1+1e-14
        np.testing.assert_allclose(loss(z), 91/512-t*t/8, atol=1e-15)
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
    qminus, qplus = grad-B, grad+B
    np.testing.assert_allclose((qminus+qplus)/2, grad, atol=1e-16)
    variance = .5*(np.sum((qminus-grad)**2, axis=1)+np.sum((qplus-grad)**2, axis=1))
    np.testing.assert_allclose(variance, .25, atol=1e-15)
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
            "Uniform two-component oracle mean and constant variance identities",
            "Exact smoothed stationary branch: zero gap, nonzero feasibility, signed original gap",
            "Exact l1 LMO and orthant projection; pointwise transfer inequalities"]


def exact_expected_output(n):
    x = np.zeros((1, DIM))
    momentum = np.zeros_like(x)
    gaps, signed, magnitudes, deltas = [], [], [], []
    for k in range(n+1):
        beta = (k+1)**(-.25)
        gap, original, delta = certificates(x, beta)
        gaps.append(float(gap.mean()))
        signed.append(float(original.mean()))
        magnitudes.append(float(np.abs(original).mean()))
        deltas.append(float(delta.mean()))
        if k == n:
            break
        x = np.repeat(x, 2, axis=0)
        momentum = np.repeat(momentum, 2, axis=0)
        noise = np.tile([-1., 1.], len(x)//2)
        sample = gradient(x) + noise[:, None]*B
        alpha = (k+1)**(-.5)
        momentum = sample if k == 0 else (1-alpha)*momentum+alpha*sample
        x = step(x, momentum, beta, (k+1)**(-.75))
    lo = (n+1)//2
    chosen = lo+int(np.argmin(gaps[lo:n+1]))
    return {"horizon": n, "last_half_inclusive": [lo, n], "selected_index": chosen,
            "histories_at_horizon": 2**n, "expected_true_gap": gaps,
            "expected_signed_original_gap": signed, "expected_abs_original_gap": magnitudes,
            "expected_feasibility": deltas,
            "method": "Full binary-tree enumeration, exact expectation up to floating point arithmetic"}


def plot_certificates(values):
    n = np.arange(1, STEPS+1)
    gap, signed, delta = values
    series = [gap, np.abs(signed), delta]
    titles = ["True smoothed gap", "Original gap magnitude", "Feasibility violation"]
    labels = [r"$N^{-1}\sum_{k<N}G_k$", r"$N^{-1}\sum_{k<N}|H_k|$",
              r"$N^{-1}\sum_{k<N}\mathrm{dist}_D(x_k)$"]
    display = np.unique(np.r_[np.arange(1, 100), np.geomspace(100, STEPS, 700).astype(int)])-1
    colors = plt.cm.viridis(np.linspace(.08, .92, len(SEEDS)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, raw, title, label in zip(axes, series, titles, labels):
        cumulative = np.cumsum(raw, axis=0)/n[:, None]
        mean = cumulative.mean(axis=1)
        se = cumulative.std(axis=1, ddof=1)/np.sqrt(len(SEEDS))
        for s in range(len(SEEDS)):
            shown = np.where(cumulative[display, s] > 0, cumulative[display, s], np.nan)
            axis.loglog(n[display], shown, color=colors[s], lw=.8, alpha=.65)
        axis.fill_between(n[display], np.maximum(mean[display]-2*se[display], 1e-15),
                          mean[display]+2*se[display], where=mean[display] > 0, color="black", alpha=.15)
        axis.loglog(n[display], np.where(mean[display] > 0, mean[display], np.nan),
                    color="black", lw=2, label="8-seed mean; band ±2 SE")
        reference = mean[1023]*(n/1024)**(-.25)
        axis.loglog(n[display], reference[display], "--", color="#d95f02", lw=1.8,
                    label=r"$N^{-1/4}$ slope guide")
        axis.set(xlabel="Sampled component gradients / updates N", ylabel=label, title=title)
        positive = mean[mean > 1e-14]
        axis.set_ylim(bottom=max(float(positive.min())*.7, 1e-8))
        axis.grid(True, which="major", alpha=.15)
    axes[0].legend(fontsize=8)
    axes[2].text(.03, .97, "Initial zero violations omitted on log axis", transform=axes[2].transAxes,
                 va="top", fontsize=8)
    fig.suptitle("Nonconvex 1024D: cumulative gap and feasibility certificates", fontsize=13)
    fig.savefig(OUT/"gaps_feasibility.png", dpi=180)
    fig.savefig(OUT/"gaps_feasibility.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
    tail = slice(STEPS-2048, STEPS)
    for axis, raw, title, label in zip(axes, [gap, signed, delta],
                                      ["True smoothed gap", "Signed original gap", "Feasibility violation"],
                                      [r"$G_k$", r"$H_k$", r"$\mathrm{dist}_D(x_k)$"]):
        for s in range(len(SEEDS)):
            axis.plot(n[tail]-1, raw[tail, s], color=colors[s], lw=.6, alpha=.6)
        axis.set(xlabel="Iteration k", ylabel=label, title=title)
        axis.grid(True, alpha=.15)
    fig.suptitle("Instantaneous certificates: fluctuations are permitted", fontsize=13)
    fig.savefig(OUT/"instantaneous_certificates.png", dpi=180)
    fig.savefig(OUT/"instantaneous_certificates.pdf")
    plt.close(fig)


def main():
    started = time.perf_counter()
    checks = validate_math()
    n = np.arange(1, STEPS+1)
    beta, alpha, gamma = n**(-.25), n**(-.5), n**(-.75)
    noise = np.stack([2*np.random.default_rng(s).integers(0, 2, STEPS)-1 for s in SEEDS], axis=1)
    values = np.empty((3, STEPS, len(SEEDS)))
    x = np.zeros((len(SEEDS), DIM))
    momentum = np.zeros_like(x)
    max_norm = 0.
    for k in range(STEPS):
        grad = gradient(x)
        sample = grad+noise[k, :, None]*B
        momentum = sample if k == 0 else (1-alpha[k])*momentum+alpha[k]*sample
        values[:, k] = certificates(x, beta[k], grad)
        x = step(x, momentum, beta[k], gamma[k])
        max_norm = max(max_norm, float(np.max(np.sum(np.abs(x), axis=1))))
    optimizer_seconds = time.perf_counter()-started
    gap, signed, delta = values
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
               "Pointwise and cumulative transfer inequalities hold for all recorded iterates"]
    exact = exact_expected_output(EXACT_HORIZON)
    chosen = exact["selected_index"]
    selected_x = np.zeros((len(SEEDS), DIM))
    selected_momentum = np.zeros_like(selected_x)
    for k in range(chosen):
        sample = gradient(selected_x)+noise[k, :, None]*B
        selected_momentum = sample if k == 0 else (1-alpha[k])*selected_momentum+alpha[k]*sample
        selected_x = step(selected_x, selected_momentum, beta[k], gamma[k])
    checks.append("Theorem output index computed by full expectation at horizon 10")
    table = []
    for horizon in [64, 256, 1024, 4096, 16384, 32768]:
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
    np.savez_compressed(OUT/"results.npz", a=A, b=B, u=U, rho=RHO, seeds=SEEDS,
                        sample_noise=noise.astype(np.int8), beta=beta, alpha=alpha, gamma=gamma,
                        true_smoothed_gap=gap, signed_original_gap=signed, feasibility=delta,
                        final_x=x, exact_output_index=chosen, exact_selected_seed_outputs=selected_x)
    metadata = {"dimension": DIM, "steps": STEPS, "seeds": SEEDS,
                "problem": "min_{||x||_1<=1} .25*(||x-a-b||^2+||x-a+b||^2) - (5/8)*(u^T x)^2 + indicator_R_+^1024(x)",
                "u": "Alternating +1/8,-1/8 on first64 entries, zero elsewhere",
                "nonconvexity": "Hessian I-(5/4)uu^T has eigenvalues -1/4 and 1; feasible negative curvature at the uniform first64 stationary point",
                "source_manuscript": str(SOURCE),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "source_theorem": "Proposition 2.11 and pointwise Lemma 2.2 transfer bounds",
                "schedules": {"gamma": "(k+1)^(-3/4)", "beta": "(k+1)^(-1/4)", "alpha": "(k+1)^(-1/2)"},
                "prefactors": "All one; no tuning, no warm start, no schedule shift",
                "initialization": "x0=0; m0=first sampled gradient",
                "oracle": "IID uniform component sampling; grad f(x) plus independent Rademacher times b; covariance bb^T",
                "constants": {"global_gradient_lipschitz": 1., "oracle_variance": .25,
                              "gradient_bound_on_C": float(M), "oracle_second_moment_bound": 375/256,
                              "diameter_C": 2., "operator_norm_T": 1., "regularity_kappa": 1.,
                              "objective_lower_bound_on_C": 59/512, "objective_upper_bound_on_C": 375/512},
                "gap_definition": "True gradient of f + ||x_-||^2/(2beta), LMO over l1ball; never use noisy momentum as the reporting gradient",
                "original_gap_definition": "H=<grad f(x),x>+max(0,-min grad f(x)); signed off D; plot average absolute magnitude",
                "averaging": "Every pre-update x_0 through x_(N-1), separately per seed, then average seeds",
                "log_plot_zero_handling": "Exactly zero entries are omitted, never replaced by positive data; initial cumulative feasibility is zero at N=1,2",
                "guarantees": {"literal_equation65": "N^-1 sum E G_k=O(N^-1/4)",
                               "output_rule": "k_n in argmin_{ceil(n/2)<=k<=n} E G_k; common deterministic index, not per-run argmin",
                               "cumulative_original_diagnostics": "Derived from pathwise transfers: avg_delta<=M avg_beta+sqrt(avg_beta avg_G), avg_abs_H<=avg_G+M avg_delta; hence expectation O(N^-1/4)",
                               "long_horizon_output": "Only cumulative certificates reported; no claim of computing exact long-horizon expectation-selected output",
                               "limitations": "No monotonicity or last-iterate rate claimed. Empirical finite-seed trends do not prove the theorem."},
                "exact_output_demonstration": exact, "checks": checks, "summary": table,
                "optimizer_seconds_including_certificates": optimizer_seconds,
                "python_executable": sys.executable, "numpy_version": np.__version__,
                "matplotlib_version": matplotlib.__version__,
                "draft_caveats": ["Use positive penalty gradient (x-proj_D(x))/beta; Algorithm1 printed sign is opposite",
                                  "Use fixed-schedule Proposition2.11, not adaptive Algorithm1",
                                  "Raw oracle unbiasedness and bounded variance are explicit here; draft leaves hypotheses incomplete",
                                  "Common Lf=5/4 safely bounds both smoothness and gradient magnitude in the draft's notation"]}
    plot_certificates(values)
    metadata["total_seconds"] = time.perf_counter()-started
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"optimizer_seconds": optimizer_seconds, "total_seconds": metadata["total_seconds"],
                      "exact_output_index": chosen, "summary": table, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
