# Method names and update rules

**Theory status corrected on 2 October 2026:** `stochastic-kl-momentum` denotes our custom exploratory EMA comparator, not a faithful reproduction of a published convergence theorem. It is excluded from the replacement primary comparison. Historical IDs and measurements are preserved; see [the interpretation correction](COMPARISON_SCOPE_CORRECTION.md).

The display names below describe the implemented algorithms. Machine-readable method IDs remain unchanged. “FW” means Frank–Wolfe; “KL” means Kullback–Leibler divergence.

**“Exact gradient” and “full gradient” mean the same thing in this experiment:** the analytical gradient of the full GW objective, evaluated with floating-point arithmetic and without sampling objective terms or random sketch probes. Use **exact gradient** consistently in display names. This does not mean exact real arithmetic, an exact inner optimization solve, or a globally optimal result.

Both **Transport FW + line search (exact gradient)** and **Smoothed row FW + momentum (exact gradient)** use unsampled gradient inputs. They differ in momentum, marginal constraints, penalty, and update rule; they are not two names for the same algorithm.

## Canonical names

Here, a **sketch gradient** is the Rademacher-probe estimator implemented in `GWProblem.sketch_gradient`. Native POT sampling is a separate estimator and does not use these probes.

| Method ID | Display name | Gradient input | Momentum | Constraint and update |
|---|---|---|---|---|
| `smoothing-fw` | Smoothed row FW + momentum (sketch gradient) | Fresh random sketch | EMA | Row set C; add the current column penalty gradient, then use row-wise FW with the prescribed step schedule. |
| `smoothing-fw-exact` | Smoothed row FW + momentum (exact gradient) | Current exact gradient | EMA | Same penalized row-FW algorithm, with the exact gradient oracle replacing the sketch oracle. |
| `full-kl` | Transport KL-proximal gradient (exact gradient) | Current exact gradient | None | Transport set U(a,b); KL-proximal step solved by iterative log-domain marginal scaling. |
| `stochastic-kl` | Transport KL-proximal gradient (sketch gradient) | Fresh random sketch | None | Same transport KL-proximal update, using the current sketch directly. |
| `stochastic-kl-momentum` | Transport KL-proximal gradient + momentum (sketch gradient) | Fresh random sketch | EMA | Same transport KL-proximal update, using the averaged gradient direction. No column penalty. |
| `native-sampled-gw` | Native Sampled GW (POT) | POT's conditional sampling from the current coupling, with native cost normalization | None | Native POT update on U(a,b); the positive-epsilon configuration uses a KL-type proximal/Sinkhorn step. |
| `fw` | Transport FW + line search (exact gradient) | Current exact gradient | None | Assignment linear minimization over U(a,b), followed by a quadratic line search along the FW segment. |
| `smoothing-row-projection` | Smoothed row projection + momentum (sketch gradient) | Fresh random sketch | EMA | Row set C; add the current column penalty gradient, then project a gradient step onto each row simplex. |
| `pg` | Transport projected gradient (exact gradient) | Current exact gradient | None | Euclidean projected-gradient step onto U(a,b), with an iterative Dykstra projection. |

`smoothing-fw-exact` is a reporting/manifest ID: it invokes the `smoothing-fw` implementation with `--oracle exact`. Batch size is ignored by that exact oracle. The sketch names describe the sketch configurations used for these comparisons; other oracle options exposed by the software do not alter the meaning of these recorded method IDs/configurations.

## Constraints, smoothing, and momentum

Let P be a nonnegative transport matrix, with prescribed row masses a and column masses b. The **row set** is C = {P ≥ 0 : P1 = a}. The **transport set** is U(a,b) = {P ≥ 0 : P1 = a, Pᵀ1 = b}. Thus a row-set update preserves the row constraint while allowing column error. A transport-set update targets both constraints.

The smoothed methods optimize the full GW objective f with the column penalty

\[
f_\mu(P)=f(P)+\frac{\|P^\top\mathbf 1-b\|_2^2}{2\mu}.
\]

Here “smoothed” refers to the penalty representation of the column constraint. It is not smoothing by sampling objective values. The implementation uses a decreasing, prescribed smoothing parameter μ.

For the methods marked EMA, the gradient input gₖ initializes d₀ = g₀ and then follows

\[
d_k=(1-\rho_k)d_{k-1}+\rho_k g_k,\qquad k\geq1.
\]

With the exact oracle, gₖ = ∇f(Pₖ), but **dₖ generally differs from ∇f(Pₖ)** because it retains earlier gradients. The smoothed methods use

\[
H_k=d_k+\mathbf1\bigl(P_k^\top\mathbf1-b\bigr)^\top/\mu_k.
\]

The column penalty gradient is evaluated at the current iterate and is never included in the EMA. The momentum KL comparator uses dₖ without that penalty. A leading momentum constant of one does not remove momentum when the later schedule weights are less than one.

The KL step targets the minimizer of ⟨G,Q⟩ + KL(Q‖P)/η over U(a,b), where G is the current gradient or the EMA direction. Its log-domain scaling solver has a tolerance and iteration cap. The transport Euclidean projection likewise has a tolerance and cap, monitored using its projection primal–dual gap. These inner solves can leave numerical marginal error or stop without meeting their criterion; their display names do not promise exact feasibility. Row-simplex projection has a direct formula, evaluated in floating point.

The KL divergence is a step/proximity term, not an added entropy term in the benchmark's reported GW objective. Native POT also retains its own sampling distribution, normalization, clipping, inner solver, and stopping rules; its sample counts are not interchangeable with Rademacher-probe counts. Its selected positive-epsilon update should not be relabeled as the transport assignment FW baseline merely because the native function's docstring uses FW terminology.

## Objective evaluation and diagnostics

**All reported raw and rounded GW objective values use the full, unsampled objective**, including values reported for the sketch and native sampled methods. “Sketch” and “sampled” describe the optimization oracle. They do not mean that plotted losses are Monte Carlo estimates. Final validation evaluates the actual saved raw and rounded plans in float64; it does not substitute a newly optimized plan.

Exact gradients computed for objective/gap or momentum-error diagnostics are separate from optimizer oracle calls and do not change the algorithm's trajectory. The true smoothed gap uses ∇f at its labeled current point, not the EMA as a substitute. After K updates, the retained last-used momentum is associated with the input Pₖ₋₁; its same-input error and its separately labeled tracking error against ∇f(Pₖ) are different diagnostics. No additional terminal oracle draw is implied.

The source audit covered the update branches in `run_benchmark.py`, the exact/sketch oracle and EMA in `smoothing_fw.py`, and the source-preserving native wrapper in `native_sampled_gw.py`. The loss/gradient, row update, KL step, and projection definitions are in `gw_core.py`; the reviewed native implementation is archived in `provenance/pot_ot_gromov__estimators.py`. This glossary changes presentation only, not source algorithms or recorded method identities.
