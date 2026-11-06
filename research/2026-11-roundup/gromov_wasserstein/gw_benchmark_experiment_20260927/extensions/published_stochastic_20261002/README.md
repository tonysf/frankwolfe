# Published stochastic baseline replacement

Prepared 2 October 2026. This directory implements the replacement for the custom momentum-KL primary comparator. GPU validation and scientific campaign results are recorded separately. Historical results retain their original identities; see [the scientific correction](../../reports/COMPARISON_SCOPE_CORRECTION.md).

## Method and source

Use **stochastic entropic mirror descent (Fatkhullin–He, AISTATS 2024, equation (2), Theorem 4.3)**. [Article](https://proceedings.mlr.press/v238/fatkhullin24a.html), [paper](https://proceedings.mlr.press/v238/fatkhullin24a/fatkhullin24a.pdf).

For the frozen, full unregularized squared-loss GW objective, set the paper's convex regularizer to zero, the constraint set to U(a,b), and the distance-generating function to negative Shannon entropy. Its Bregman divergence on mass-one couplings is KL. Use a fresh independent q=128 Rademacher sketch gradient at each iterate:

\[
P_{k+1}=\arg\min_{P\in U(a,b)}
\left\{\langle\widehat\nabla f(P_k),P\rangle
+\eta_k^{-1}D_{\rm KL}(P\|P_k)\right\}.
\]

There is no gradient EMA, extrapolation, entropy penalty in the objective, incumbent selection, or added smoothing schedule. This is an application of the paper's general method to GW, not a reproduction of a GW experiment conducted in that paper.

## Assumptions and parameter admissibility

The GW loss is a nonnegative smooth quadratic; U(a,b) is compact and convex, with positive product initialization a*b^T. Existing sketch kernels differentiate the symmetrized scalar sketch objective. Independence and E[ZZ^T/q]=I give conditional unbiasedness; compactness and finite Rademacher support give uniformly bounded gradient variance in the dual norm for fixed q.

The following is our direct verification of the paper's relative-smoothness assumption for the canonical GW objective, not a new algorithm or a fitted parameter. Define

\[
C_{\max}=\max\{|\min A-\max B|,|\max A-\min B|\}^2,
\qquad \ell=2C_{\max}.
\]

Writing H=Q-P, the absolute quadratic Taylor remainder is at most Cmax*||H||_1^2. Pinsker's inequality bounds this by 2*Cmax*KL(Q||P). This supplies both relative-smoothness inequalities with ell=2*Cmax and the entrywise l1 norm, whose dual is the entrywise maximum norm. The bound uses the full objective with no half factor. For canonical distances in [0,1], Cmax<=1; use the actual canonical extrema.

The theorem permits nonincreasing positive step sizes with eta0<=1/(2*ell). A concrete admissible family is

\[
\eta_k=\frac{c}{4C_{\max}\sqrt{k+1}},\qquad 0<c\le1.
\]

The exponent and prefactor bound come from this theorem specialization, not from the manuscript FW schedule. The prefactor c may be tuned within its admissible interval using a prospective grid and budget. Cmax=0 is a constant-zero objective and must be explicitly identified instead of dividing by zero.

**FW prefactor tuning remains allowed.** The user clarified that its requirements are asymptotic; unit gamma/rho prefactors are not required for fairness. Preserve the established exponents, estimator and update. Use each method's own admissible parameter family; do not graft the FW momentum rule onto SMD or demand identical algorithm-specific schedules. Keep common data, initialization, oracle access and declared resource budgets where applicable.

## Output and interpretation

Theorem 4.3 uses a fixed, prespecified iteration horizon T and returns a pre-update iterate P_R with probability eta_R/sum(eta_0,...,eta_(T-1)), R in {0,...,T-1}. Draw the index with randomness independent of optimizer probes and tuning. Its guarantee concerns an expected Bregman forward-backward-envelope stationarity measure, not global optimality or the last-iterate loss. A run stopped at a random wall-clock horizon does not automatically inherit this theorem's output guarantee.

Report that statistic/output convention separately from empirical current-iterate feasible GW loss, elapsed solver time, rounding cost and gradient-sample counts. The FW manuscript's own theorem statistic is the full-trajectory mean of true smoothed FW gaps. Different stationarity measures do not become numerically comparable by sharing a plot label. Feasible GW loss remains a common empirical metric; neither theorem supplies a global-optimum guarantee for it.

## Numerical implementation and execution status

The user selected **1e-7 for both stopping criteria**: summed L1 marginal residual and normalized primal-dual gap. [inner_solver.json](inner_solver.json) records the values and gap normalization. Both criteria must pass; failure never loosens the tolerance. The complete runner is `run_published.py` and the inner solver is `certified_prox.py`.

The inner solver uses FP64 log-state, Sinkhorn vectors, blocked arithmetic and certificate reductions, while the production oracle and stored coupling use FP32. It audits the actual stored coupling's marginals with FP64 sums. A separately rounded FP64 feasible witness supplies the primal value, and the unshifted Sinkhorn dual supplies a lower bound for the original objective, including generalized-KL mass constants. The witness's distance from the stored coupling is recorded. It cross-checks the signed primal-minus-dual gap against an independent stable KL expression and accepts only when the larger gap plus its explicit FP64 roundoff allowance, divided by max(1, abs(primal), abs(dual)), is at most 1e-7. These are floating-point numerical certificates, not interval bounds.

The cap is 10,000 Sinkhorn iterations, with both criteria checked every 10 iterations and at the cap. Failed attempts retain their certificate and work counts, exit nonzero and are not accepted as updates. No dense coupling is gathered across GPUs. FP64 logs also preserve entries that underflow in the FP32 coupling.

Equation (2) specifies an exact proximal minimizer. The paper does not supply an inexact-prox theorem, so the selected finite tolerances do not establish the paper's outer guarantee in finite precision.

The runner uses a prespecified iteration horizon, fresh q128 gradients and independent output selection. It saves the selected pre-update coupling and last iterate separately, along with repaired versions for empirical diagnostics. Source/data hashes, gradient counts, inner iteration/certificate counts, underflows and precision are recorded. All inner solves and acceptance checks count toward solver time. Exact GW diagnostics, output repair, serialization and final FP64 validation have separate timing counters. A time checkpoint never changes the horizon or chooses the output. The same runner preserves the FW schedules and permitted prefactors; its sampled FW snapshot is explicitly only a diagnostic, not a claim about the FW theorem statistic.

## Validation and running

The local CPU suite passes **47 tests**, covering independent constrained optima, fresh-gradient trajectories, independent output randomness, cap failures, absent/false certificates, small-step roundoff, underflow recovery, and uneven/empty distributed shards. Run it with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q extensions/published_stochastic_20261002
```

`validate_distributed.py` exercises two successive updates in FP32 and FP64, independent small-instance solutions and collective failure handling. `validate.sbatch` runs the suite and four-A100 checks, then three-update integration runs of each method on the full 31,890-cell SHARE-seq instance. These short runs validate precision, memory and instrumentation; they do not establish a performance ranking. Submit only an immutable, checksum-recorded source bundle under the documented Ruche account. Prospective tuning and final scientific runs follow successful GPU validation; historical momentum-KL measurements cannot be reused as SMD results.

Four-A100 validation completed successfully on 2 October 2026 (Ruche job 1986380): 47 CPU tests, 40 distributed GPU checks, and three updates of each method on all 31,890 SHARE-seq cells. Every SMD update met both selected 1e-7 criteria at the first check, after 10 Sinkhorn iterations. These are short integration tests, not a sustained speed or quality comparison. See the [validation record](../../reports/PUBLISHED_SMD_VALIDATION.json) for certificates, timings, memory and preserved failure history.

## Frozen real-data campaign

The 4 October submission uses the existing BMMC 6,224-cell, SNARE-seq 10,000-cell and full SHARE-seq 31,890-cell instances. `campaign.json` freezes source metadata hashes and the entire design before any new tuning results. The smaller two instances use one A100 per run; SHARE-seq uses four A100s. Paired methods run sequentially in the same allocation. Site limits cap concurrent A100 use at four.

Each dataset/method receives twelve candidates, three development seeds (710–712), and both product and randomized initializations. All **432 tuning runs have T=200**. FW varies gamma0 in {0.1,0.3,1} and mu_scale in {0.1,1,10,100}, with rho0=0.3 and unchanged exponents. SMD varies only its admissible prefactor fraction over twelve logarithmically spaced values from 0.01 through 1. Both inner tolerances remain 1e-7. One candidate per dataset/method minimizes the median FP64 last-rounded GW loss over the six development cells; ties use candidate ID. Every selected cell must pass validation. Infrastructure or numerical failures stop the affected task and block its dependent selection/finals rather than becoming accepted outputs.

The **36 final runs have T=2,000**, three fresh seeds (810–812), and both initializations. This is an explicit fixed-iteration, equal-oracle-work study: candidates share q128 and the same T, but elapsed time is not equal. This revises the earlier custom-baseline campaigns' time-budget stopping so the published method retains its prescribed fixed-horizon random output. Time checkpoints at 10/30/100/300 seconds are supplementary observations at their actual times; a trajectory ending sooner has missing later checkpoints, with no extrapolation, automatic extension, or winner-selected stopping. Current/last rounded losses and the independently sampled raw SMD output are reported separately. Seed repetitions are not independent biological datasets. No full-trajectory FW-gap or global-optimality claim is made.

The controller binds source, settings, input arrays, seeds, rank records and all numerical certificates. Data repackaging occurs in a CPU allocation and preserves every canonical array value; SHARE-seq is linked to the already prepared full instance. A scheduler dependency chain submits preparation, all tuning tasks, per-dataset selection, all final tasks and the final report at once. Reports remain explicit about failed, blocked or missing work.

To fit the current workdir quota, tuning persists metrics and hashes but no dense plans. Finals retain all four selected/last raw/rounded snapshots for BMMC and SNARE; SHARE-seq retains dense snapshots for seed 810 and both initializations, while seeds 811–812 retain value hashes and all FP64 validated metrics. This adds approximately 92 GB of dense outputs. No previous experiment data is removed. The storage option is tested to preserve the numerical trajectory and all final validation metrics.
