# Stochastic-FRAMES experiment suite

This package is a NumPy experiment layer around the repository's existing
`StochasticFrames` optimizer. It studies problems

\[
  \min_{x\in C}\;\mathbb E[f(x,\xi)] + g(h(x)).
\]

At iteration `k`, only the stochastic gradient of the smooth term enters the
momentum state:

\[
  d_0=\nabla f(x_0,\xi_0),\qquad
  d_k=(1-\rho_k)d_{k-1}+\rho_k\nabla f(x_k,\xi_k).
\]

The direction sent to the linear minimization oracle is

\[
  q_k=d_k+J_h(x_k)^*\frac{h(x_k)-
  \operatorname{prox}_{\beta_k g}(h(x_k))}{\beta_k}.
\]

Thus the composite Moreau term is exact and is never accumulated in momentum.
The adapter uses `linear_operator_adjoint_at` already supported by the core
optimizer, leaving its behavior and the QPT archive path unchanged.

## Registry

| ID | Registry name | Model | Constraint |
|---|---|---|---|
| E0 | `simplex_regression` | online least squares, `g=0` | simplex |
| E1 | `sparse_logistic` | logistic loss plus L1 | L2 ball |
| E2 | `tv_denoising` | coordinate least squares plus TV | box `[0,1]^d` |
| E3 | `robust_portfolio` | factor risk plus worst stress | simplex |
| E4 | `matrix_completion` | observed-entry loss, nonnegative indicator | nuclear-norm ball |
| E5 | `phase_retrieval` | quartic phase loss plus TV | L2 ball |
| E6 | `factorized_correlation` | sensing of `UU^T`, unit-diagonal indicator | Frobenius ball |

Every factory has `tiny` and `small` profiles. Tiny is intended for registry
smoke tests; small is still laptop-scale.

E6, and the quadratic-lift component it exercises, are explicitly
**experimental nonlinear-composite extensions beyond the linear-`T` FRAMES
assumptions**. They use

\[
  h(U)=UU^T,\qquad J_h(U)^*[Y]=(Y+Y^T)U.
\]

The large complex QPT benchmark remains the existing GPU scale-out experiment.

## Reproducibility and fair sampling

`problem_seed`, `initialization_seed`, and `sampling_seed` are separate.
`SamplePlan` precomputes every minibatch before optimization. `run_many` reuses
the same plan object for runs that differ only in method or schedules, so
momentum, no-momentum, and smoothing ablations see identical observations.
The deterministic baseline uses the exact full finite-sum gradient at the same
initial iterate; its sampled-observation count therefore grows by the full
population size per step.

Schedules are callables of the zero-based iteration or scalar constants:

- momentum: `min(1, rho_scale * 4/(k+8)^(2/3))`;
- smoothing: `smoothing_scale * beta0/(k+1)^(1/4)`;
- FW step: `min(1, step_scale/sqrt(k+1))`.

`no-momentum` and `deterministic` set `rho_k=1`. `fixed-smoothing` holds
`beta_k=beta0*smoothing_scale`. Python callers may provide arbitrary callable
schedules; archives store their realized numeric values rather than callables.

Exact gradients, objectives, gaps, and task metrics are evaluated at
checkpoints only after optimizer timing stops. Archives include active-beta and
fixed-reference-beta smoothed objectives/gaps, estimator error, all requested
oracle counts, checkpoint iterates, final state, and problem-specific metrics.
NPZ persistence uses `allow_pickle=False` and validates schema metadata when
loading.

Checkpoint `k < T` uses the pre-update triple `(x_k, d_k, beta_k)`, so its
oracle/time coordinate includes the work needed to form iteration `k`'s
direction. The terminal checkpoint uses `x_T` with the carried final estimate
`d_{T-1}` and smoothing value `beta_{T-1}`; it adds no oracle work. For a
Lipschitz penalty, the existing optimizer also performs its nonsmooth-gap
diagnostic LMO and map adjoint inside each timed iteration. The suite's LMO and
map counts report those physical calls even though checkpoint metrics are
post-hoc.

## Commands

Run the complete tiny registry comparison:

```bash
python -m paper.experiments.stochastic_frames_suite \
  --problem all \
  --methods momentum no-momentum deterministic fixed-smoothing \
  --seeds 0 1 2 \
  --steps 1000 \
  --batch-size 16 \
  --rho-scale 1.0 \
  --smoothing-scale 1.0 \
  --step-scale 1.0 \
  --output-dir results/stochastic_frames
```

For a quick check:

```bash
python -m paper.experiments.stochastic_frames_suite \
  --problem e0 e6 --methods momentum no-momentum \
  --seeds 0 --steps 10 --batch-size 2 --no-plots
```

The Python API exposes `ExperimentConfig`, `run_experiment`, `run_many`,
`save_result`, `load_result`, `aggregate_results`, and the plotting helpers.
