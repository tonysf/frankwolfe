# Direct FRAMES--Quiroga comparison protocol

This protocol defines the paper-quality comparison between stochastic FRAMES
and the adaptive factored-gradient update printed by Quiroga and Kyrillidis.
It uses only the Quiroga-style global probes and `2d`-outcome POVM implemented
by `qpt_quiroga_sensing.py`. Results from the local-Pauli benchmark are a
separate experiment and cannot enter this comparison.

## Methods and roles

- **Primary comparison:** stochastic FRAMES versus **printed-adaFGD**, meaning
  the update printed in Quiroga and Kyrillidis and implemented by
  `quiroga_adafgd_step`. Do not silently replace it with the line search from
  the paper's cited reference.
- **Diagnostic control:** full-gradient FGD with Armijo backtracking on the
  selected fixed printed-adaFGD objective. This establishes whether an
  instance is recoverable by ordinary FGD and helps distinguish a failure of
  the printed adaptive step from a failure of the sensing problem. It does not
  share FRAMES's time-varying penalty, is not a third primary competitor, and
  must not be reported as “Quiroga's method.”
- MiFGD, Nesterov variants, momentum SGD, and the local-Pauli runs are outside
  this direct comparison.

## Primary best-for-best lane

The primary comparison does **not** force a common TP coefficient or a common
penalized objective. This rule supersedes the earlier shared-`alpha`,
constant-`beta` protocol. Fairness comes from shared sensing data, row sets,
initial factors, ranks, logical row budgets, checkpoints, metrics, and recovery
gate, while each method receives its own calibrated TP treatment.

For active rows `S`, let `m = |S|`, let
`r_S(U) = A_S(U U^H) - y_S`, and define
`H(U) = ||Tr_out(U U^H) - I||_F^2`.

Printed-adaFGD uses the fixed objective

```text
Q_printed(U) = 0.5 ||r_S(U)||_2^2 + lambda_Q H(U).
```

Use `alpha_Q = lambda_Q / m` as the portable calibration coordinate. Tune
`alpha_Q` and `eta_scale` on calibration instances; for each run set
`lambda_Q = m * alpha_Q` once and hold it fixed over all iterations. Freeze
the selected values for every held-out run in that regime and archive both
`alpha_Q` and `lambda_Q`.

FRAMES uses its intended changing smoothing penalty. For zero-based update
index `t = 0, 1, ...`,

```text
Q_frames,t(U) = (1 / (2m)) ||r_S(U)||_2^2 + alpha_F(t) H(U)
alpha_F(t)    = alpha_F(0) * (t + 1)^p
beta_t        = 1 / (2 alpha_F(t)),                    p > 0.
```

Tune the initial penalty `alpha_F(0)` and FRAMES-specific controls on
calibration instances. Predeclare `p` (the planned default is `0.25`), or
include it explicitly in the calibration grid. The primary lane requires
`p > 0`, so the TP penalty increases and the smoothing parameter decreases;
`p = 0` is a constant-penalty control only. Freeze the initial value and
complete schedule before held-out evaluation, and archive the exact applied
`alpha_F(t)` and `beta_t` sequence. At a checkpoint after `T > 0` updates,
the last applied coefficient is `alpha_F(T - 1) = alpha_F(0) * T^p`, not
`alpha_F(T)`.

Select the best calibrated configuration separately for each method under a
predeclared, gate-based rule, then compare those frozen configurations
head-to-head on paired held-out instances. Because the objectives differ and
the FRAMES penalty changes over time, raw penalized-objective values are
method diagnostics and must not be used as a cross-method score.

## Superseded matched-penalty ablation

The former mapping `lambda = m * alpha` with constant
`beta = 1 / (2 * alpha)` may be retained only as a clearly labeled
matched-penalty ablation. It is not the primary comparison and its archives
must carry a different protocol/cohort identifier.

Earlier held-out runs with different fixed `beta` values remain exploratory,
not because unequal penalties are now forbidden, but because their tuning and
selection were not frozen under this best-for-best protocol and their seeds
have already been exposed. Do not count those seeds as unseen primary test
instances.

## Paired experimental unit

Use three nested levels and name them consistently:

- A **data/channel instance** fixes the qubit count, channel seed, observation
  mode and seed, active-row design, and subset seed. This is the independent
  scientific unit.
- A **paired initialization instance** additionally fixes the factor rank,
  initialization seed, and serialized initial factor. All methods receive
  this same factor.
- A **FRAMES sampling replicate** additionally fixes only the stochastic
  sampling seed. It is nested within one paired initialization instance and
  is not a new initialization or channel.

For every method in a paired initialization instance:

- generate and hash the data, active row indices, observations, truth factor,
  and initial factor once, then pass the identical arrays to each method;
- use rank one for the main unitary-channel study and the same serialized
  `U0`; do not regenerate an allegedly equivalent initialization inside each
  runner;
- use the same floating-point precision and metric implementation; and
- evaluate checkpoints from the saved factor with common, full-precision
  metric code.

FRAMES optimizes over the operator-norm ball `||U||_2 <= tau`.
Printed-adaFGD and Armijo FGD are unconstrained and therefore have no `tau`.
Predeclare one `tau` per qubit-count/regime for all of its FRAMES arms, ensure
that both `U0` and the rank-one truth lie in its ball, and archive it. Report
this domain difference prominently; do not claim that the methods have an
identical `tau` constraint and do not add an unreported projection to either
FGD method.

## Budgets and checkpoints

Use **logical (nominal) data-row epochs**, not iteration count, as the common
progress axis:

```text
logical row epochs = cumulative charged active-row measurements / m.
```

Predeclare one epoch horizon and one checkpoint grid, and emit every method's
metrics at the same logical row-epoch checkpoints. Charge sampled active rows
for FRAMES, `m` active rows for each printed update, and `m` for the gradient
plus `m` for every trial-objective evaluation in Armijo. These counters define
the comparison budget; they are not measurements of implementation work.

In particular, a structured subset kernel may traverse or construct values
for the full design before selecting its active rows. Record full-design
passes, subset-specific passes, operator calls, and wall time separately.
Never call logical row accesses “actual rows processed” or infer computational
cost from row epochs. Metric-only checkpoint evaluations are not optimizer
work; record their logical charges, implementation work, and time separately
instead of hiding them or charging them to only one method.

Equal row epochs do not imply equal computation. At each checkpoint report:

- logical active-row charges and implementation-level full/subset passes;
- optimizer-only and end-to-end wall time on the same hardware and software
  stack;
- iterations, stochastic-oracle calls, batch size, and LMO calls for FRAMES;
- spectral-norm backend, tolerance, wall time, matrix-vector/operator calls,
  and adjoint-column-equivalent work for printed-adaFGD; and
- objective evaluations and backtracks for the Armijo control.

The current paired runner executes printed-adaFGD before FRAMES in a fixed
order. Its per-method optimizer times are descriptive, not a controlled
head-to-head benchmark. For timing claims, use fresh processes with balanced
or randomized method order and identical affinity, thread, warm-up, hardware,
and software settings; otherwise label the timing comparison descriptive.

Include fixed-budget endpoints and, only when every gate component was
evaluated at each checkpoint, time/logical-row-access to the first gate pass.
Count every numerical exception as a failure and preserve its status, reason,
stderr, and diagnostic log. The current FRAMES exception path may retain only
the shared initial state rather than the last finite iterate; do not describe
that state as a last-finite checkpoint. Atomically retaining a true last-finite
factor is follow-up runner hardening, not a property of the present archive.

## Data regimes and recovery gate

The primary benchmark is noiseless, so a successful checkpoint must satisfy
all three existing conditions:

```text
process fidelity                 >= 0.99
measurement RMSE                 <= 1e-6
||Tr_out(U U^H) - I||_F / sqrt(d) <= 1e-3
```

Always report the three gate components and aggregate decision using common
metric code. Also report active-row loss/RMSE, factor norms, and
`||U||_2 / tau` for FRAMES. Report method-native diagnostics separately:
the fixed `alpha_Q`, `lambda_Q`, and printed objective for adaFGD; and the
`alpha_F(t)`/`beta_t` trajectory and time-varying smoothed objective for
FRAMES. Raw penalized objectives are not comparable across methods and must
not determine the head-to-head ranking.

Run the regimes in this order:

1. **n=2, full design:** all 128 rows.
2. **n=2, published-scale subset:** one fixed 96-of-128 training subset per
   data/channel instance, shared across methods.
3. **n=4, full design:** all 8,192 rows, only after the n=2 pipeline and
   accounting checks pass.

For every 96-row run, compute and archive both the 96-row training RMSE and a
final **full-design RMSE over all 128 noiseless rows**. Use the full-design
RMSE in the recovery gate and main table so an underdetermined fit is not
mistaken for channel recovery. Fidelity and TP metrics remain mandatory.

Until the archive retains every checkpoint factor (or evaluates every
checkpoint on all 128 rows), the subset full-design gate is **endpoint-only**.
An active-96-row checkpoint pass may be labeled an active-row fit, but never
recovery or first passage to the recovery gate. Report subset time-to-recovery
only after full-design RMSE is available at each checkpoint used to establish
that time.

## Calibration and held-out evaluation

Use designated calibration channel seeds and calibrate the methods
independently:

- for printed-adaFGD, search `alpha_Q` and `eta_scale` while keeping the
  selected TP coefficient fixed within every trajectory;
- for FRAMES, search `alpha_F(0)`, the increasing penalty schedule when it is
  not fixed in advance, batch/schedule controls, and `tau`.

Predeclare the candidate grids, comparable per-method tuning budgets, shared
evaluation horizon, and a gate-first selection rule based only on common
recovery metrics. Give both methods the same calibration data and disclose
every candidate, numerical failure, and tuning cost. Freeze both complete
selected configurations and the shared held-out budget before running unseen
channel seeds. Boundary refinement is exploratory until it is evaluated on
fresh seeds that played no role in choosing the expanded range.

“Best” means selected only from calibration and any predeclared validation
stage, never from held-out outcomes. Every validation and held-out record must
store the calibration/selection manifest SHA-256, the selected printed-adaFGD
and FRAMES configuration IDs. The immutable referenced manifest must store
the candidate-grid and selection-rule IDs and all candidate outcomes.
Held-out records may aggregate only when the recorded frozen selection identities
match; cohorts tied to different selection manifests must never be combined.

Keep seed roles distinct in the archive: channel/truth, observation, row
subset, initialization, and FRAMES sampling. A held-out channel seed exposed
during earlier mismatched-penalty work is not unseen for the new primary
analysis.

For the paper-quality phase, run multiple nested FRAMES sampling replicates
for each paired initialization instance. Sampling replicates measure
stochastic optimizer variation; they are not independent initializations or
channels. Aggregate in this order: summarize sampling replicates within each
paired initialization instance; summarize initialization instances within
each data/channel instance; then aggregate across independent held-out
data/channel instances. The effective scientific sample size is the number of
independent data/channel instances, not the number of initialization or
FRAMES RNG streams. Run each deterministic FGD method once per paired
initialization instance.

Current direct sidecars bundle printed-adaFGD with each FRAMES sampling run,
so changing only the FRAMES seed redundantly reruns the deterministic method.
The aggregator must verify and collapse identical deterministic endpoints;
these copies are not extra scientific replicates. Before a large campaign,
the production launcher should run that trajectory once and reuse it. This is
an efficiency and reproducibility limitation, not additional evidence.

## Runner invocation

After committing the runner, replace `FULL_COMMIT`, `PRINTED_ALPHA`,
`FRAMES_ALPHA0`, and `PRINTED_ETA` with the pinned commit and predeclared
calibration-candidate values, and replace `UNIQUE` with a fresh run identifier.
From the repository root, this is a safe n=2 full-design smoke invocation:

```bash
python -m paper.experiments.qpt_quiroga_direct_comparison \
  --n-qubits 2 \
  --row-epochs 2 \
  --checkpoint-epochs 1 \
  --printed-tp-weight-per-row PRINTED_ALPHA \
  --frames-initial-tp-weight-per-row FRAMES_ALPHA0 \
  --frames-penalty-exponent 0.25 \
  --eta-scale PRINTED_ETA \
  --rank 1 \
  --tau 3.0 \
  --subset-size 0 \
  --subset-seed 0 \
  --channel-seed 0 \
  --initialization-seed 0 \
  --sampling-seed 0 \
  --observation-mode noiseless \
  --noise-seed 0 \
  --shot-seed 0 \
  --batch-size 32 \
  --spectral-norm-method dense \
  --cohort-id n2-full-best-for-best-smoke-v1 \
  --trial-id n2-full-best-for-best-c0-i0-s0-smoke-UNIQUE \
  --split calibration \
  --expected-commit FULL_COMMIT \
  --require-clean \
  --save /ABSOLUTE/OUTPUT/OUTSIDE/CHECKOUT/quiroga_direct_n2_full_smoke_UNIQUE.npz \
  --quiet \
  --fail-on-method-failure
```

The two logical row epochs and one-epoch checkpoint cadence are only a
plumbing smoke test, not a production, calibration, or paper-result budget.
`PRINTED_ALPHA` and `FRAMES_ALPHA0` are independent method-specific values;
the command does not impose or imply an objective match.
The runner writes the requested NPZ and an inferred same-stem JSON sidecar,
and refuses to run if either path already exists. Use a new path and trial ID
for every invocation; do not overwrite or repurpose a prior pair. Replace the
absolute output placeholder with a directory outside the Git checkout. This is
required when using `--require-clean`: writing a sidecar inside the checkout
would make subsequent tasks fail their clean-worktree preflight.

## Staging and provenance

Before production:

1. run the focused unit tests and a tiny CPU comparison locally;
2. verify printed fixed-penalty normalization and the FRAMES increasing-penalty
   sequence, paired hashes, selection-manifest/config-ID propagation, logical
   row accounting, aligned checkpoints, endpoint full-design RMSE for subsets,
   failure-status/log preservation, and deterministic replay;
3. run one short Ruche smoke job through Slurm using the production
   environment and resource shape; and
4. inspect its status, stderr, archive, and resource use before submitting a
   bounded production array.

Do not run framework tests or experiments on a Ruche login node. Record the
Git commit and dirty-state diff, complete command/configuration, source-file
hashes, host and hardware, Slurm job/task IDs, requested resources, module and
package versions, thread settings, start/end times, every seed, data/row/
observation/initialization hashes, spectral-solver settings, and output path.
For best-for-best records, also archive the selection-manifest SHA-256, both
selected configuration IDs, `alpha_Q`, `lambda_Q`, `alpha_F(0)`, `p`, and the
realized FRAMES penalty/smoothing coefficients at every checkpoint.
Use unique immutable run directories and a manifest; preserve failed and
superseded runs with their status instead of overwriting them.

The current sidecar aggregation is metadata-only. Before paper use, run a
separate JSON/NPZ/checksum validator that cross-checks arrays and metadata,
expected keys, shapes, dtypes and finiteness, and every archive checksum.

The primary paper claim is based only on paired held-out results from the two
independently tuned, frozen best-for-best configurations under one selection
manifest. The matched-penalty ablation and Armijo recovery control must be
shown separately and described according to the different questions they
answer.
