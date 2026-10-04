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
  same penalized objective. This establishes whether an instance is
  recoverable by ordinary FGD and helps distinguish a failure of the printed
  adaptive step from a failure of the sensing problem. It is not a third
  primary competitor and must not be reported as “Quiroga's method.”
- MiFGD, Nesterov variants, momentum SGD, and the local-Pauli runs are outside
  this direct comparison.

## Primary shared-objective lane

For active training rows `S`, let `m = |S|`, let
`r_S(U) = A_S(U U^H) - y_S`, and define

```text
H(U)       = ||Tr_out(U U^H) - I||_F^2
L_alpha(U) = (1 / (2m)) ||r_S(U)||_2^2 + alpha H(U).
```

Every paired primary comparison must use one common, predeclared `alpha`.
The two implementations are parameterized as follows:

```text
printed-adaFGD: lambda = m * alpha
FRAMES:         beta   = 1 / (2 * alpha), held constant (no smoothing decay)
```

Thus the summed objective used by printed-adaFGD is exactly
`m * L_alpha`; FRAMES's mean-loss smoothed objective is exactly `L_alpha`.
As scalar functions on the common factor space, their values and gradients
differ only by the positive factor `m`; the FRAMES-only domain restriction is
documented below. Archive `m`, `alpha`, `lambda`, and `beta`, and validate
these identities in the runner. A fixed numerical `lambda` must never be
reused when `m` changes; it would change `alpha`.

The primary result is either the complete paired `alpha` grid or a single
`alpha` selected by a method-independent rule on calibration instances.
Choosing one penalty for printed-adaFGD and another for FRAMES is not a
primary shared-objective comparison.

## Secondary method-tuned lane

A separate best-attempt lane may tune `alpha` as well as algorithm-specific
parameters independently for each method. Label it **method-tuned,
different-objective** whenever the selected `alpha` values differ. It answers
how well each method can be made to work, not which optimizer is better on
the same objective.

In particular, earlier held-out runs that used `beta=100` for printed-adaFGD
and `beta=1000` for FRAMES have mismatched penalties. Preserve them as
exploratory evidence, but do not reuse them in the primary comparison or
count their held-out seeds as unseen primary test instances.

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

Always report the three components as well as the aggregate gate. Also report
training measurement loss, `L_alpha`, factor norm, and constraint activity
`||U||_2 / tau` for FRAMES.

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

Use designated calibration channel seeds to set the shared `alpha` grid,
adaFGD `eta_scale`, FRAMES batch/schedule controls, `tau`, epoch horizon, and
all selection rules. Freeze the complete protocol before running unseen
held-out channel seeds. Boundary refinement is exploratory until it is
evaluated on fresh seeds that played no role in choosing the expanded range.

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

After committing the runner, replace `FULL_COMMIT` with its full commit hash
and `UNIQUE` with a fresh run identifier. From the repository root, this is
a safe n=2 full-design smoke invocation:

```bash
python -m paper.experiments.qpt_quiroga_direct_comparison \
  --n-qubits 2 \
  --row-epochs 2 \
  --checkpoint-epochs 1 \
  --tp-weight-per-row 0.005 \
  --eta-scale 0.003 \
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
  --cohort-id n2-full-smoke-v1 \
  --trial-id n2-full-c0-i0-s0-smoke-UNIQUE \
  --split calibration \
  --expected-commit FULL_COMMIT \
  --require-clean \
  --save /ABSOLUTE/OUTPUT/OUTSIDE/CHECKOUT/quiroga_direct_n2_full_smoke_UNIQUE.npz \
  --quiet \
  --fail-on-method-failure
```

The two logical row epochs and one-epoch checkpoint cadence are only a
plumbing smoke test, not a production, calibration, or paper-result budget.
The runner writes the requested NPZ and an inferred same-stem JSON sidecar,
and refuses to run if either path already exists. Use a new path and trial ID
for every invocation; do not overwrite or repurpose a prior pair. Replace the
absolute output placeholder with a directory outside the Git checkout. This is
required when using `--require-clean`: writing a sidecar inside the checkout
would make subsequent tasks fail their clean-worktree preflight.

## Staging and provenance

Before production:

1. run the focused unit tests and a tiny CPU comparison locally;
2. verify objective equality, paired hashes, logical-row accounting and
   implementation-work reporting, aligned checkpoints, endpoint full-design
   RMSE for subsets, failure-status/log preservation, and deterministic replay;
3. run one short Ruche smoke job through Slurm using the production
   environment and resource shape; and
4. inspect its status, stderr, archive, and resource use before submitting a
   bounded production array.

Do not run framework tests or experiments on a Ruche login node. Record the
Git commit and dirty-state diff, complete command/configuration, source-file
hashes, host and hardware, Slurm job/task IDs, requested resources, module and
package versions, thread settings, start/end times, every seed, data/row/
observation/initialization hashes, spectral-solver settings, and output path.
Use unique immutable run directories and a manifest; preserve failed and
superseded runs with their status instead of overwriting them.

The current sidecar aggregation is metadata-only. Before paper use, run a
separate JSON/NPZ/checksum validator that cross-checks arrays and metadata,
expected keys, shapes, dtypes and finiteness, and every archive checksum.

The primary paper claim is based only on paired held-out results from the
shared-objective lane. The method-tuned lane and Armijo recovery control must
be shown separately and described according to the questions they answer.
