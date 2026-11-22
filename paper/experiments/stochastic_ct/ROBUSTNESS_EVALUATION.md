# Stochastic FraMES: mechanism and robustness evaluation

This campaign expands the existing real-data CT comparison to test stochastic
benefit, tuning sensitivity, parameter transfer, initialization sensitivity and
the role of the spectral domain. It preserves earlier experiments and uses
separately frozen sources. It does not add a new reconstruction model or a new
learned reconstruction comparator.

The model remains the rank-factorized least-squares problem

\[
 f(U,V)=\frac{1}{2Nd}\sum_{i\in\mathcal I}
       \|P_i(UV^\top)-y_i\|_2^2,
 \qquad U,V\in\mathbb R^{512\times r},\quad d=956.
\]

Smoothing and proximal methods target the product of spectral balls intersected
with the nonnegative orthant. Joint L-BFGS-B targets nonnegative factors without
spectral bounds. Signed measurements and the previous preprocessing are retained.
Data-fit differences across these domains are not automatically attributed to
the domain difference. Reference-image error and held-out loss remain separate
from the optimization objective.

## Cohort and view sets

The two existing development slices are used exclusively for numerical
parameter selection. The existing evaluation cohort contains fourteen slices
from six groups. Six additional confirmation slices are reserved before their
optimization results are seen, using an outcome-independent selection rule.
They are fresh slices within the same six groups, not six new independent
specimens. Exact slice IDs and the reservation rule are in
`confirmation_reservation.json` and the frozen `cohort.json`.

The main view counts are \(N\in\{120,360,1080,2160\}\). Their nested training
sets contain the existing 120 calibration angles. The held-out 360 angles are
disjoint from every training set. Thus the standard factors, amplitude fit and
reference spectral radii are identical across view counts for a given slice.
Changing \(N\) also changes the inverse problem; the separate fixed-\(N\)
batch study identifies the effect of stochastic sampling more directly.

Five oracle seeds are used for stochastic methods on the main grid. The stress
panel uses ten; deterministic methods are run once for each distinct problem
and initialization. The six-slice diagnostic panel is selected by a fixed rule,
one slice per acquisition group, without using optimization outcomes.

## Algorithms and mechanism controls

SFRAMES retains the manuscript's schedules
\(\gamma_k=\gamma_0(k+1)^{-3/4}\),
\(\beta_k=\beta_0(k+1)^{-1/4}\) and
\(\rho_k=(k+1)^{-1/2}\), with the first sampled gradient initializing the
average. The exact orthant Moreau gradient is added after averaging.
The no-averaging control uses the same sampled gradients and schedules, but
replaces the average with the current gradient. Both controls share their
random-stream convention so a paired comparison can hold minibatch draws fixed.

Deterministic FraMES uses the exact current full gradient without averaging and
the deterministic schedules \(\gamma_k=\gamma_0(k+1)^{-2/3}\) and
\(\beta_k=\beta_0(k+1)^{-1/3}\), as in Proposition 2.12 of the manuscript.
Full-batch SFRAMES retains averaging and the stochastic schedules; these are
distinct controls. Each spectral LMO uses the established thin-SVD polar rule,
including its numerical support-gap certificate.

ProxSARAH retains the paper's constant-parameter v3 rule, with full snapshots,
relaxation 0.99, \(m=\lfloor\sqrt N\rfloor\),
\(b=\lfloor\sqrt N/C\rfloor\), \(C=2/(3\cdot0.99^2)\), and the established
curvature scaling. SPRING-SARAH retains its practical current-curvature steps,
batch 64 and refresh period 16. Joint L-BFGS-B retains the established joint
factor objective, analytic gradient, native line search and exact checkpoint
state. Earlier PALM, NNLS and multiplicative-update results remain available as
historical reference comparisons; they are not crossed with every new ablation.

## Frozen development protocol

For each smoothing method, set
\(\beta_0=c_\beta/\max(L_U^0,L_V^0)\), using the frozen reference calibration
curvatures. All exponents above are fixed. The coarse grid is

\[
 c_\beta\in\{0.25,1,4,16,64\},\qquad
 \gamma_0\in\{0.025,0.05,0.1,0.2,0.4\}.
\]

Screen all 25 settings at \(N=1080\), on both development slices, using one
oracle seed and 8,640 gradient views per direction. The six families are
SFRAMES with batches 16, 64, 256 and full; SFRAMES without averaging with batch
64; and deterministic FraMES. This stage contains **300 physical runs**.

Within each family, retain the best three candidates for the joint loss and
feasibility criterion and additionally the loss-only winner if it is distinct.
Confirm these settings at both 360 and 1,080 views on both development slices,
with 34,560 views per direction and three stochastic repetitions or one
deterministic run. There are at most **224 confirmation runs**. The loss-only
winner at the confirmation budget is selected from this confirmed shortlist,
not claimed to be the optimum over an untested full-budget grid.

The joint criterion first maximizes the fraction of runs attaining both
repaired training loss at most 30% of initialization and raw orthant distance
at most 1% of the initial factor-pair norm. Ties use lower repaired loss and
then lower raw distance. If no setting attains both targets, minimize the
median of the larger normalized target ratio, then loss and distance. Metrics
are aggregated over seeds within slice/view cases, then slices within each
view count, then view counts. Target attainment uses means at the same levels;
losses and distances use medians. Exact ties favor smaller \(c_\beta\), then
smaller \(\gamma_0\).

Retain the full response surface and attainment at feasibility tolerances
0.1%, 0.5%, 1% and 2%; the additional thresholds do not select the default.
Saved dense trajectories also support summaries at 8,640, 17,280, 34,560 and
69,120 views, wherever the run reaches the corresponding budget. Values use
the most recent recorded iterate within the budget, without interpolation.

ProxSARAH's curvature scale is selected from
\(\{2^{-5},2^{-4},\ldots,2^2\}\), on both development slices, both view counts
and three seeds at 34,560 views per direction: **96 runs**. Joint L-BFGS-B
memory is selected from \(\{5,10,20,40,80\}\) under the same slice/view/budget
conditions with one deterministic run: **20 runs**. Its remaining parameters
are the established relative KKT tolerance \(10^{-3}\), function tolerance
\(10^{-9}\) and maximum 20 line-search trials. Baselines minimize the nested
median relative evaluated training loss. SPRING's published practical rule and
previous refresh setting are frozen rather than assigned a new ad hoc step.

These are finite, declared grids. Any boundary winner is explicitly flagged;
it is not described as an interior optimum or as proof of broad robustness.
Nonfinite development candidates remain in the record and are ineligible.
Projection, operational and provenance failures halt for investigation rather
than being silently filtered as unsuccessful parameter choices.

## Evaluation studies

After development selection is frozen, evaluate:

| Study | Cases and controls |
|---|---|
| Main comparison | All fourteen existing slices; four view counts; rank 64; five stochastic seeds. Joint- and loss-selected SFRAMES, full-batch SFRAMES, deterministic FraMES, ProxSARAH, SPRING-SARAH and joint L-BFGS-B. |
| Batch size | Six-slice panel at 1,080 views; batches 16, 64, 256 and full, tuned separately on development data; deterministic FraMES and the tuned no-averaging control. |
| Averaging control | The same panel and batch 64, replacing averaging with the current gradient while retaining the joint-selected SFRAMES prefactors and oracle draws. |
| Radius | Panel, 1,080 views, multipliers 1, 2 and 4 of the standard reference factor operator norms; SFRAMES, FraMES and ProxSARAH. |
| Rank | Panel, 1,080 views, ranks 32, 64 and 128; SFRAMES, FraMES, ProxSARAH and joint L-BFGS-B. |
| Initialization | Panel, 1,080 views, standard, substantially perturbed and random nonnegative factors; those four methods, with fixed reference radii and curvature scales. |
| Stress repetitions | Panel, 2,160 views, ten oracle seeds for joint-selected SFRAMES and ProxSARAH. |
| Fresh confirmation | Six reserved slices, 360 and 1,080 views, the main methods and five stochastic seeds. |
| Per-condition retuning | Select separately at 360 and 1,080 views from the same development evidence; evaluate on the panel only where the selected parameters differ from the frozen global setting. |
| Stationarity | The predetermined Mix 1 and lava-stone panel slices at 1,080 views, SFRAMES with three seeds and FraMES once, at four times the main budget. |

Radius, rank, initialization and per-condition studies use three stochastic
repetitions. Physically identical configurations appearing in multiple studies
share one run. Each run's `memberships` preserves every study/variant label,
including coincident joint- and loss-selected SFRAMES settings. Main and stress
seed overlaps likewise do not cause duplicate execution.

An independently audited reuse catalog can satisfy a planned case from an
already completed dense trajectory only when the objective, initialization,
parameters, stochastic streams and hard-budget semantics match. Normalized
records explicitly identify `historical_reuse`, retain the original executed
source hashes and all origin-artifact digests, and do not claim that the new
engine executed the old run. Unknown equivalences are rerun; sparse historical
development traces are not used under the new dense-recording protocol. The
catalog and materialization helper are frozen before execution, and reused
case counts are reported separately from newly executed cases.

The radius and initialization studies keep the optimization domain fixed when
comparing starts: the standard reference factors define the balls, not the
new starting factors. Alternative starts are valid for the smallest tested
radius and use only calibration views. Reference curvature scales also remain
fixed across starts, so the initialization control does not silently retune
the algorithm.

## Work, diagnostics and interpretation

The main directional gradient-view cap is 69,120. Every gradient trial counts,
including rejected L-BFGS-B trials. Curvature-estimation work is recorded
separately and retained in total operator counts; no optimizer time is compared.
This continues the existing gradient-budget convention, not a claim of equal
total operator work across methods.

Record raw loss, repaired loss, held-out loss, reference-image error, raw
nonnegativity distance and work after every accepted native iteration. Record
a terminal state when charged work has not produced another accepted iterate.
Orthant clipping followed by radial spectral repair is diagnostic only.
L-BFGS-B is evaluated at its actual nonnegative factors without spectral repair.

The stationarity runs record the true full-gradient smoothed Frank-Wolfe gap
at every iterate, with the corresponding \(\beta_k\), and its cumulative
iteration mean. These diagnostics concern the smoothing algorithms. They do
not rank exactly feasible proximal methods by the smoothed gap, and measured
finite-run slopes are not claimed to verify an asymptotic theorem. Diagnostic
forward/adjoint work is separately recorded and excluded from solver budgets.

Trajectory means and descriptive standard deviations retain equal group,
within-group slice and within-slice repetition weights. Paired uncertainty
summaries resample groups rather than treating seeds as independent specimens.
Fresh within-group slices provide additional confirmation but do not increase
the number of independent acquisition groups.

## Execution and records

`robustness_campaign.py` prepares frozen sources and plans, runs the coarse
screen, confirms the shortlist, tunes the baselines, freezes global and
per-condition selections, then runs all evaluation studies. Four single-thread
workers use Elite CPUs 0, 2, 4 and 6 under persistent user-service execution.
Each run saves its factors, optimizer/estimator state, random streams, counters
and dense observations for exact continuation.

`protocol.json` pins source bytes, prepared inputs, the cohort, development
plans and publication destination. `plans/` stores the stage plans;
`stages/` stores validated stage receipts. `selection.json` records all scores,
the chosen parameters and boundary flags. `completion.json` is written only
after every planned evaluation result validates. Numerical completion and
report publication have distinct statuses, so a publication guard cannot be
mistaken for a failed or incomplete numerical campaign.

The final paper retains two figures and one table; detailed settings, response
surfaces and secondary studies belong in reproducibility materials. Raw data,
calibration arrays, environments and optimizer checkpoints remain outside Git.
