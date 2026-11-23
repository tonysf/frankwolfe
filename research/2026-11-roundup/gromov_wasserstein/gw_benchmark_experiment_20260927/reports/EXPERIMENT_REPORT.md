# GW benchmark experiment

**Scientific interpretation corrected on 2 October 2026:** the completed momentum-KL comparisons are exploratory tests against a custom comparator. They do not supply the requested comparison against a published stochastic algorithm under its stated theory. Their measurements are retained; the primary-baseline designation is withdrawn. See [the correction](COMPARISON_SCOPE_CORRECTION.md). Replacement GPU results are not yet available.

The [full SHARE-seq experiment](SHARESEQ_FULL_CAMPAIGN.md) is complete: 31,890 paired cells on four A100s, all six final runs valid. At 300 solver seconds, FW completes 3.90 times as many outer updates, while KL has 0.2974% lower median feasible GW loss and wins all three paired seeds. This uses transferred SNARE settings without SHARE-seq tuning.

The [completed SNARE-seq and BMMC follow-up](CROSS_DATASET_CAMPAIGN.md) contains 432 valid tuning outputs, 60 valid finals and six validated component profiles. At 300 solver seconds, independently tuned stochastic KL with momentum has lower feasible GW loss in all 30 matched seed pairs. Exact-cell matching on SNARE remains near or below chance for both methods. FAUST is deferred.

The [completed independently tuned BMMC scaling campaign](BMMC_SCALING_CAMPAIGN.md) contains 288 tuning runs and 40 valid final runs at 3,000/6,224 cells per view. At 300 solver seconds, stochastic KL with momentum beats smoothed row FW with momentum in all 20 matched seed pairs; FW leads at the early 30-second checkpoints. These larger results supersede any extrapolation of the original 1,000-cell ranking below. Independent KL tuning and changed randomized-run budgets prevent attributing the reversal to size alone.

**The historical exploratory comparison used two stochastic methods with the same gradient estimator and momentum rule: Smoothed row FW + momentum (sketch gradient) and Transport KL-proximal gradient + momentum (sketch gradient).** In the original small study, the KL method returns lower median rounded GW loss on both synthetic families; smoothed row FW returns lower median loss on the fixed BMMC dataset. This compares prescribed solver-time budgets, not a fixed gradient-sample budget, and does not establish a general ranking.

Start with the [stochastic comparison and figures](stochastic_primary/report.md), [matching audit](STOCHASTIC_MATCH_AUDIT.md), and [method definitions](METHOD_NAMES.md). Deterministic methods are separate diagnostic controls and contextual benchmarks. They are outside the exploratory stochastic comparison discussed here. All original results remain in the [complete archive report](stage_c_final_named/report.md).

## What is matched

The historical exploratory comparison contains **110 corresponding instance/seed/start cells and 220 valid final outputs**. Both methods use the same q=128 Rademacher sketch gradient, iteration-indexed momentum coefficient rho(k)=0.3/(k+1)^(1/2), first-gradient momentum initialization, and reuse of that first gradient for the first update. Each pair shares the problem instance, initial coupling, initialization and stochastic seeds, precision policy, A100 model/application libraries/CUDA runtime, prescribed solver-time budget and output-rounding procedure. Twenty-five pairs differ in Ruche OS/driver revisions and 64 ran on different hosts; the complete machine environment is not identical in every pair.

The algorithms differ in constraint treatment. Smoothed row FW keeps row marginals feasible, penalizes the current column residual, and uses a row-wise linear minimization step. Transport KL-proximal gradient enforces both marginals through an approximate KL-proximal solve. Their step constants and constraint parameters consequently differ.

Two limitations matter:

- **Equal solver time does not give equal oracle work.** KL inner solves take time, so the methods perform different numbers of gradient calls and Rademacher probes. Their iteration-indexed momentum coefficients also differ at a given wall time. Figures show actual recorded probe counts and elapsed solver-plus-output-repair time. Sparse time checkpoints do not establish performance at a common probe budget.
- **Tuning is not symmetric.** Momentum-KL inherits the selected plain-KL step settings and the smoothing method's momentum coefficient, without an independent tuning grid. These results compare the recorded configurations, not independently best-tuned versions or an isolated change to one update rule.

This pair was included before held-out execution. Its former primary designation is withdrawn by the correction above; the recorded runs and settings remain unchanged. A strict comparison at equal oracle budgets would require a separately specified experiment.

## Results of the stochastic comparison

These are final float64 rounded-GW-loss medians. Lower is better. Synthetic entries give each of five independent problems equal weight after taking the median over five stochastic seeds. BMMC is one fixed dataset with five repeats; those repeats do not measure uncertainty across datasets. IQRs, actual output times and feasibility curves are in the [focused report](stochastic_primary/report.md).

Product initialization, 300-second solver budget:

| Method | Clean synthetic n=512 | Noisy synthetic n=512 | Fixed BMMC n=1,000 |
|---|---:|---:|---:|
| Smoothed row FW + momentum (sketch gradient) | 0.0437653 | 0.0400180 | 0.153029 |
| Transport KL-proximal gradient + momentum (sketch gradient) | 0.0362137 | 0.0330920 | 0.153592 |

Randomized feasible initialization, 30-second solver budget:

| Method | Clean synthetic n=512 | Noisy synthetic n=512 | Fixed BMMC n=1,000 |
|---|---:|---:|---:|
| Smoothed row FW + momentum (sketch gradient) | 0.0437761 | 0.0400349 | 0.155959 |
| Transport KL-proximal gradient + momentum (sketch gradient) | 0.0377796 | 0.0345512 | 0.161650 |

The median ordering is the same under both initialization conditions, but these tables use different durations and cannot isolate an initialization effect. No significance claim or uniform winner follows. All 220 outputs pass final validity checks; 103 of the 110 momentum-KL runs retain inner stopping failures, totaling 57,996 failed inner stopping checks. A valid rounded output does not certify convergence of every inner solve.

Smoothed row FW has raw column-L1 residuals of about 0.10 at 30 seconds and 0.022 at 300 seconds on synthetic data, while rounded loss changes little. The L2 residual is much smaller (about 0.0012 at the clean endpoint); that does not make repair negligible. GW loss at an unrounded coupling is not treated as feasible objective quality. Each inexact method receives the same copy-rounding opportunity, without a polishing phase.

All reported objective values use the **full, unsampled GW objective**. Sampling concerns the optimizer's gradient estimator. “Exact gradient” and the earlier label “full gradient” both mean the full analytical gradient evaluated in floating point. The [method definitions](METHOD_NAMES.md) specify the oracle, momentum, constraint set and update for every recorded method.

## Secondary stochastic comparisons and separate controls

Plain Transport KL-proximal gradient (sketch gradient) provides a momentum ablation: the momentum variant's endpoint medians are slightly worse in all six family/start groups, although some earlier checkpoints favor momentum. Smoothed row projection + momentum (sketch gradient) uses separately tuned constants, including a different momentum prefactor, so it does not isolate the row update. Its favorable synthetic median hides a split outcome distribution: clean runs have 17 low-loss and eight high-loss endpoints, and noisy runs have 12 and 13. Native Sampled GW (POT) uses its own conditional estimator, normalization and sampling convention; its sample counts are not interchangeable with Rademacher probe counts. Five native numerical failures remain visible in the complete results.

Exact-gradient variants answer separate diagnostic questions: whether sketch noise affects the smoothed method, how initialization interacts with the synthetic construction, and what loss other solvers reach at these sizes. Results remain in the [complete archive report](stage_c_final_named/report.md), [noiseless-control analysis](NOISELESS_CONTROL.md), and [momentum diagnostic report](momentum_replay_named/report.md). The exact-gradient smoothed FW variant still uses momentum; the exact-gradient transport FW and KL variants do not. None is presented here as an oracle-matched stochastic competitor.

Clean synthetic f*=0 is known; other values are not certified optima. The [21 per-instance feasible references](stage_c_final/feasible_references/feasible_references.md) identify saved plans, residuals and observation times. References spanning different budgets are descriptive rather than matched-budget rankings.

## Completion of the original study

The original Stage A–C study contains **1,164 completed development trials and 723 terminal final trials: 718 valid returned outputs and five preserved native failures**. These totals include all methods and controls, beyond the 220 exploratory stochastic outputs. All 796 supplementary momentum diagnostics and their cross-archive checks are complete. Larger Stage D instances remain unperformed under the original protocol's sampling-benefit gate.

The experiment folder is `/home/tony/gw_benchmark_experiment_20260927`; its Ruche counterpart is `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927`. [Every final run](stage_c_final/runs.csv), [archive verification](FINAL_ARCHIVE_COLLECTION.md), [original figure review](FINAL_FIGURE_QA.md), and the [protocol coverage record](PROTOCOL_COVERAGE.md) preserve the supporting evidence.

## Failures, observation times and precision

All 723 final trials are terminal and their provenance, frozen selection and initialization checks pass. **718 actual saved outputs pass the final validity checks**, with maximum summed rounded marginal-L1 of 1.96e-7. There are 271 completions with custom inner stopping failures, 50 completions with native warnings, and five native failures. Across all final runs, 130,536 known inner stopping failures and 30,482 warnings are retained. Native inner counts remain unknown; warnings are not convergence certificates.

The [five-failure audit](FINAL_FAILURES.md) covers BMMC seeds 100–103 and noisy synthetic problem 2000/seed 103. Unchanged native conditional sampling encountered NaN probabilities. No failing final plan or float64 endpoint is invented. Earlier checkpoints remain usable at their actual times. The failing transport state was not saved, so the first invalid arithmetic operation cannot be identified uniquely. Failed-run solver clocks stop at the last completed callback and are lower bounds; runner and launcher elapsed times have different scopes.

Of 3,508 requested final checkpoint windows, **3,485 were observed and 23 are explicitly missing**: seven because of failures, two because of native early stops, and 14 because one initialization/update crossed multiple targets. The frozen logger stores the first crossed opportunity and marks other crossed targets missing. An observation labeled 1 second can therefore become available much later; one Smoothed row FW + momentum (exact gradient) run first completed an update at 24.642 solver seconds. Nothing is backdated or interpolated. The [checkpoint audit](FINAL_CHECKPOINT_COVERAGE.md) verifies every trace and explains these cases.

The practical x-axis is actual solver time plus repair cost for that current output. Separate tables retain preparation, cumulative repair, exact diagnostics, final float64 validation, serialization and total instrumented time. Native callback time overlaps some components and is not added twice. Startup and initial diagnostics can be substantial; [timing notes](FINAL_TIMING_NOTES.md) describe observed long intervals without claiming a profiled hardware cause. Final plans are promoted from their actual stored float32 values for float64 evaluation; they are not regenerated or rerounded. The independent [first-three-output CPU audit](final_first_outputs_audit.md) has a narrower scope than the complete saved-plan marginal audit.

## Theory and oracle evidence

The [manuscript mapping](THEORY_MAPPING.md) transcribes Proposition 2.11 and Corollary 2.10. For k=0,1,... the dedicated theory track uses gamma=(k+1)^(-3/4), rho=(k+1)^(-1/2), and mu=mu0*(k+1)^(-1/4), with gamma/rho prefactors one, m*mu0=1 and q=128. Momentum contains only the smooth gradient; the current exact positive column penalty is added afterward. Momentum starts with the first sampled gradient, reused for the first update; coupling starts at the product plan.

The manuscript's Algorithm 1 line 6 reverses the penalty sign. The implementation follows the derivative of the defined positive squared-distance penalty, consistent with Section 2 and the supplied protocol. For the equivalent affine matrix constraint in Section 2, manuscript beta=mu/n. This explicit mapping is not a certification of the draft's proofs.

The [ten theory replications](a100_theory/theory_replicates.md) each evaluate all 1,000 pre-update true smoothed gaps. The mean across full-trajectory averages is 4.61572, with run IQR [4.61748, 4.64236]. This is stochastic dispersion, not an expectation confidence interval. Terminal/sparse averages are not substituted for the theorem statistic. Original-gap and feasibility conclusions remain subsequence statements; no last-iterate rate or fitted slope is claimed. A [recorded certificate check](CERTIFICATE_CHECK.md) finds no positive gap-bound excess among 11,618 evaluated points, with finite-precision and domain-feasibility limitations explicitly retained.

The [fixed-plan A100 oracle study](a100_diagnostics/report.md) covers n=128/512, three plan classes, two estimators, batches 8/32/128 and 100 independent evaluations per setting, including FW decision error. At the product plan, exact gradients take about 0.537/0.540 ms, versus 1.141/1.165 ms for q=128 sketches. The tested sketch has substantial estimation error and no measured cost advantage at those sizes. The Python-loop anchor oracle is a correctness reference. The [precision control](precision_control.md) retains raw column-L1 around 0.107–0.111 in both float32 and float64 at n=128 after 2,000 updates; float32 roundoff alone does not explain that finite-horizon residual.

## Diagnostic completion and reproducibility

The original practical smoothing logger omitted momentum errors. A separate fixed-iteration replay covers 504 tuning trajectories, 286 finals and six pilot trajectories. Each accepted companion must reproduce original source/configuration/seeds, checkpoint scalars, counters and bitwise raw/rounded endpoints. Original outputs, timing and selection remain authoritative. The [A100 replay preflight](momentum_replay_a100_preflight.md), all six pilots and the [504-run tuning audit](momentum_replay_tuning_audit.md) passed. **All 796 companions now verify**, with 3,714 diagnostic rows and four preserved missing windows. The [completion audit](MOMENTUM_REPLAY_COMPLETION.md), [combined curves](momentum_replay_named/report.md), and [796-record original-archive check](replay_original_bindings_final.json) pass; both final diagnostic figure pairs were reviewed. Their compute is accounted separately.

At output P_K, the last-used momentum belongs to input P_(K−1). The replay reports its error at that actual input, plus a separately labeled stale-momentum error against the gradient at P_K. No unused terminal estimator d_K is created. This point offset is explicit and is not substituted for the dedicated theorem statistic. For Smoothed row FW + momentum (sketch gradient), the 300-second product-checkpoint Frobenius-error medians are about 21.85/22.13 on clean/noisy n=512 and 33.12 on BMMC, versus about 0.000204/0.000200 and 0.00246 for Smoothed row FW + momentum (exact gradient). Its moving average has tracking lag even without sampling noise. These unnormalized finite-horizon diagnostics do not establish a rate or isolate the cause of objective differences. [Dispatch and scope](MOMENTUM_REPLAY_SUBMISSION.md).

Tuning used 12 prospective candidates per tunable method, fixed development seeds/budgets, and a minimum flat-median normalized final rounded-loss rule. All 1,164 trials and 72 candidates passed output validation; warnings remained visible. [Selections](STAGE_C_SELECTION.md) were frozen before held-out optimization. Final synthetic problem seeds are 2000–2004 and stochastic seeds 100–104. BMMC preprocessing, canonical float32 arrays, independent view permutations and hashes are frozen in the [instance inventory](instance_inventory.json). No biological labels or downstream findings enter the study.

Ruche jobs use account `fwllm`, partition `gpua100`, one A100-SXM4-40GB per task, and the user-wide four-GPU limit. PyTorch is 2.5.1+cu121; AMP/TF32 are disabled, matmul precision is highest, and deterministic settings are recorded. BLAS/OpenMP/PyTorch intra-op threads are one. Practical campaign runners retain inter-op 32; dedicated smoothing/theory runners and standalone environment capture use one. Native Sampled GW is a faithfully instrumented hybrid POT implementation, with native estimator/stopping rules and CPU transfers. Its timings are not a pure-GPU algorithmic speed ratio.

Immutable original tuning/final releases are `aaeb5fa` and `af7b7c5`; replay runtime is `b585198`. Final-source A100 preflight passed 125 tests, 18 projection cases and 20 actual runner smokes. Failed preflights and their fixes remain in the [job ledger](job_ledger.json). All 7,839 final archive files (2,147,932,319 bytes) and 56 Slurm logs were verified after stable transfer. PyTorch allocator peaks include instrumentation: the maximum observed final-run allocation is about 129.3 MiB, reserved 176 MiB; these are not total device-memory measurements, and these small-instance measurements do not establish a large-instance memory advantage. Original final budgets total 29.125 nominal solver GPU-hours. [Scheduler accounting](COMPUTE_ACCOUNTING.md) records 41.1583 A100 allocation-hours for the completed original study and its validation, including failed preflights; the final array accounts for 36.0122 of those hours. Allocated time includes process and instrumentation overhead and is not measured GPU utilization. Total allocation across all 15 submitted job IDs is **55.8 A100 hours**: 41.1583 for original work and 14.6417 for diagnostic reconstruction/validation. All 113 primary task records are terminal and reconciled; 224 Slurm step rows are excluded to avoid double counting. The 796 companions separately sum to 50,104.8355 controller seconds (13.9180 hours), counted once and not added to allocation time. The original one-time data-preparation duration was not fully measured and is not reconstructed.

The Stage D gate remains closed because the measured oracle cost/error evidence does not establish a sampling benefit. Synthetic n=2048/8192 and BMMC n=3000/6224 were not run. A larger hardware crossover remains untested. This decision follows the protocol's conditional scaling rule. Stages A–C, required diagnostic supplements, artifact verification and allocation accounting are complete; the larger experiments remain explicitly unperformed.

Source and report preservation is documented in the [delivery record](FINAL_BACKUP.md); the [artifact index](completed_artifact_index.json) lists file hashes. Raw data and run archives are retained separately from the review package.
