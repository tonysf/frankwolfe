# Final report closeout — completed

Closed on 2026-09-28. All 723 original final trials are terminal (718 valid outputs, five preserved native failures); all five archives and 22 practical figure pairs are verified. All 796 required diagnostic companions verify, both diagnostic figure pairs were reviewed, and every original-archive binding matches. Allocation accounting covers every submitted job. Stage D is deferred under the protocol gate. See [main report](EXPERIMENT_REPORT.md), [diagnostic completion](MOMENTUM_REPLAY_COMPLETION.md), and [compute accounting](COMPUTE_ACCOUNTING.md).

The checklist below is the preserved **historical pre-closeout review**. Its pending wording and prospective commands describe that earlier state; use fresh output directories to regenerate reports, since the completed report directories already exist.

---

# Final report readiness — pending

Reviewed 2026-09-28 against Protocol §§10–15, the frozen campaign reporter, replay reporter, feasible-reference helper, current experiment report and coverage snapshot. This is a source/artifact review, not a new numerical experiment or a completion certificate. The selected original array **1961580** and replay arrays **1961715/1961716** remain pending closeout in this review. The six pilot replay attachments already passed the separate A100 preflight and cross-archive pilot check. No runtime, selection, manifest or archived scientific output was changed.

The required report machinery is present. The remaining work is complete collection and provenance validation, report generation, visual review, cross-linking and evidence-based interpretation. No additional optimizer experiment is identified as necessary to generate the required Stage C report. Stage D remains provisionally closed; it requires a separate decision after the complete Stage C evidence is assessed.

| Requirement | Implemented output | Final readiness / interpretation |
|---|---|---|
| Current feasible rounded loss versus time, §§11,14 | `summarize_campaign.py` writes PNG/PDF `panel_*_rounded_objective` using actual solver time plus the repair for that output. | Pending final generation. Current recorded outputs only; no best-iterate selection. Horizontal IQR uses actual observation times, not nominal target times. Float32 checkpoint evaluations are distinct from final float64 endpoint tables. |
| Raw marginal L1 and L2 versus solver time, §§10,14 | Separate `panel_*_raw_feasibility` plots; final tables contain raw row L1, column L1/L2, rounded both-L1 and repair L1. | Pending final generation. Per-run/point JSON retains mass error, minimum entry and remaining residuals. Raw loss is explicitly not feasible quality. |
| True smoothed gap versus iteration and oracle work, §§11,13B,14 | Completed [iteration](a100_diagnostics/true_smoothed_gap_vs_iteration.pdf), [probe work](a100_diagnostics/true_smoothed_gap_vs_probes.pdf), [full-trajectory mean](a100_diagnostics/full_trajectory_mean_gap.pdf), and [ten-repeat theory report](a100_theory/theory_replicates.md). The campaign report links this evidence. | Already covered by the dedicated track. Campaign traces retain sparse true gaps, mu, iteration and work counters, but the frozen practical reporter does not create a new practical true-gap panel. This is not a missing required figure: preserve the dedicated links. A sparse-checkpoint mean is never substituted for the theorem statistic. Exact-gradient work is not equated with stochastic probes. |
| Original feasible FW gap versus time, §§10,14 | `panel_*_original_gap` for n≤512, including small n=128 PG panels, at actual solver-plus-current-repair time. | Pending final generation. Uses true-gradient assignment gaps at rounded outputs. No original-gap panel is promised for BMMC n=1000. Signed roundoff remains visible using the symmetric-log gap axis. |
| Momentum error and point alignment, §10 | [Separate replay reporter](../summarize_momentum_replays.py), 796 prescribed records: 504 tuning, 286 finals, six pilot. Original KL-momentum supplement traces already carry their own indexed error and are not replay duplicates. | Pending complete replay collection/verification. For output K≥1 the primary error is at actual input P_(K−1); stale-current tracking compares the same stored momentum to grad f(P_K). No d_K existed at terminal output. **Literal limitation remains:** the primary error and current-output loss/gap do not share the displayed iterate. The replay repairs the missing log with explicit indices, not a fabricated current-point estimator. |
| Timing, peak memory and failures, §§10–11,14 | Final endpoint/resource tables cover preparation, solver, cumulative repairs, exact diagnostics, final float64 validation, serialization, total instrumented time, peak allocated/reserved GPU memory, CPU RSS, warnings, underflow and known/unknown inner failures. Per-run counters/configurations remain in JSON/CSV. | Pending final generation. Native callback time overlaps repair/diagnostic components and must not be added again. Peak memory includes instrumentation. Native inner counts remain unknown, not zero. Failed-run partial solver clocks need explicit lower-bound caveats in the final narrative. |
| Replications, matched starts, uncertainty, §12 | Campaign report validates frozen selection lineage and initialization hashes; medians/IQR use unique stochastic/initialization units, then one within-instance median per synthetic problem. | Five independent synthetic problems per family; fixed BMMC has repeat dispersion, not independent-dataset uncertainty. Exact product starts contribute once per instance, not five artificial repeats. Randomized-panel variation includes paired stochastic and initialization changes. |
| Full coverage and missing windows, §§11–12,15 | Manifest-driven 723 original final cells, with status, failure penalties, missing checkpoint targets, per-window prescribed/observed units and open markers. Replay JSON and `curve_summaries.csv` additionally expose prescribed/observed run and instance counts and `coverage_complete`, including empty targets. | A legitimate failed trial can count as execution-complete without a valid final output. A failed/unverified replay remains terminal but contributes no diagnostic curve. Neither an early native stop nor a failed final is extended to a later checkpoint. Require all five original final manifests and all three replay manifests. |
| Feasible reference values, §14 | [Reference helper](../summarize_feasible_references.py) writes per-instance Markdown/JSON/CSV from the complete provenance-valid campaign report. | Pending all 723 terminal records and report validation. Clean synthetic f*=0 is retained separately from signed best observed float64 values. Noisy/BMMC f_ref is descriptive, not certified optimal. Mixed budgets and source run/time/residuals remain explicit. Absolute-loss figures are valid; a new normalized-gap plot is not required. |
| Reproducibility, §15 | Frozen source/input/manifest/receipt checks, saved-plan marginal validation, selection provenance, original archives and companion proof hashes. [Cross-archive audit](../audit_replay_original_bindings.py) binds all 796 replay records back to separately collected originals. | Pending complete archives. The campaign reporter independently checks saved-plan feasibility; final loss values come from the runner's float64 evaluator. Do not call this a fresh independent recomputation of every objective. The independent first-three-output audit has a narrower, clearly documented scope. |

The Protocol §10 inequality check is already recorded as `certificate_rhs`/`certificate_excess` by the dedicated smoothing runner. Its maximum violation could be summarized from existing Stage B traces if useful, but this is an optional report-only consistency table, not a reason to rerun optimizers or claim a theorem proof.

## Commands after complete collection

Run locally from `/home/tony/gw_benchmark_experiment_20260927`, after stable transfer checksums, complete shard ledgers and scheduler accounting have been preserved. The commands below are **not executed by this review**. Limit local numerical-library threads to one. The final archive destinations below are the collector-confirmed names matching the frozen remote root basenames; they do not assert these still-running archives have already been collected. Retain each archive's relative run layout and `_campaign`/`_replay` ledgers. Do not include early-preview copies as duplicate roots.

Generate the combined five-phase practical report; the 30-second product checkpoint comes from the same 300-second trajectory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python summarize_campaign.py \
  --manifest configs/stage_c_final_manifests/final_product.json \
  --manifest configs/stage_c_final_manifests/final_randomized.json \
  --manifest configs/stage_c_final_manifests/small_pg.json \
  --manifest configs/momentum_final_manifests/momentum_product.json \
  --manifest configs/momentum_final_manifests/momentum_randomized.json \
  --root ruche_results/final_product_af7b7c5 \
  --root ruche_results/final_randomized_af7b7c5 \
  --root ruche_results/small_pg_af7b7c5 \
  --root ruche_results/momentum_product_af7b7c5 \
  --root ruche_results/momentum_randomized_af7b7c5 \
  --selection configs/stage_c_selection.json \
  --out reports/stage_c_final --require-complete
```

This reporter searches all supplied roots for each run; these arguments are not pairwise manifest/root mappings. Its completion flag means terminal execution plus valid provenance and no manifest omissions; numerical failures remain in the report. Check the required phase counts **253+330+30+55+55=723**, the selection-provenance and initialization audits, and all failure rows. Never invoke `--write-selection` on final evidence. The machine-readable `full_study_complete` remains false by design: closeout must state the actual completed scope and Stage D decision explicitly.

Then generate descriptive feasible references into a fresh directory:

```bash
.venv/bin/python summarize_feasible_references.py \
  --report reports/stage_c_final/report.json \
  --out reports/stage_c_final/feasible_references
```

Generate the separate momentum report. Unlike the campaign reporter, its repeated manifest/root arguments are paired in order. Pilot attachments are reused from the verified preflight, not repeated as new scientific runs. Use a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python summarize_momentum_replays.py \
  --manifest configs/momentum_diagnostic_replays/tuning.json \
  --root ruche_results/momentum_diagnostic_replay_tuning \
  --manifest configs/momentum_diagnostic_replays/pilot.json \
  --root ruche_results/momentum_diagnostic_replay_pilot \
  --manifest configs/momentum_diagnostic_replays/finals.json \
  --root ruche_results/momentum_diagnostic_replay_finals \
  --out reports/momentum_replay_completion --require-complete
```

Its completion flag requires all **796** terminal records and valid provenance; `complete_with_unverified_replays` is distinct from `all_diagnostics_verified`. Original clocks label the scientific plots. Replay `controller_elapsed_seconds` is summed once; nested worker times are not added twice. Utility hashes resolve the frozen controller revision, so later documentation commits do not invalidate an otherwise correct archive.

Finally cross-bind the complete replay report to the separately collected original archives and canonical input files. This hashes bytes/metadata only; it does not load arrays, rerun solvers or modify originals. The pilot root is the historical dataset/method directory, without a `pilot/` prefix:

```bash
.venv/bin/python audit_replay_original_bindings.py \
  --report reports/momentum_replay_completion/report.json \
  --original-root tuning=ruche_results/tuning_aaeb5fa \
  --original-root pilot=ruche_results/1961459 \
  --original-root final_product=ruche_results/final_product_af7b7c5 \
  --original-root final_randomized=ruche_results/final_randomized_af7b7c5 \
  --data-root data \
  --out reports/replay_original_bindings_final.json
```

Require `all_match=true`, coverage 796 and unchanged-input evidence. Six genuine pilot originals already passed the helper-level integration; that is not a substitute for this final full audit.

## Remaining report-only closeout

1. Preserve transfer-before/after/local checksums, final scheduler accounting and all legitimate numerical failures before publishing summaries. Inspect 723 original and 796 companion prescribed records separately; do not merge them into a larger scientific sample size.
2. Generate the reports above and visually inspect every final PNG/PDF for method legends, actual-time axes, IQR readability, signed near-zero values, open incomplete-population markers and missing native windows. A completed array is not sufficient evidence that every output is valid.
3. Update [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md), [PROTOCOL_COVERAGE.md](PROTOCOL_COVERAGE.md), the artifact index and job ledger with verified panel counts, selected constants, failure coverage, feasible references, replay report and cross-archive audit links. Their current pending-final and prospective-replay wording is intentionally stale until closeout evidence arrives; it must then be rewritten without implying Stage D ran.
4. State the replay point-offset limitation and any unverified-original count next to the momentum findings. Preserve the actual failed native run and earlier observed checkpoints. Distinguish solver-plus-repair quality, total instrumented process time, startup/scheduler time and additional replay compute. The original one-time data-preparation duration was not fully measured and cannot be reconstructed.
5. Close resource accounting with one bounded, final `sacct` query for the experiment job IDs in the reconciled job ledger and submission receipts. Archive primary allocation rows with at least `JobIDRaw,JobName,State,ExitCode,Start,End,ElapsedRaw,AllocTRES,Partition`; include all allocated failed preflights and retries. Sum A100 allocation seconds as primary allocation elapsed seconds times its allocated A100 count. Count each array task allocation once; do not also count an aggregate parent, `.batch`, `.extern` or other step rows. Cancelled jobs that never started contribute zero. An `sbatch` rejection without a job ID creates no allocation. Scheduler waiting is not GPU time, CPU preparation jobs and local CPU controls are not A100 allocations, and missing accounting must be labeled unavailable rather than inferred. Report actual A100 allocation seconds separately from original nominal solver budgets and summed replay controller elapsed; none is a substitute for the others. No new accounting helper or numerical run is required.
6. Reassess Stage D using the completed feasible-output comparisons, same-device oracle cost/error evidence and measured memory. No automatic threshold or winner is implemented. Current small-instance evidence does not establish a sampling advantage; do not infer a larger-size crossover or biological result. A justified negative/deferred scaling decision can close this stage while explicitly leaving larger experiments unperformed.

No new mandatory plot implementation or numerical run was identified by this bounded review. The optional practical sparse-gap panels, a compact certificate-inequality table, or a KL-momentum error overlay would be supplementary presentation only and must not delay or replace the required complete reports above.
