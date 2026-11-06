# Campaign results and progress

Parser/aggregation smoke test using preserved CPU outputs; not A100 campaign evidence.

Status: **complete_provided_manifests**. Expected runs: 8; attempted: 8; terminal records: 8; valid final outputs: 8.

Execution completion includes failed terminal trials; output validity and selection readiness are separate report fields. Saved raw/rounded plan marginals are independently recomputed in NumPy float64 and compared to runner validation; final objective values come from the runner float64 evaluator. Checkpoint curves retain the trace evaluations in the recorded solver precision (float32 in the A100 campaign); they are distinct from final float64 validation. Missing and failed runs carry +infinite tuning penalties. Any such run makes its whole candidate ineligible. Eligible candidate scores are the flat median of prescribed final rounded losses divided by their fixed product-plan losses. Warnings and inner stopping failures remain visible and do not alone disqualify a valid output.

| Phase | Expected | Attempted | Valid final | Failed/invalid/infrastructure | Missing/pending | Early native stop/max-iter | Missing checkpoint observations |
|---|---:|---:|---:|---:|---:|---:|---:|
| final_product | 4 | 4 | 4 | 0 | 0 | 0 | 0 |
| tuning | 4 | 4 | 4 | 0 | 0 | 0 | 0 |

Final saved-output summaries (median [Q1, Q3]). These are actual terminal outputs, not outputs moved to the nominal budget. Native early outputs retain their actual times; missing later checkpoint windows stay missing. Quality statistics exclude invalid finals, whose counts remain explicit. Timing and memory include provenance-verified terminal failures.

| Phase / data | Method | Valid / prescribed units | Budget (s) | Rounded loss, float64 | Raw loss, float64 | Actual solver (s) | Final output time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| final_product / synthetic n=16 noise=0.0 | smoothing-fw | 1/1 | 0.03 | 0.1004 [0.1004, 0.1004] | 0.10175 [0.10175, 0.10175] | 0.030253 [0.030253, 0.030253] | 0.03045 [0.03045, 0.03045] |
| final_product / synthetic n=16 noise=0.0 | full-kl | 1/1 | 0.03 | 0.10062 [0.10062, 0.10062] | 0.10062 [0.10062, 0.10062] | 0.030745 [0.030745, 0.030745] | 0.030947 [0.030947, 0.030947] |
| final_product / synthetic n=16 noise=0.0 | stochastic-kl | 1/1 | 0.03 | 0.10057 [0.10057, 0.10057] | 0.10057 [0.10057, 0.10057] | 0.030634 [0.030634, 0.030634] | 0.030834 [0.030834, 0.030834] |
| final_product / synthetic n=16 noise=0.0 | fw | 1/1 | 0.03 | 0 [0, 0] | 0 [0, 0] | 0.030189 [0.030189, 0.030189] | 0.030365 [0.030365, 0.030365] |

| Phase / data | Method | Raw row L1 | Raw column L1 | Raw column L2 | Rounded both L1 | Repair displacement L1 |
|---|---|---:|---:|---:|---:|---:|
| final_product / synthetic n=16 noise=0.0 | smoothing-fw | 5.5511e-17 [5.5511e-17, 5.5511e-17] | 0.15153 [0.15153, 0.15153] | 0.043809 [0.043809, 0.043809] | 1.6653e-16 [1.6653e-16, 1.6653e-16] | 0.15153 [0.15153, 0.15153] |
| final_product / synthetic n=16 noise=0.0 | full-kl | 2.498e-16 [2.498e-16, 2.498e-16] | 1.3878e-16 [1.3878e-16, 1.3878e-16] | 5.1926e-17 [5.1926e-17, 5.1926e-17] | 6.9389e-17 [6.9389e-17, 6.9389e-17] | 2.7452e-16 [2.7452e-16, 2.7452e-16] |
| final_product / synthetic n=16 noise=0.0 | stochastic-kl | 3.1225e-16 [3.1225e-16, 3.1225e-16] | 1.3184e-16 [1.3184e-16, 1.3184e-16] | 4.6548e-17 [4.6548e-17, 4.6548e-17] | 1.1102e-16 [1.1102e-16, 1.1102e-16] | 3.1962e-16 [3.1962e-16, 3.1962e-16] |
| final_product / synthetic n=16 noise=0.0 | fw | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |

Time components in seconds. Repair and diagnostic columns are cumulative instrumentation costs; the time-to-output axis adds only the current output repair. Native callback totals overlap these components and are not added twice. Additional Python/native preparation overhead remains in the detailed timing dictionary and total elapsed time.

| Phase / data | Method | Preparation | Solver | Repairs | Diagnostics | Final validation | Serialization | Total instrumented |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| final_product / synthetic n=16 noise=0.0 | smoothing-fw | 1.0003 [1.0003, 1.0003] | 0.030253 [0.030253, 0.030253] | 0.00073523 [0.00073523, 0.00073523] | 0.25915 [0.25915, 0.25915] | 0.00058805 [0.00058805, 0.00058805] | 0.001414 [0.001414, 0.001414] | 1.2928 [1.2928, 1.2928] |
| final_product / synthetic n=16 noise=0.0 | full-kl | 0.015531 [0.015531, 0.015531] | 0.030745 [0.030745, 0.030745] | 0.00071352 [0.00071352, 0.00071352] | 0.0016873 [0.0016873, 0.0016873] | 0.0006274 [0.0006274, 0.0006274] | 0.0013201 [0.0013201, 0.0013201] | 0.050809 [0.050809, 0.050809] |
| final_product / synthetic n=16 noise=0.0 | stochastic-kl | 0.015615 [0.015615, 0.015615] | 0.030634 [0.030634, 0.030634] | 0.00073254 [0.00073254, 0.00073254] | 0.0017569 [0.0017569, 0.0017569] | 0.00059904 [0.00059904, 0.00059904] | 0.0012413 [0.0012413, 0.0012413] | 0.05076 [0.05076, 0.05076] |
| final_product / synthetic n=16 noise=0.0 | fw | 0.015795 [0.015795, 0.015795] | 0.030189 [0.030189, 0.030189] | 0.00069548 [0.00069548, 0.00069548] | 0.0016488 [0.0016488, 0.0016488] | 0.00060131 [0.00060131, 0.00060131] | 0.0012311 [0.0012311, 0.0012311] | 0.050468 [0.050468, 0.050468] |

Peak memory is the maximum observed process peak across the summarized runs, including diagnostics; it is not isolated solver workspace.

| Phase / data | Method | GPU allocated max (GiB) | GPU reserved max (GiB) | CPU RSS max (MiB) | Failed/invalid | Pending | Native early/max-iter | Warnings | Underflow steps | Inner failures / unknown units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| final_product / synthetic n=16 noise=0.0 | smoothing-fw | — | — | 357.34 | 0 | 0 | 0 | 0 | 0 | 0 known / 0 unknown |
| final_product / synthetic n=16 noise=0.0 | full-kl | — | — | 360.07 | 0 | 0 | 0 | 0 | 0 | 0 known / 0 unknown |
| final_product / synthetic n=16 noise=0.0 | stochastic-kl | — | — | 361.79 | 0 | 0 | 0 | 0 | 0 | 0 known / 0 unknown |
| final_product / synthetic n=16 noise=0.0 | fw | — | — | 361.79 | 0 | 0 | 0 | 0 | 0 | 0 known / 0 unknown |

| Method | Candidate | Eligible | Normalized flat-median score | Valid / prescribed runs | Inner failures | Warnings |
|---|---|---|---:|---:|---:|---:|
| full-kl | c00 | True | 0.99913 | 1/1 | 0 known; 0 unknown runs | 0 |
| fw | c00 | True | 0 | 1/1 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c00 | True | 0.99694 | 1/1 | 0 known; 0 unknown runs | 0 |
| stochastic-kl | c00 | True | 0.99863 | 1/1 | 0 known; 0 unknown runs | 0 |

Authoritative selection audit (no selection is made from held-out data):

| Method | Supplied choice | Audit | Pending tuning runs |
|---|---|---|---:|
| full-kl | c00 | matches | 0 |
| fw | c00 | matches | 0 |
| smoothing-fw | c00 | matches | 0 |
| stochastic-kl | c00 | matches | 0 |

Every run, including failures and unavailable final metrics, is listed in [runs.csv](runs.csv) and report.json. The CSV includes actual rounded float64 loss, raw row/column residuals, true solver-budget coverage, timing, memory, inner failures, warnings, and failure penalties. Native stopping flags never imply convergence; native conditional samples are not converted into sketch-probe counts.

Replicate statistics use unique stochastic/initialization units at a fixed instance. Identical deterministic product starts contribute once, so they never acquire artificial error bars. Across different problem instances, plots use one within-instance median per problem and show the median/IQR across those problem values. A single fixed BMMC instance instead shows stochastic/initialization variation. IQRs describe dispersion, not confidence intervals or theorem expectations.

Checkpoint summaries contain observed outputs only; missing windows are never filled or interpolated. Horizontal bars show IQR of actual observation times at a common requested window, never an assigned nominal timestamp. Open plot markers indicate incomplete coverage. Native terminal outputs produced before the budget remain recorded at their actual time in the run table and are not extended to later checkpoints. Final failure penalties and all missing-coverage counts remain separate from descriptive statistics over available checkpoints.

No fitted slopes, best-so-far output selection, deterministic duplicate error bars, or GPU/CPU algorithmic speedup claims are produced.


Stage D assessment remains an evidence review; this report does not automatically open the scale-up gate.

| Requirement | Evidence and limitation |
|---|---|
| Feasible quality | Current rounded checkpoint curves and independently checked final float64 endpoint summaries, with failures and native early termination retained. |
| Oracle trade-off | Use the dedicated same-device cost/error/regret diagnostics. Kernel call counts are not interchangeable with native sample counts. |
| Memory | Tables report observed instrumented process peaks, including preparation/diagnostics. Small-instance peaks do not establish capacity or speed at larger sizes. |

Complete the prescribed Stage C evidence; require correct kernels, a useful measured oracle cost/error trade-off, competitive feasible outputs, and memory headroom. No arbitrary quality threshold or automatic winner is introduced.

Dedicated figures use true gradients. Practical checkpoint gaps are sparse diagnostics, not a full-trajectory theorem average. Tuned constants and row projection are not new theorem guarantees.

| Required final phase | Prescribed runs | Supplied runs | Terminal receipts |
|---|---:|---:|---:|
| final_product | — | 4 | 4 |
| final_randomized | — | 0 | 0 |
| small_pg | — | 0 | 0 |

- [protocol coverage](../../PROTOCOL_COVERAGE.md)
- [oracle diagnostics](../../a100_diagnostics/report.json)
- [true gap vs iteration](../../a100_diagnostics/true_smoothed_gap_vs_iteration.png)
- [true gap vs oracle work](../../a100_diagnostics/true_smoothed_gap_vs_probes.png)
- [theory replicates](../../a100_theory/theory_replicates.json)

Noiseless synthetic instances have known f*=0; noisy synthetic and BMMC objectives are reported as absolute feasible losses, without a certified optimum or clipped reference gaps. Randomized starts pair stochastic and initialization seeds, so their variation is not attributed to initialization alone. The row-projection ablation is separately tuned and cannot isolate the row LMO while ignoring changed constants. Supplemental stochastic KL momentum inherits its selected KL step/tolerance and selected smoothing momentum constant without extra tuning; it is an empirical ablation, not a new theorem guarantee.

Figures:

- [panel_00_rounded_objective.png](panel_00_rounded_objective.png)
- [panel_00_rounded_objective.pdf](panel_00_rounded_objective.pdf)
- [panel_00_original_gap.png](panel_00_original_gap.png)
- [panel_00_original_gap.pdf](panel_00_original_gap.pdf)
- [panel_00_raw_feasibility.png](panel_00_raw_feasibility.png)
- [panel_00_raw_feasibility.pdf](panel_00_raw_feasibility.pdf)
- [tuning_candidate_scores.png](tuning_candidate_scores.png)
- [tuning_candidate_scores.pdf](tuning_candidate_scores.pdf)
