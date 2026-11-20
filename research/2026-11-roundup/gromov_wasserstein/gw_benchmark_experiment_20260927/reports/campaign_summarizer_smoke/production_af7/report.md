# Campaign results and progress

Manifest-driven report for the supplied campaign phases. Completion refers only to these manifests; it does not assert that every protocol phase or the full study is complete. Statistics are descriptive and retain failed-run and missing-checkpoint coverage.

Status: **partial**. Expected runs: 1164; attempted: 1; terminal records: 1; valid final outputs: 1.

Execution completion includes failed terminal trials; output validity and selection readiness are separate report fields. Saved raw/rounded plan marginals are independently recomputed in NumPy float64 and compared to runner validation; final objective values come from the runner float64 evaluator. Checkpoint curves retain the trace evaluations in the recorded solver precision (float32 in the A100 campaign); they are distinct from final float64 validation. Missing and failed runs carry +infinite tuning penalties. Any such run makes its whole candidate ineligible. Eligible candidate scores are the flat median of prescribed final rounded losses divided by their fixed product-plan losses. Warnings and inner stopping failures remain visible and do not alone disqualify a valid output.

| Phase | Expected | Attempted | Valid final | Failed/invalid/infrastructure | Missing/pending | Early native stop/max-iter | Missing checkpoint observations |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuning | 1164 | 1 | 1 | 0 | 1163 | 0 | 3489 |

| Method | Candidate | Eligible | Normalized flat-median score | Valid / prescribed runs | Inner failures | Warnings |
|---|---|---|---:|---:|---:|---:|
| full-kl | c00 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c01 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c02 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c03 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c04 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c05 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c06 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c07 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c08 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c09 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c10 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| full-kl | c11 | False | +inf | 0/7 | 0 known; 7 unknown runs | 0 |
| native-sampled-gw | c00 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c01 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c02 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c03 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c04 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c05 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c06 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c07 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c08 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c09 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c10 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| native-sampled-gw | c11 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| pg | c00 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c01 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c02 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c03 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c04 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c05 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c06 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c07 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c08 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c09 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c10 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| pg | c11 | False | +inf | 0/6 | 0 known; 6 unknown runs | 0 |
| smoothing-fw | c00 | False | +inf | 1/21 | 0 known; 20 unknown runs | 0 |
| smoothing-fw | c01 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c02 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c03 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c04 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c05 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c06 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c07 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c08 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c09 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c10 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-fw | c11 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c00 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c01 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c02 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c03 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c04 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c05 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c06 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c07 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c08 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c09 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c10 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| smoothing-row-projection | c11 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c00 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c01 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c02 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c03 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c04 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c05 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c06 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c07 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c08 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c09 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c10 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
| stochastic-kl | c11 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |

Authoritative selection audit (no selection is made from held-out data):

| Method | Supplied choice | Audit | Pending tuning runs |
|---|---|---|---:|
| full-kl | not supplied | not_supplied | 84 |
| native-sampled-gw | not supplied | not_supplied | 252 |
| pg | not supplied | not_supplied | 72 |
| smoothing-fw | not supplied | not_supplied | 251 |
| smoothing-row-projection | not supplied | not_supplied | 252 |
| stochastic-kl | not supplied | not_supplied | 252 |

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
| final_product | 253 | 0 | 0 |
| final_randomized | 330 | 0 | 0 |
| small_pg | 30 | 0 | 0 |
| momentum_product | 55 | 0 | 0 |
| momentum_randomized | 55 | 0 | 0 |

- [protocol coverage](../../PROTOCOL_COVERAGE.md)
- [oracle diagnostics](../../a100_diagnostics/report.json)
- [true gap vs iteration](../../a100_diagnostics/true_smoothed_gap_vs_iteration.png)
- [true gap vs oracle work](../../a100_diagnostics/true_smoothed_gap_vs_probes.png)
- [theory replicates](../../a100_theory/theory_replicates.json)

Noiseless synthetic instances have known f*=0; noisy synthetic and BMMC objectives are reported as absolute feasible losses, without a certified optimum or clipped reference gaps. Randomized starts pair stochastic and initialization seeds, so their variation is not attributed to initialization alone. The row-projection ablation is separately tuned and cannot isolate the row LMO while ignoring changed constants. Supplemental stochastic KL momentum inherits its selected KL step/tolerance and selected smoothing momentum constant without extra tuning; it is an empirical ablation, not a new theorem guarantee.

Figures:

No valid observed panel or eligible tuning scores are available yet.
