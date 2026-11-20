# Campaign results and progress

Manifest-driven report for the supplied campaign phases. Completion refers only to these manifests; it does not assert that every protocol phase or the full study is complete. Statistics are descriptive and retain failed-run and missing-checkpoint coverage.

Status: **partial**. Expected runs: 1164; attempted: 0; terminal records: 0; valid final outputs: 0.

Execution completion includes failed terminal trials; output validity and selection readiness are separate report fields. Saved raw/rounded plan marginals are independently recomputed in NumPy float64 and compared to runner validation; objective values come from the runner float64 evaluator. Missing and failed runs carry +infinite tuning penalties. Any such run makes its whole candidate ineligible. Eligible candidate scores are the flat median of prescribed final rounded losses divided by their fixed product-plan losses. Warnings and inner stopping failures remain visible and do not alone disqualify a valid output.

| Phase | Expected | Attempted | Valid final | Failed/invalid/infrastructure | Missing/pending | Early native stop/max-iter | Missing checkpoint observations |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuning | 1164 | 0 | 0 | 0 | 1164 | 0 | 3492 |

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
| smoothing-fw | c00 | False | +inf | 0/21 | 0 known; 21 unknown runs | 0 |
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
| smoothing-fw | not supplied | not_supplied | 252 |
| smoothing-row-projection | not supplied | not_supplied | 252 |
| stochastic-kl | not supplied | not_supplied | 252 |

Every run, including failures and unavailable final metrics, is listed in [runs.csv](runs.csv) and report.json. The CSV includes actual rounded float64 loss, raw row/column residuals, true solver-budget coverage, timing, memory, inner failures, warnings, and failure penalties. Native stopping flags never imply convergence; native conditional samples are not converted into sketch-probe counts.

Replicate statistics use unique stochastic/initialization units at a fixed instance. Identical deterministic product starts contribute once, so they never acquire artificial error bars. Across different problem instances, plots use one within-instance median per problem and show the median/IQR across those problem values. A single fixed BMMC instance instead shows stochastic/initialization variation. IQRs describe dispersion, not confidence intervals or theorem expectations.

Checkpoint summaries contain observed outputs only; missing windows are never filled or interpolated. Horizontal bars show IQR of actual observation times at a common requested window, never an assigned nominal timestamp. Open plot markers indicate incomplete coverage. Native terminal outputs produced before the budget remain recorded at their actual time in the run table and are not extended to later checkpoints. Final failure penalties and all missing-coverage counts remain separate from descriptive statistics over available checkpoints.

No fitted slopes, best-so-far output selection, deterministic duplicate error bars, or GPU/CPU algorithmic speedup claims are produced.


Figures:

No valid observed panel or eligible tuning scores are available yet.
