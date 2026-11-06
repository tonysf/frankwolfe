# Campaign results and progress

Manifest-driven report for the supplied campaign phases. Completion refers only to these manifests; it does not assert that every protocol phase or the full study is complete. Statistics are descriptive and retain failed-run and missing-checkpoint coverage.

Status: **complete_provided_manifests_with_numerical_issues**. Expected runs: 1164; attempted: 1164; terminal records: 1164; valid final outputs: 1164.

Execution completion includes failed terminal trials; output validity and selection readiness are separate report fields. Saved raw/rounded plan marginals are independently recomputed in NumPy float64 and compared to runner validation; final objective values come from the runner float64 evaluator. Checkpoint curves retain the trace evaluations in the recorded solver precision (float32 in the A100 campaign); they are distinct from final float64 validation. Missing and failed runs carry +infinite tuning penalties. Any such run makes its whole candidate ineligible. Eligible candidate scores are the flat median of prescribed final rounded losses divided by their fixed product-plan losses. Warnings and inner stopping failures remain visible and do not alone disqualify a valid output.

| Phase | Expected | Attempted | Valid final | Failed/invalid/infrastructure | Missing/pending | Early native stop/max-iter | Missing checkpoint observations |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuning | 1164 | 1164 | 1164 | 0 | 0 | 0 | 0 |

| Method | Candidate | Eligible | Normalized flat-median score | Valid / prescribed runs | Inner failures | Warnings |
|---|---|---|---:|---:|---:|---:|
| full-kl | c00 | True | 0.99983 | 7/7 | 0 known; 0 unknown runs | 0 |
| full-kl | c01 | True | 0.99984 | 7/7 | 31 known; 0 unknown runs | 0 |
| full-kl | c02 | True | 1 | 7/7 | 291 known; 0 unknown runs | 0 |
| full-kl | c03 | True | 0.99841 | 7/7 | 1 known; 0 unknown runs | 0 |
| full-kl | c04 | True | 0.99827 | 7/7 | 62 known; 0 unknown runs | 0 |
| full-kl | c05 | True | 0.99997 | 7/7 | 286 known; 0 unknown runs | 0 |
| full-kl | c06 | True | 0.010292 | 7/7 | 0 known; 0 unknown runs | 0 |
| full-kl | c07 | True | 0.17558 | 7/7 | 407 known; 0 unknown runs | 0 |
| full-kl | c08 | True | 0.99975 | 7/7 | 289 known; 0 unknown runs | 0 |
| full-kl | c09 | True | 0.010292 | 7/7 | 20 known; 0 unknown runs | 0 |
| full-kl | c10 | True | 0.010292 | 7/7 | 93 known; 0 unknown runs | 0 |
| full-kl | c11 | True | 0.99597 | 7/7 | 294 known; 0 unknown runs | 0 |
| native-sampled-gw | c00 | True | 0.85441 | 21/21 | 0 known; 21 unknown runs | 441 |
| native-sampled-gw | c01 | True | 0.91248 | 21/21 | 0 known; 21 unknown runs | 378 |
| native-sampled-gw | c02 | True | 0.99991 | 21/21 | 0 known; 21 unknown runs | 442 |
| native-sampled-gw | c03 | True | 1 | 21/21 | 0 known; 21 unknown runs | 424 |
| native-sampled-gw | c04 | True | 0.82337 | 21/21 | 0 known; 21 unknown runs | 434 |
| native-sampled-gw | c05 | True | 0.88747 | 21/21 | 0 known; 21 unknown runs | 411 |
| native-sampled-gw | c06 | True | 0.99974 | 21/21 | 0 known; 21 unknown runs | 424 |
| native-sampled-gw | c07 | True | 0.99999 | 21/21 | 0 known; 21 unknown runs | 415 |
| native-sampled-gw | c08 | True | 0.81959 | 21/21 | 0 known; 21 unknown runs | 409 |
| native-sampled-gw | c09 | True | 0.87455 | 21/21 | 0 known; 21 unknown runs | 369 |
| native-sampled-gw | c10 | True | 0.99732 | 21/21 | 0 known; 21 unknown runs | 83 |
| native-sampled-gw | c11 | True | 0.99999 | 21/21 | 0 known; 21 unknown runs | 389 |
| pg | c00 | True | 0.99934 | 6/6 | 0 known; 0 unknown runs | 0 |
| pg | c01 | True | 0.99935 | 6/6 | 0 known; 0 unknown runs | 0 |
| pg | c02 | True | 0.99934 | 6/6 | 0 known; 0 unknown runs | 0 |
| pg | c03 | True | 0.99206 | 6/6 | 0 known; 0 unknown runs | 0 |
| pg | c04 | True | 0.992 | 6/6 | 0 known; 0 unknown runs | 0 |
| pg | c05 | True | 0.99213 | 6/6 | 0 known; 0 unknown runs | 0 |
| pg | c06 | True | 0.0047914 | 6/6 | 16 known; 0 unknown runs | 0 |
| pg | c07 | True | 0.0058815 | 6/6 | 16 known; 0 unknown runs | 0 |
| pg | c08 | True | 0.0064419 | 6/6 | 16 known; 0 unknown runs | 0 |
| pg | c09 | True | 0.0044391 | 6/6 | 19 known; 0 unknown runs | 0 |
| pg | c10 | True | 0.0044391 | 6/6 | 19 known; 0 unknown runs | 0 |
| pg | c11 | True | 0.0044391 | 6/6 | 14 known; 0 unknown runs | 0 |
| smoothing-fw | c00 | True | 0.99999 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c01 | True | 0.99994 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c02 | True | 0.99993 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c03 | True | 0.99999 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c04 | True | 0.99998 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c05 | True | 0.99998 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c06 | True | 0.99999 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c07 | True | 0.99994 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c08 | True | 0.99993 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c09 | True | 0.99999 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c10 | True | 0.99997 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-fw | c11 | True | 0.99998 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c00 | True | 1 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c01 | True | 0.99999 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c02 | True | 0.99996 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c03 | True | 1 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c04 | True | 0.99996 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c05 | True | 0.99987 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c06 | True | 0.99955 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c07 | True | 1 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c08 | True | 0.99971 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c09 | True | 0.9995 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c10 | True | 0.99928 | 21/21 | 0 known; 0 unknown runs | 0 |
| smoothing-row-projection | c11 | True | 1 | 21/21 | 0 known; 0 unknown runs | 0 |
| stochastic-kl | c00 | True | 0.99999 | 21/21 | 0 known; 0 unknown runs | 0 |
| stochastic-kl | c01 | True | 0.99999 | 21/21 | 0 known; 0 unknown runs | 0 |
| stochastic-kl | c02 | True | 1 | 21/21 | 871 known; 0 unknown runs | 0 |
| stochastic-kl | c03 | True | 0.99986 | 21/21 | 1 known; 0 unknown runs | 0 |
| stochastic-kl | c04 | True | 0.99986 | 21/21 | 30 known; 0 unknown runs | 0 |
| stochastic-kl | c05 | True | 1 | 21/21 | 890 known; 0 unknown runs | 0 |
| stochastic-kl | c06 | True | 0.99841 | 21/21 | 0 known; 0 unknown runs | 0 |
| stochastic-kl | c07 | True | 0.99864 | 21/21 | 271 known; 0 unknown runs | 0 |
| stochastic-kl | c08 | True | 0.99997 | 21/21 | 888 known; 0 unknown runs | 0 |
| stochastic-kl | c09 | True | 0.95653 | 21/21 | 2 known; 0 unknown runs | 0 |
| stochastic-kl | c10 | True | 0.99971 | 21/21 | 886 known; 0 unknown runs | 0 |
| stochastic-kl | c11 | True | 0.99969 | 21/21 | 871 known; 0 unknown runs | 0 |

Authoritative selection audit (no selection is made from held-out data):

| Method | Supplied choice | Audit | Pending tuning runs |
|---|---|---|---:|
| full-kl | c06 | matches | 0 |
| native-sampled-gw | c08 | matches | 0 |
| pg | c09 | matches | 0 |
| smoothing-fw | c08 | matches | 0 |
| smoothing-row-projection | c10 | matches | 0 |
| stochastic-kl | c09 | matches | 0 |

Every run, including failures and unavailable final metrics, is listed in [runs.csv](runs.csv) and report.json. The CSV includes actual rounded float64 loss, raw row/column residuals, true solver-budget coverage, timing, memory, inner failures, warnings, and failure penalties. Native stopping flags never imply convergence; native conditional samples are not converted into sketch-probe counts.

Replicate statistics use unique stochastic/initialization units at a fixed instance. Identical deterministic product starts contribute once, so they never acquire artificial error bars. Across different problem instances, plots use one within-instance median per problem and show the median/IQR across those problem values. A single fixed BMMC instance instead shows stochastic/initialization variation. IQRs describe dispersion, not confidence intervals or theorem expectations.

Checkpoint summaries contain observed outputs only; missing windows are never filled or interpolated. Horizontal bars show IQR of actual observation times at a common requested window, never an assigned nominal timestamp. Open plot markers indicate incomplete coverage. Native terminal outputs produced before the budget remain recorded at their actual time in the run table and are not extended to later checkpoints. Final failure penalties and all missing-coverage counts remain separate from descriptive statistics over available checkpoints.

No fitted slopes, best-so-far output selection, deterministic duplicate error bars, or GPU/CPU algorithmic speedup claims are produced.

Run issues and numerical events:

| Run | Status | Inner failures | Warnings | Issue / source |
|---|---|---:|---:|---|
| tuning/full-kl/c01/bmmc1000/s0_i0 | completed_with_inner_failures | 31 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c01/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c02/metric512_s1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c02/metric512_s1000/s0_i0/failures.json |
| tuning/full-kl/c02/metric512_s1001/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c02/metric512_s1001/s0_i0/failures.json |
| tuning/full-kl/c02/metric512_s1002/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c02/metric512_s1002/s0_i0/failures.json |
| tuning/full-kl/c02/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c02/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/full-kl/c02/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c02/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/full-kl/c02/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c02/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/full-kl/c02/bmmc1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c02/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c03/bmmc1000/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c03/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c04/bmmc1000/s0_i0 | completed_with_inner_failures | 62 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c04/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c05/metric512_s1000/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c05/metric512_s1000/s0_i0/failures.json |
| tuning/full-kl/c05/metric512_s1001/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c05/metric512_s1001/s0_i0/failures.json |
| tuning/full-kl/c05/metric512_s1002/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c05/metric512_s1002/s0_i0/failures.json |
| tuning/full-kl/c05/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 42 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c05/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/full-kl/c05/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c05/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/full-kl/c05/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c05/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/full-kl/c05/bmmc1000/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c05/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c07/metric512_s1000/s0_i0 | completed_with_inner_failures | 75 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c07/metric512_s1000/s0_i0/failures.json |
| tuning/full-kl/c07/metric512_s1001/s0_i0 | completed_with_inner_failures | 29 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c07/metric512_s1001/s0_i0/failures.json |
| tuning/full-kl/c07/metric512_s1002/s0_i0 | completed_with_inner_failures | 63 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c07/metric512_s1002/s0_i0/failures.json |
| tuning/full-kl/c07/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 94 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c07/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/full-kl/c07/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 65 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c07/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/full-kl/c07/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 57 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c07/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/full-kl/c07/bmmc1000/s0_i0 | completed_with_inner_failures | 24 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c07/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c08/metric512_s1000/s0_i0 | completed_with_inner_failures | 38 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c08/metric512_s1000/s0_i0/failures.json |
| tuning/full-kl/c08/metric512_s1001/s0_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c08/metric512_s1001/s0_i0/failures.json |
| tuning/full-kl/c08/metric512_s1002/s0_i0 | completed_with_inner_failures | 39 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c08/metric512_s1002/s0_i0/failures.json |
| tuning/full-kl/c08/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 45 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c08/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/full-kl/c08/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c08/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/full-kl/c08/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c08/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/full-kl/c08/bmmc1000/s0_i0 | completed_with_inner_failures | 42 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c08/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c09/bmmc1000/s0_i0 | completed_with_inner_failures | 20 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c09/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c10/metric512_s1000/s0_i0 | completed_with_inner_failures | 14 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c10/metric512_s1000/s0_i0/failures.json |
| tuning/full-kl/c10/metric512_s1001/s0_i0 | completed_with_inner_failures | 5 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c10/metric512_s1001/s0_i0/failures.json |
| tuning/full-kl/c10/metric512_s1002/s0_i0 | completed_with_inner_failures | 8 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c10/metric512_s1002/s0_i0/failures.json |
| tuning/full-kl/c10/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 14 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c10/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/full-kl/c10/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 6 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c10/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/full-kl/c10/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 11 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c10/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/full-kl/c10/bmmc1000/s0_i0 | completed_with_inner_failures | 35 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c10/bmmc1000/s0_i0/failures.json |
| tuning/full-kl/c11/metric512_s1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c11/metric512_s1000/s0_i0/failures.json |
| tuning/full-kl/c11/metric512_s1001/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c11/metric512_s1001/s0_i0/failures.json |
| tuning/full-kl/c11/metric512_s1002/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c11/metric512_s1002/s0_i0/failures.json |
| tuning/full-kl/c11/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c11/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/full-kl/c11/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 39 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c11/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/full-kl/c11/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c11/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/full-kl/c11/bmmc1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/full-kl/c11/bmmc1000/s0_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1000/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1001/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1001/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1001/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1002/s0_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1002/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_s1002/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1000/s1_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1001/s1_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1001/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1002/s1_i0 | completed_with_inner_failures | 45 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c02/metric512_noisy_s1002/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c02/bmmc1000/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/bmmc1000/s0_i0/failures.json |
| tuning/stochastic-kl/c02/bmmc1000/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/bmmc1000/s1_i0/failures.json |
| tuning/stochastic-kl/c02/bmmc1000/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c02/bmmc1000/s2_i0/failures.json |
| tuning/stochastic-kl/c03/bmmc1000/s1_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c03/bmmc1000/s1_i0/failures.json |
| tuning/stochastic-kl/c04/bmmc1000/s0_i0 | completed_with_inner_failures | 9 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c04/bmmc1000/s0_i0/failures.json |
| tuning/stochastic-kl/c04/bmmc1000/s1_i0 | completed_with_inner_failures | 7 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c04/bmmc1000/s1_i0/failures.json |
| tuning/stochastic-kl/c04/bmmc1000/s2_i0 | completed_with_inner_failures | 14 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c04/bmmc1000/s2_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1000/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1000/s1_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1001/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1001/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1001/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1002/s0_i0 | completed_with_inner_failures | 45 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1002/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_s1002/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1000/s1_i0 | completed_with_inner_failures | 39 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1001/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1001/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1002/s1_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c05/metric512_noisy_s1002/s2_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c05/bmmc1000/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/bmmc1000/s0_i0/failures.json |
| tuning/stochastic-kl/c05/bmmc1000/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/bmmc1000/s1_i0/failures.json |
| tuning/stochastic-kl/c05/bmmc1000/s2_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c05/bmmc1000/s2_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1000/s0_i0 | completed_with_inner_failures | 17 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1000/s1_i0 | completed_with_inner_failures | 17 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1000/s2_i0 | completed_with_inner_failures | 18 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1001/s0_i0 | completed_with_inner_failures | 15 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1001/s1_i0 | completed_with_inner_failures | 13 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1001/s2_i0 | completed_with_inner_failures | 14 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1002/s0_i0 | completed_with_inner_failures | 13 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1002/s1_i0 | completed_with_inner_failures | 11 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_s1002/s2_i0 | completed_with_inner_failures | 14 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 16 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1000/s1_i0 | completed_with_inner_failures | 10 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1000/s2_i0 | completed_with_inner_failures | 16 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 19 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1001/s1_i0 | completed_with_inner_failures | 21 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1001/s2_i0 | completed_with_inner_failures | 12 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 14 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1002/s1_i0 | completed_with_inner_failures | 16 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c07/metric512_noisy_s1002/s2_i0 | completed_with_inner_failures | 14 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c07/bmmc1000/s2_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c07/bmmc1000/s2_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1000/s0_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1000/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1001/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1001/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1001/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1002/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1002/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_s1002/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1000/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1000/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1001/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1001/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 45 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1002/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c08/metric512_noisy_s1002/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c08/bmmc1000/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/bmmc1000/s0_i0/failures.json |
| tuning/stochastic-kl/c08/bmmc1000/s1_i0 | completed_with_inner_failures | 39 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/bmmc1000/s1_i0/failures.json |
| tuning/stochastic-kl/c08/bmmc1000/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c08/bmmc1000/s2_i0/failures.json |
| tuning/stochastic-kl/c09/metric512_s1000/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c09/metric512_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c09/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c09/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1000/s1_i0 | completed_with_inner_failures | 39 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1000/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1001/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1001/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1001/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1002/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1002/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_s1002/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1000/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1001/s1_i0 | completed_with_inner_failures | 39 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1001/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1002/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c10/metric512_noisy_s1002/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c10/bmmc1000/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/bmmc1000/s0_i0/failures.json |
| tuning/stochastic-kl/c10/bmmc1000/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/bmmc1000/s1_i0/failures.json |
| tuning/stochastic-kl/c10/bmmc1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c10/bmmc1000/s2_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1000/s0_i0 | completed_with_inner_failures | 45 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1000/s1_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1001/s0_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1001/s1_i0 | completed_with_inner_failures | 39 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1001/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1002/s0_i0 | completed_with_inner_failures | 43 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1002/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_s1002/s2_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1000/s0_i0 | completed_with_inner_failures | 41 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1000/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1001/s0_i0 | completed_with_inner_failures | 45 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1001/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1001/s2_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1002/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1002/s1_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/stochastic-kl/c11/metric512_noisy_s1002/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/stochastic-kl/c11/bmmc1000/s0_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/bmmc1000/s0_i0/failures.json |
| tuning/stochastic-kl/c11/bmmc1000/s1_i0 | completed_with_inner_failures | 44 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/bmmc1000/s1_i0/failures.json |
| tuning/stochastic-kl/c11/bmmc1000/s2_i0 | completed_with_inner_failures | 40 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/stochastic-kl/c11/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1001/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1001/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1002/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1002/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_s1002/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c00/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c00/bmmc1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c00/bmmc1000/s1_i0 | completed_with_warnings | — | 16 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c00/bmmc1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c00/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1000/s0_i0 | completed_with_warnings | — | 16 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1001/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1001/s1_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1001/s2_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1002/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1002/s1_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_s1002/s2_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 15 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 15 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c01/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c01/bmmc1000/s0_i0 | completed_with_warnings | — | 16 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c01/bmmc1000/s1_i0 | completed_with_warnings | — | 16 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c01/bmmc1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c01/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1000/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1000/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1001/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1001/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1002/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1002/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_s1002/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c02/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c02/bmmc1000/s0_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c02/bmmc1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c02/bmmc1000/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c02/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1000/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1001/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1001/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1002/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_s1002/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c03/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c03/bmmc1000/s0_i0 | completed_with_warnings | — | 12 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c03/bmmc1000/s1_i0 | completed_with_warnings | — | 12 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c03/bmmc1000/s2_i0 | completed_with_warnings | — | 14 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c03/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1000/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1001/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1001/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1002/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_s1002/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c04/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c04/bmmc1000/s0_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c04/bmmc1000/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c04/bmmc1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c04/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1000/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1001/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1001/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1002/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1002/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_s1002/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c05/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c05/bmmc1000/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c05/bmmc1000/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c05/bmmc1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c05/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1000/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1001/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1001/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1001/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1002/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1002/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_s1002/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c06/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c06/bmmc1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c06/bmmc1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c06/bmmc1000/s2_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c06/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1000/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1000/s1_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1000/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1001/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1001/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1001/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1002/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_s1002/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c07/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 22 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c07/bmmc1000/s0_i0 | completed_with_warnings | — | 8 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c07/bmmc1000/s1_i0 | completed_with_warnings | — | 12 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c07/bmmc1000/s2_i0 | completed_with_warnings | — | 11 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c07/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1000/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1001/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1001/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1001/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1002/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1002/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_s1002/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c08/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 21 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c08/bmmc1000/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c08/bmmc1000/s1_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c08/bmmc1000/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c08/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1000/s0_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1000/s1_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1000/s2_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1001/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1001/s1_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1001/s2_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1002/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1002/s1_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_s1002/s2_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 18 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c09/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c09/bmmc1000/s0_i0 | completed_with_warnings | — | 14 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c09/bmmc1000/s1_i0 | completed_with_warnings | — | 17 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c09/bmmc1000/s2_i0 | completed_with_warnings | — | 16 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c09/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1000/s0_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1000/s1_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1000/s2_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1001/s0_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1001/s1_i0 | completed_with_warnings | — | 1 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1001/s2_i0 | completed_with_warnings | — | 1 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1002/s1_i0 | completed_with_warnings | — | 1 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_s1002/s2_i0 | completed_with_warnings | — | 2 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 1 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 3 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 1 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 1 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c10/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 4 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c10/bmmc1000/s0_i0 | completed_with_warnings | — | 16 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c10/bmmc1000/s1_i0 | completed_with_warnings | — | 15 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c10/bmmc1000/s2_i0 | completed_with_warnings | — | 16 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c10/bmmc1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1000/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1000/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1001/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1001/s1_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1001/s2_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1002/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1002/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_s1002/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1000/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1000/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1000/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1000/s2_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1001/s0_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1001/s0_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1001/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1001/s1_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1001/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1001/s2_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1002/s0_i0 | completed_with_warnings | — | 19 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1002/s0_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1002/s1_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1002/s1_i0/failures.json |
| tuning/native-sampled-gw/c11/metric512_noisy_s1002/s2_i0 | completed_with_warnings | — | 20 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/metric512_noisy_s1002/s2_i0/failures.json |
| tuning/native-sampled-gw/c11/bmmc1000/s0_i0 | completed_with_warnings | — | 12 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/bmmc1000/s0_i0/failures.json |
| tuning/native-sampled-gw/c11/bmmc1000/s1_i0 | completed_with_warnings | — | 13 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/bmmc1000/s1_i0/failures.json |
| tuning/native-sampled-gw/c11/bmmc1000/s2_i0 | completed_with_warnings | — | 9 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/native-sampled-gw/c11/bmmc1000/s2_i0/failures.json |
| tuning/pg/c06/metric128_s1000/s0_i0 | completed_with_inner_failures | 4 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c06/metric128_s1000/s0_i0/failures.json |
| tuning/pg/c06/metric128_s1001/s0_i0 | completed_with_inner_failures | 6 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c06/metric128_s1001/s0_i0/failures.json |
| tuning/pg/c06/metric128_s1002/s0_i0 | completed_with_inner_failures | 5 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c06/metric128_s1002/s0_i0/failures.json |
| tuning/pg/c06/metric128_noisy_s1000/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c06/metric128_noisy_s1000/s0_i0/failures.json |
| tuning/pg/c07/metric128_s1000/s0_i0 | completed_with_inner_failures | 4 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c07/metric128_s1000/s0_i0/failures.json |
| tuning/pg/c07/metric128_s1001/s0_i0 | completed_with_inner_failures | 6 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c07/metric128_s1001/s0_i0/failures.json |
| tuning/pg/c07/metric128_s1002/s0_i0 | completed_with_inner_failures | 5 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c07/metric128_s1002/s0_i0/failures.json |
| tuning/pg/c07/metric128_noisy_s1000/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c07/metric128_noisy_s1000/s0_i0/failures.json |
| tuning/pg/c08/metric128_s1000/s0_i0 | completed_with_inner_failures | 4 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c08/metric128_s1000/s0_i0/failures.json |
| tuning/pg/c08/metric128_s1001/s0_i0 | completed_with_inner_failures | 6 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c08/metric128_s1001/s0_i0/failures.json |
| tuning/pg/c08/metric128_s1002/s0_i0 | completed_with_inner_failures | 5 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c08/metric128_s1002/s0_i0/failures.json |
| tuning/pg/c08/metric128_noisy_s1000/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c08/metric128_noisy_s1000/s0_i0/failures.json |
| tuning/pg/c09/metric128_s1000/s0_i0 | completed_with_inner_failures | 8 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c09/metric128_s1000/s0_i0/failures.json |
| tuning/pg/c09/metric128_s1001/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c09/metric128_s1001/s0_i0/failures.json |
| tuning/pg/c09/metric128_s1002/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c09/metric128_s1002/s0_i0/failures.json |
| tuning/pg/c09/metric128_noisy_s1001/s0_i0 | completed_with_inner_failures | 8 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c09/metric128_noisy_s1001/s0_i0/failures.json |
| tuning/pg/c09/metric128_noisy_s1002/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c09/metric128_noisy_s1002/s0_i0/failures.json |
| tuning/pg/c10/metric128_s1000/s0_i0 | completed_with_inner_failures | 8 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c10/metric128_s1000/s0_i0/failures.json |
| tuning/pg/c10/metric128_s1001/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c10/metric128_s1001/s0_i0/failures.json |
| tuning/pg/c10/metric128_s1002/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c10/metric128_s1002/s0_i0/failures.json |
| tuning/pg/c10/metric128_noisy_s1001/s0_i0 | completed_with_inner_failures | 8 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c10/metric128_noisy_s1001/s0_i0/failures.json |
| tuning/pg/c10/metric128_noisy_s1002/s0_i0 | completed_with_inner_failures | 1 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c10/metric128_noisy_s1002/s0_i0/failures.json |
| tuning/pg/c11/metric128_s1000/s0_i0 | completed_with_inner_failures | 7 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c11/metric128_s1000/s0_i0/failures.json |
| tuning/pg/c11/metric128_noisy_s1001/s0_i0 | completed_with_inner_failures | 7 | 0 | /home/tony/gw_benchmark_experiment_20260927/ruche_results/tuning_aaeb5fa/tuning/pg/c11/metric128_noisy_s1001/s0_i0/failures.json |

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

- [protocol coverage](../PROTOCOL_COVERAGE.md)
- [oracle diagnostics](../a100_diagnostics/report.json)
- [true gap vs iteration](../a100_diagnostics/true_smoothed_gap_vs_iteration.png)
- [true gap vs oracle work](../a100_diagnostics/true_smoothed_gap_vs_probes.png)
- [theory replicates](../a100_theory/theory_replicates.json)

Noiseless synthetic instances have known f*=0; noisy synthetic and BMMC objectives are reported as absolute feasible losses, without a certified optimum or clipped reference gaps. Randomized starts pair stochastic and initialization seeds, so their variation is not attributed to initialization alone. The row-projection ablation is separately tuned and cannot isolate the row LMO while ignoring changed constants. Supplemental stochastic KL momentum inherits its selected KL step/tolerance and selected smoothing momentum constant without extra tuning; it is an empirical ablation, not a new theorem guarantee.

Figures:

- [tuning_candidate_scores.png](tuning_candidate_scores.png)
- [tuning_candidate_scores.pdf](tuning_candidate_scores.pdf)
