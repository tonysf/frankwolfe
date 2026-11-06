# Final checkpoint coverage audit

Result: **passed**. All 23 missing windows are explicitly recorded in the original traces. The reporter preserves their absence and the actual times of observed outputs. This read-only audit checked 723 runs, 4233 CSV rows and 3485 observed windows; 103219 consistency checks found 0 issues.

| Phase | Runs | Expected windows | Observed | Missing | Native failures | Native early stop | Native first-iteration overshoot | Other first-iteration overshoot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final_product | 253 | 1518 | 1505 | 13 | 7 | 2 | 4 | 0 |
| final_randomized | 330 | 1320 | 1312 | 8 | 0 | 0 | 0 | 8 |
| momentum_product | 55 | 330 | 329 | 1 | 0 | 0 | 0 | 1 |
| momentum_randomized | 55 | 220 | 219 | 1 | 0 | 0 | 0 | 1 |
| small_pg | 30 | 120 | 120 | 0 | 0 | 0 | 0 | 0 |

The 13 missing product windows are **7 failure + 2 early-stop + 4 native first-iteration overshoot** windows. Detailed failure causes are already audited in [FINAL_FAILURES.md](FINAL_FAILURES.md).

## Ten non-native missing windows

All seven affected runs record a checkpoint for the 1 s target at completed iteration 1 and explicit missing rows for the other crossed targets. The table separates timed initialization from the first update. Five missing targets were already crossed during initialization; five were crossed by the first update.

| Phase / method | Instance | Seed / init seed | Initialization solver s | First update s | Iteration-1 solver s | Missing targets s |
|---|---|---|---:|---:|---:|---|
| final_randomized / smoothing-fw | metric512_s2000.npz | 102 / 900102 | 5.271788 | 0.004522 | 5.276310 | 3.0 |
| final_randomized / smoothing-fw | metric512_s2001.npz | 100 / 900100 | 7.624164 | 0.004157 | 7.628322 | 3.0 |
| final_randomized / smoothing-fw-exact | metric512_s2004.npz | 104 / 900104 | 24.637820 | 0.003756 | 24.641576 | 3.0, 10.0 |
| final_randomized / fw | bmmc1000.npz | 100 / 900100 | 0.115010 | 20.754751 | 20.869761 | 3.0, 10.0 |
| final_randomized / fw | bmmc1000.npz | 102 / 900102 | 0.117941 | 20.770590 | 20.888531 | 3.0, 10.0 |
| momentum_product / stochastic-kl-momentum | metric512_s2000.npz | 101 / 100 | 3.934619 | 0.168308 | 4.102928 | 3.0 |
| momentum_randomized / stochastic-kl-momentum | metric512_s2000.npz | 101 / 900101 | 2.714012 | 0.377952 | 3.091964 | 3.0 |

The long smoothing and product-momentum initialization clocks include the initial oracle and setup; they are not the time of a single optimizer update. The randomized FW first updates take about 20.8 s each. Their logs and final counters confirm successful 26-update runs, but sparse timings do not isolate the source of this delay. Inner tolerance failures are separately recorded for the two KL momentum runs and do not replace or explain the checkpoint bookkeeping.

## Native early stops and overshoots

| Instance | Seed | Event | Completed iterations | Actual solver s | Missing targets s |
|---|---:|---|---:|---:|---|
| metric512_s2000.npz | 103 | Early stop | 641 | 268.229638 | 300.0 |
| metric512_s2001.npz | 104 | Iteration-1 overshoot | 1 | 3.661225 | 3.0 |
| metric512_s2002.npz | 100 | Early stop | 413 | 159.324301 | 300.0 |
| metric512_s2004.npz | 102 | Iteration-1 overshoot | 1 | 6.377628 | 3.0 |
| metric512_noisy_s2000.npz | 101 | Iteration-1 overshoot | 1 | 13.252555 | 3.0, 10.0 |

Both early stops return with `completed_with_warnings`, without exhausting the iteration cap or reaching the time budget; they miss only 300 s. The 268.230 s stop has 500 recorded Sinkhorn nonconvergence warnings; the 159.324 s stop has 307. These are native stopping events, not convergence certificates. Their final outputs retain `kind=final`, a blank nominal target, and actual times; no 300 s output is inserted. Native initialization is included in the first native clock segment.

## Archived policy and clock interpretation

The audit verified every archived Python snapshot against source revision `af7b7c549351b81de2aa886b987bc3e56ffeff9c` and metadata hashes. In that source, `run_benchmark.py:38–53` advances through all crossed thresholds and returns the earliest target plus a list of misses. Lines 381–415 time initialization before the initial diagnostic. Lines 488–499 finish and synchronize the update, then consume the grid and write explicit missed rows. The native callback applies the same grid (`run_native_sampled.py:151–166`); lines 187–232 keep native final evaluations separate and mark unreached windows. The JSON includes source hashes, per-run metadata/trace hashes, affected stdout evidence, and every phase/method/instance denominator.

Every missing overshoot row has the same iteration and actual solver clock as its earlier observed target. No completed iteration is reused for multiple observed windows; no observation is timestamped earlier than its target. Every report observation agrees exactly with the corresponding original CSV values. All affected stdout summaries agree with original metadata status, duration and completed count.

When one completion crosses several targets, there is only one stored evaluated iterate. The earliest target receives that observation at its actual time; the remaining targets remain missing. A target label such as 1 s must not be read as an output available at precisely 1 s. For example, the exact-smoothing randomized run above first produces a completed iterate at 24.642 solver seconds.

`time_to_this_feasible_output = actual_solver_seconds + repair_seconds_current`. It excludes earlier diagnostics and repairs and is not total instrumented wall time. Preparation, diagnostics, cumulative repair and total elapsed time remain separate metadata fields. An initial diagnostic may have a larger per-output clock than the next point when its repair was more expensive; this does not imply a backward solver clock. The sparse traces cannot empirically reconstruct every prior iterate timestamp or identify the hardware cause of long initialization/first-update intervals.

## Missing counts by method and instance

| Phase | Method | Instance | Runs | Expected windows | Observed | Missing |
|---|---|---|---:|---:|---:|---:|
| final_product | native-sampled-gw | bmmc1000.npz | 5 | 30 | 24 | 6 |
| final_product | native-sampled-gw | metric512_noisy_s2000.npz | 5 | 30 | 27 | 3 |
| final_product | native-sampled-gw | metric512_s2000.npz | 5 | 30 | 29 | 1 |
| final_product | native-sampled-gw | metric512_s2001.npz | 5 | 30 | 29 | 1 |
| final_product | native-sampled-gw | metric512_s2002.npz | 5 | 30 | 29 | 1 |
| final_product | native-sampled-gw | metric512_s2004.npz | 5 | 30 | 29 | 1 |
| final_randomized | fw | bmmc1000.npz | 5 | 20 | 16 | 4 |
| final_randomized | smoothing-fw | metric512_s2000.npz | 5 | 20 | 19 | 1 |
| final_randomized | smoothing-fw | metric512_s2001.npz | 5 | 20 | 19 | 1 |
| final_randomized | smoothing-fw-exact | metric512_s2004.npz | 5 | 20 | 18 | 2 |
| momentum_product | stochastic-kl-momentum | metric512_s2000.npz | 5 | 30 | 29 | 1 |
| momentum_randomized | stochastic-kl-momentum | metric512_s2000.npz | 5 | 20 | 19 | 1 |

Groups with zero missing windows are retained in the complete [JSON audit](FINAL_CHECKPOINT_COVERAGE.json). No data were recomputed or amended. The existing replay monitor continues independently.

Final report SHA-256: `39730329a43e14f0ce3af063f8263bb5b6b45e6acc45696ccbff164813d9086c`.
