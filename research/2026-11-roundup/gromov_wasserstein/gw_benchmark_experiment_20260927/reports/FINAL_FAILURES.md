# Selected final native failures

**Five native numerical failures; no infrastructure failure among these records.** The verified product archive contains 253 terminal runs: 141 completed, 57 completed with custom inner failures, 50 completed with native warnings, and five failed. Native Sampled GW has 55 prescribed product trials: 50 returned outputs with warnings and five failed. This audit covers the failures; warnings are not convergence evidence.

All five failed at unchanged native conditional target sampling: `RandomState.choice` received NaN probabilities from `T[row, :] / sum(T[row, :])`. Four are fixed BMMC stochastic seeds 100–103; one is noisy synthetic n=512, problem seed 2000, stochastic seed 103. All retain selected c08, float32, batch 128, epsilon 0.01, native product initialization and a 300-second target budget. Initialization seed 100 is recorded but the native product start consumes no initialization randomness. POT is 0.9.6.post1; source is immutable `af7b7c549351b81de2aa886b987bc3e56ffeff9c`.

| Instance / stochastic seed | Completed updates | Failure attempt | Warnings: nonconvergence + numerical | Observed targets (s) | Missing targets (s) |
|---|---:|---:|---:|---|---|
| bmmc1000.npz / 100 | 342 | 343 | 338 + 2 | 1, 3, 10, 30, 100 | 300 |
| bmmc1000.npz / 101 | 381 | 382 | 376 + 2 | 1, 3, 10, 30, 100 | 300 |
| bmmc1000.npz / 102 | 178 | 179 | 175 + 2 | 1, 3, 10, 30 | 100, 300 |
| bmmc1000.npz / 103 | 131 | 132 | 128 + 1 | 1, 3, 10, 30 | 100, 300 |
| metric512_noisy_s2000.npz / 103 | 540 | 541 | 538 + 2 | 1, 3, 10, 30, 100 | 300 |

The five records preserve **23 observed budget checkpoints and 7 missing windows**, plus five initial rows. Their **1564 warnings** comprise 1555 Sinkhorn nonconvergence and 9 numerical errors at inner iteration zero. Each run ends with one or two numerical-error warnings before the sampling exception. Native inner iteration/failure totals remain unknown; warning counts are not substituted for them.

| Instance / seed | Last callback solver (s), lower bound | Runner instrumented elapsed (s) | Launcher process (s) | Last output: target / actual solver / output time (s) |
|---|---:|---:|---:|---|
| bmmc1000.npz / 100 | 171.238500 | 172.705742 | 176.546119 | 100 / 100.037642 / 100.038430 |
| bmmc1000.npz / 101 | 201.238053 | 203.288521 | 234.280602 | 100 / 100.206779 / 100.207614 |
| bmmc1000.npz / 102 | 87.712685 | 88.961235 | 92.187632 | 30 / 30.173325 / 30.174060 |
| bmmc1000.npz / 103 | 65.688757 | 66.997755 | 70.198363 | 30 / 30.038290 / 30.039044 |
| metric512_noisy_s2000.npz / 103 | 283.582349 | 285.114978 | 294.962152 | 100 / 100.276506 / 100.277321 |

Every `failure_timing_complete` is false. Solver time stops at the last completed callback; the failing attempt is not separately timed. Runner and launcher elapsed include rejected work and different preparation/startup scopes, so they must not be added together. The unclassified remainder is retained in JSON and is not a measured failed-iteration duration. No 300-second output was produced.

| Instance / seed | Last observed rounded loss, float32 | Rounded both-L1 | Last observed raw zero entries / total |
|---|---:|---:|---:|
| bmmc1000.npz / 100 | 0.1519754529 | 1.0000076e-07 | 970331 / 1000000 |
| bmmc1000.npz / 101 | 0.151232659817 | 1.086155e-07 | 969359 / 1000000 |
| bmmc1000.npz / 102 | 0.151231944561 | 1.0512304e-07 | 899900 / 1000000 |
| bmmc1000.npz / 103 | 0.152908682823 | 1.0302756e-07 | 899114 / 1000000 |
| metric512_noisy_s2000.npz / 103 | 0.0326602458954 | 1.1641532e-09 | 261626 / 262144 |

These are earlier checkpoint values, not replacement terminal outputs. All five `final_float64_validation` fields are null and all five `final.npz` files are absent. The failing transport/probability vectors were not saved. Last-observed diagnostic files are earlier snapshots; this audit does not load or recompute them.

Independent source review removed only the monitoring hooks from each archived instrumented function and recovered the native semantic AST exactly. Native archive hashes, custom source hashes, frozen Git objects, recorded input/specification hashes, campaign receipts and terminal ledger records agree. Each launcher returned 1 with `infrastructure_error=null`; the traceback identifies native conditional sampling. The preserved Sinkhorn source warns on zero scaling denominators or nonfinite scaling factors, restores prior scaling vectors and returns using its current kernel. Observed support loss and late numerical warnings are consistent with finite-precision degradation. **The first invalid arithmetic operation is not identified:** zero-row normalization and nonfinite-state propagation cannot be distinguished from these logs. No correction, retry or numerical rerun was used.

The previously audited [first native failure](FINAL_NATIVE_FAILURE.md) is unchanged: metadata, trace, receipt, failure event and execution traceback are byte-identical to its preview; counters, times and warnings agree. The four BMMC failures share the observed exception/site, without proving the same underlying first invalid operation.

All inspected trial-file hashes match the stable collection inventory and remain unchanged. [FINAL_FAILURES.json](FINAL_FAILURES.json) retains per-trial checkpoints, warnings, timing, source/receipt/file hashes and coverage. The final campaign report must preserve failed statuses, ineligible finals (+infinite penalties), earlier actual-time observations and missing-window denominators. No setting is reselected from held-out outcomes.

The complete [selected final report](stage_c_final/report.md) was cross-checked at report SHA256 `39730329a43e14f0ce3af063f8263bb5b6b45e6acc45696ccbff164813d9086c`: all five failed identities, failed/provenance-valid/ineligible statuses, infinite penalties, warnings, timing dictionaries, and 23 observed checkpoint losses/actual times match exactly; their seven missing windows remain missing. The complete report has 723 terminal records and 718 valid finals. It has 23 missing checkpoint windows across all phases; only seven are attributed to these five failures.

The initial cross-check used predecessor report SHA256 `9c262c04c5323944c33080f6830a57291d1127676611fca8c5f55bcce8edff4f`, preserved at [the pre-margin-fix archive](stage_c_final_before_plot_margin_fix/report.json). The current binding follows the [plot-only equality audit](FINAL_PLOT_MARGIN_FIX.json): only generation timestamp and reporter source hash changed; all scientific, aggregate and coverage fields are identical. No broader failure audit or numerical computation was repeated.
