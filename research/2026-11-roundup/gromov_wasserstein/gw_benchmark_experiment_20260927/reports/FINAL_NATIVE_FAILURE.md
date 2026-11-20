# Failed final native Sampled GW trial

**Classification: a native numerical failure, not an infrastructure failure.** Trial `final_product/native-sampled-gw/c08/metric512_noisy_s2000/s103_i100` failed after 540 completed native iterations/updates, during the conditional sampling of the next iteration. Slurm array 1961580, task 0, job 1961581; source `af7b7c549351b81de2aa886b987bc3e56ffeff9c`; POT 0.9.6.post1; float32, batch 128, epsilon 0.01. This is a failed final replication, not grounds for retuning or replacing the trial.

The preserved [traceback](../ruche_results/final_native_failure_preview/trial/native_source/execution_failure.json) identifies the unchanged native expression `T[row,:] / sum(T[row,:])` passed to NumPy's conditional sampler. `RandomState.choice` raised `ValueError: probabilities contain NaN`. There were **540 warnings**: 538 Sinkhorn nonconvergence warnings, followed by two numerical-error warnings at inner iteration 0. The launcher recorded exit code 1, status `failed`, and no infrastructure error.

The copied native source shows the original cost normalization, proximal log correction, Sinkhorn call, and transport update unchanged. Independently removing the monitoring hooks exactly recovers the native semantic AST and its archived hash. All archived source hashes and the receipt's frozen manifest/specification match. The monitor explicitly propagates native exceptions and does not count the failed attempt as completed. No wrapper-specific fault appears in the traceback or source audit.

The last saved diagnostic coupling is from iteration 190 at 100.276506 solver seconds. It is finite, has no zero rows, and has 261,626 zero entries out of 262,144. POT's preserved Sinkhorn source emits its numerical warning when scaling encounters zero denominators or nonfinite factors; it restores prior scaling vectors and returns using the same kernel. These observations are consistent with native finite-precision/support degradation. **The failing transport/probability vector was not saved**, so the exact first invalid operation—zero-row normalization versus nonfinite-state propagation—is not proven. No rerun was used to infer it.

| Recorded cost | Seconds | Interpretation |
|---|---:|---|
| Solver clock | 283.582349 | Last completed callback; a lower bound on solver work because the failing attempt is not separately timed |
| Instrumented runner elapsed | 285.114978 | Includes rejected work and diagnostics/preparation/serialization before the final metadata write |
| Launcher process elapsed | 294.962152 | Includes interpreter/import startup and process execution; not added to the preceding elapsed time |

`failure_timing_complete` is explicitly false. The unclassified timing residual is preserved; it must not be relabeled as a measured duration of the failing native iteration. Native inner iteration/failure counts remain unknown, and warnings are reported separately.

The reporter check retains observations at requested 1/3/10/30/100-second windows, with actual solver times 1.209560/3.325954/10.187174/30.220249/100.276506 seconds. The 300-second window remains missing. Status remains `failed`, the final output is ineligible with +infinite failure penalty, and no final float64 score or `final.npz` exists. Earlier float32 checkpoint metrics are not promoted into a replacement final result. This check used the genuine copied receipt/metadata through a temporary path mapping in parser-smoke scope; full campaign-ledger validation remains part of the final production summary.

[All 17 trial/log files](../ruche_results/final_native_failure_preview/transfer_verification.json) were copied into a new preview folder and verified against identical remote hashes before and after transfer. The exact [Sinkhorn source named by the warnings](../ruche_results/final_native_failure_preview/sinkhorn_source_verification.json) was separately copied with stable hashes; it is byte-identical to the existing local archived Sinkhorn source. The snapshot totals about 1.4 MB plus 48 KB of Sinkhorn source. No other trial or input dataset was transferred, and no solver rerun, retuning, cancellation, or runtime change occurred.

Machine-readable findings and links: [FINAL_NATIVE_FAILURE.json](FINAL_NATIVE_FAILURE.json). Independent checks: [audit_checks.json](../ruche_results/final_native_failure_preview/audit_checks.json).
