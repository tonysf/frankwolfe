# Momentum replay report readiness review

No material scientific or reporting defect found in `summarize_momentum_replays.py` at SHA-256 `8972f375c610cfa3bd5d6653523379a2f1a1dfaae706404c0e4c7683af19a79b`. No reporter changes are recommended from this bounded review.

This is a **510-companion readiness review**, not a completed 796-replay result. The final companion archive was absent locally during the check. All work read local collected artifacts; no SSH query, optimizer execution, new job, or original-output change occurred. The existing replay monitor remained unchanged.

## Genuine completed-data checks

The reporter's `inspect_phase` accepted all 504 tuning and 6 pilot records, with no phase or record integrity errors. Frozen scientific manifests/configuration and the complete cross-phase cell set also passed. Source utility identity is resolved from each controller revision, so later documentation commits do not change the archived execution identity.

| Completed phase | Verified companions | Diagnostic rows | Initial rows | Checkpoint rows | Explicit original misses | Controller elapsed sum, s |
|---|---:|---:|---:|---:|---:|---:|
| Tuning | 504 | 2016 | 504 | 1512 | 0 | 7504.966519 |
| Pilot | 6 | 30 | 6 | 24 | 0 | 194.813852 |
| Total | 510 | 2046 | 510 | 1536 | 0 | 7699.780371 |

All 2,046 rows satisfy output-index and momentum-index alignment. At output iteration `K >= 1`, the primary error uses the actual stored `d_(K-1)` against `grad f(P_(K-1))`; the separate tracking error uses the same stored momentum against `grad f(P_K)`. At `K = 0`, both use the initialized `d0/P0`. The parser validates index fields, unchanged RNG/counters, original scalar proofs, diagnostics hashes and bitwise final raw/rounded outputs. It adds no terminal oracle sample. Output clocks remain the original clocks for `P_K`; the figure subtitle explicitly identifies that the primary quantity belongs to `P_(K-1)`.

The exact-oracle pilot illustrates why exact sampling does not imply zero momentum error: its primary errors range from 0 to about 3.2195 under the EMA recurrence. This is a diagnostic of momentum tracking, not merely sample noise. The reporter preserves that distinction.

## Coverage and plot semantics

A genuine-data partial invocation of `summarize` with all three prescribed manifest/root pairs returned `partial_or_invalid`, 796 prescribed, 510 terminal/verified, 286 missing, provenance valid, execution incomplete, and no figures. It retained all 90 final phase/family/method/target populations with zero observed points, missing coverage, and null statistics. Those populations prescribe 1,386 final windows. No tuning or pilot values were relabeled as final observations. The temporary output was created only to inspect reporting behavior; it is not the authoritative final report.

The plotting function selects only verified `final_product` and `final_randomized` checkpoint observations. Initial/final rows without a nominal checkpoint target stay in per-run JSON and are excluded from these curves. X uses the median **actual original solver time**, not nominal target time or replay duration. X is logarithmic; the Frobenius-error Y axes are linear. Matplotlib's ordinary linear autoscaling includes finite nonnegative medians and IQR bands; the prior practical-report symlog margin issue does not apply. No scale change is justified here.

Synthetic summaries first compute within-instance medians and then across-instance median/IQR; BMMC reports dispersion across runs of its single fixed instance. Product exact-oracle repetitions are not created as new independent samples. The JSON/CSV retain observed and prescribed run/instance counts for every target. Incomplete populations receive open markers, empty windows have no point, and the caption warns that populations may differ. Lines between observed points are display connections, not interpolation records or filled missing observations. Final plotting still needs inspection on the actual collected final data; completed tuning/pilot records cannot validate the final panels' numerical ranges or appearance.

## Replay cost and completion

The combined controller cost is **7,699.780371 seconds**, counting one `controller_elapsed_seconds` value per terminal replay. The nested worker-total sum is 7,386.631753 seconds and is not added again. Every completed record has controller duration at least its worker duration; each worker total equals the sum of its initialization, update, diagnostic, and other components in the saved scalar arithmetic. The report labels this sum as process elapsed cost, not campaign wall time or GPU kernel time.

Completion requires all prescribed records to be terminal and provenance valid. The all-diagnostics-verified flag additionally requires every record to verify; terminal rejected replays remain explicit and contribute no curves. The final report should be generated only after the final companion archive is collected and checksum-verified. This review does not replace that final proof/coverage audit or the prior independent cross-archive checks.

The related completed tuning proof audit is [momentum_replay_tuning_audit.md](momentum_replay_tuning_audit.md); production pilot validation is in [momentum_replay_a100_preflight.md](momentum_replay_a100_preflight.md).
