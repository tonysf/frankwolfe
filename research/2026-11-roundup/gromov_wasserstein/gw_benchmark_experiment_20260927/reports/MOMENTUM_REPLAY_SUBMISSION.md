# Momentum diagnostic replay submission

The A100 preflight and independent artifact audit passed before submission. Six production pilot replays are complete. The remaining **790 diagnostic replays are submitted**, with no changes to the original optimization configurations, iteration counts, outputs, selection or timing observations.

| Purpose | Array | Histories | Shards | Dependency |
|---|---:|---:|---:|---|
| Tuning diagnostics | 1961715 | 504 | 8 × 63 | Audited preflight complete |
| Final diagnostics | 1961716 | 286 | 30 × 9 and 2 × 8 | `afterok:1961580` |

Each task requests one A100, two CPUs, 12 GiB host memory and 90 minutes on Ruche `gpua100`, account `fwllm`. Both arrays have a four-task cap; Ruche's user-wide four-GPU limit governs simultaneous allocations across the original and diagnostic arrays. Runtime source remains the immutable `b5851980646be55a148b083a89cfbf6153dac77e` release. The six pilots are not resubmitted.

The first tuning submission returned exit 1, empty stdout and “Job dependency problem”; no job was created. Preflight 1961687 had completed successfully but had already been purged from Slurm's live job list. Accounting still recorded `COMPLETED`, exit `0:0`; the collected independent audit passed 18/18 checks. A scheduler-only amendment removed this satisfied, unavailable dependency. The final replay retains its dependency on unfinished original array 1961580. An independent exact comparison confirmed that all manifests, source revision, output roots, shard assignments and resource requests otherwise match the original frozen plan.

- [Successful preflight audit](momentum_replay_a100_preflight.md)
- [Original dispatch audit](momentum_replay_dispatch_audit.md)
- [Active scheduler plan](../configs/momentum_diagnostic_replay_dispatch_active.json), SHA-256 `4d44c521477333ee6d58627b8a19fe64b2101492beb2e77f02c978b3a2a36e99`
- [Rejected submission receipt](replay_tuning_submission.json)
- [Accepted tuning receipt](replay_tuning_submission_active.json)
- [Accepted final receipt](replay_finals_submission_active.json)

A replay must reproduce checkpoint metrics, counters and both endpoint arrays before its diagnostics are accepted. Failures stay visible and are not replaced with invented estimates. Controller elapsed time and Slurm allocation time are reported as supplementary experiment cost; nested worker times are not added twice. Original solver time remains the practical comparison axis, with original output-repair cost where required.

Complete reporting still requires both arrays to finish, stable checksum collection, the combined 796-record companion report, and independent binding to all collected original archives.
