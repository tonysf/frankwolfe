# Momentum diagnostic replay dispatch audit

**Passed as a prospective dispatch plan; submission remains gated.** No blocking defect was found in `configs/momentum_diagnostic_replay_dispatch.json`. This was a local read-only audit: no remote operation or submission occurred.

| Array | Replays | Partitioning | Largest original-budget proxy | Dependency |
|---|---:|---|---:|---|
| Tuning | 504 | 8 × 63 | 630 s per shard | `afterok:1961687` |
| Finals | 286 | 30 × 9; 2 × 8 | 1,350 s per shard | `afterok:1961580:1961687` |

The 790 original run specifications match the frozen scientific manifests exactly, with no omissions, duplicates, or overlap with the six pilot originals. Final coverage is 121 product trajectories and 165 randomized trajectories; the exact-gradient control contributes 11 product and 55 randomized trajectories. Pilot replays remain exclusively in preflight 1961687.

Both replay manifest SHA-256 values match the files committed at `b5851980646be55a148b083a89cfbf6153dac77e`. The source directory, wrapper path, five positional arguments, shard counts and scheduler dependencies are consistent. Original archived kernel hashes match the original `aaeb5fa`/`af7b7c5` Git objects. Companion roots are disjoint from all original outputs, canonical data and immutable replay source.

The largest final shard contains four product trajectories (1,200 seconds of original solver budgets) and five randomized trajectories (150 seconds), totaling 1,350 seconds. The other final shards total 1,080 seconds (five shards) or 1,050 seconds (two shards). These are workload proxies, not predicted replay durations. The requested 90-minute wall time still requires the A100 preflight evidence.

Each task requests one A100, two CPUs and 12 GiB host memory on `gpua100`, account `fwllm`, with explicit environment/module setup and no CUDA visibility override. Each array has `%4`; the combined four-GPU ceiling relies on Ruche's user-level GPU limit rather than these independent array caps.

**Remaining gate:** preflight 1961687 must finish successfully and its collected artifacts must pass the independent checksum/proof audit before submission. Scheduler dependencies are additional guards. This report does not certify remote release contents or preflight success. Detailed hashes, all 40 shard workloads and the local preflight status snapshot are preserved in `momentum_replay_dispatch_audit.json`.
