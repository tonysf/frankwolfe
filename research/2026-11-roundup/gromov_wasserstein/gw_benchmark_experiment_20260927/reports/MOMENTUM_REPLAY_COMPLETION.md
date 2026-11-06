# Completed momentum diagnostics

All **796/796** prescribed companions verified: 504 development trajectories, six pilots, 121 product-start finals and 165 randomized-start finals. The [combined report](momentum_replay_completion/report.md) has 3,714 diagnostic rows, including initial points. Four originally missing windows remain missing. No optimizer, selection or original timing was replaced.

The [cross-archive audit](replay_original_bindings_final.json) matches every companion to independently collected original metadata, traces, configuration, source snapshots, saved endpoints, campaign receipts and canonical input hashes. It inspected 7,975 unique files (2,392,177,130 bytes), found zero mismatches and confirmed that inputs remained unchanged. Accepted replay proofs additionally require exact original checkpoint/counter agreement, unchanged random streams and bitwise raw/rounded endpoints. These are reconstructed diagnostics, not 796 additional independent optimization outcomes.

Final replay array 1961716 completed all 32 tasks. Stable transfer verified 1,495 files (11,343,995 bytes) plus 32 Slurm logs. Its remote inventories before/after collection have SHA-256 `9e87579d16a21ac14d14148481f58ef2462225f9619cdd54b7356bb5dea6864f`. The six pilot companions remain in the verified preflight collection; tuning has 2,537 verified files. Source runtime is immutable `b5851980646be55a148b083a89cfbf6153dac77e`.

At output P_K, the primary error compares the actual last-used d_(K−1) with the gradient at its input P_(K−1); the separately labeled stale error compares it with the gradient at P_K. Initial K=0 uses d0/P0. No unused terminal oracle sample is created. The displayed original output clock labels P_K, and these sparse diagnostics do not replace the theorem's full-trajectory statistic.

At the 300-second product checkpoint, stochastic-smoothing primary Frobenius-error medians are 21.8532 (clean n=512), 22.1327 (noisy n=512), and 33.1170 (fixed BMMC n=1000). Exact-gradient smoothing gives 0.000203851, 0.000199779 and 0.00246245. Exact-gradient momentum still has EMA tracking error. Row-projection medians are 42.2632, 42.1619 and 62.5686 under its separately selected momentum constants. These unnormalized norms are not a dimension-free comparison or a causal explanation of rounded objective differences; sampled noise and momentum lag both contribute. Finite-window decreases do not establish an asymptotic rate.

Both final PNGs were visually reviewed, with the paired PDFs checked and hashed in [figure QA](MOMENTUM_REPLAY_FIGURE_QA.json). Linear y-axes place the small exact-gradient errors near zero; full values and prescribed/observed populations remain in [curve_summaries.csv](momentum_replay_completion/curve_summaries.csv).

Summed controller elapsed time is **50104.835504 seconds** (13.918010 hours), counted once per companion. Nested worker times are not added. [Scheduler accounting](COMPUTE_ACCOUNTING.md) separately records 14.641667 A100 allocation-hours for replay validation and reconstruction, including additional preflight work. Neither number is campaign wall time or measured GPU utilization.

Combined report SHA-256: `16a71de254610004cd3744cd510593b01061cb7f8e374e00121d8ebbfb25066c`. Original-binding audit SHA-256: `df6d10c825e5e751992fa7759ed0cbac412fe2806a64f37de9073d64d880052b`.
