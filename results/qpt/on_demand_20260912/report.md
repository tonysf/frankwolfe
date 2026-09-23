# On-demand stochastic-FRAMES QPT: measured scaling through 10 qubits

The implementation now executes noisy 10-qubit experiments without an observation table. High-fidelity 10-qubit reconstruction remains unresolved: the best tested final fidelity was 0.000502%. The best 8-qubit run reached 17.9% after 50,000 steps. All submitted jobs are terminal; none remains queued or running.

All runs use complex128/float64, rank one, exact TP gradients, fixed row-indexed Gaussian noise, and disabled GPU autotuning. Source is the preserved v4 snapshot based on ba85bff; no commit or push was made. The new noise recipe is not the old sequential PCG64 realization.

Validation: 470 tests passed on Ruche, plus 7 on-demand tests with GPU trajectory checks on each of V100 and A100. The full local suite passed 414 tests and skipped 56 for unavailable optional dependencies.

**Timing and memory**

Times are medians of two fresh-process timing runs. Memory comes from a separate monitored run. Optimizer time includes on-device target generation and excludes compilation, host preparation/transfers, and metrics. RSS, JAX allocator memory, and sampled driver VRAM measure different things; sampled VRAM can miss brief peaks.

| GPU | n | Backend | Batch | Steps | Optimizer s | Host RSS MiB | JAX peak MiB | Sampled VRAM MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| V100 32GB | 6 | rank-one | 32 | 1000 | 0.325 | 642.8 | 8.8 | 340 |
| V100 32GB | 6 | tensor | 32 | 1000 | 0.734 | 661.4 | 16.8 | 348 |
| V100 32GB | 8 | rank-one | 32 | 1000 | 0.959 | 668.6 | 140.0 | 472 |
| V100 32GB | 8 | tensor | 32 | 1000 | 13.114 | 703.1 | 268.0 | 600 |
| V100 32GB | 10 | rank-one | 32 | 1000 | 14.170 | 877.1 | 1168.3 | 2508 |
| V100 32GB | 10 | tensor | 32 | 1000 | 181.206 | 888.8 | 1680.3 | 2508 |
| V100 32GB | 10 | rank-one | 128 | 1000 | 32.601 | 1130.7 | 4194.9 | 10700 |
| V100 32GB | 10 | rank-one | 512 | 1000 | 115.905 | 1576.4 | 16491.5 | 24702 |
| A100 40GB | 10 | rank-one | 32 | 1000 | 6.127 | 899.3 | 1168.3 | 2616 |
| A100 40GB | 10 | tensor | 32 | 1000 | 75.147 | 917.3 | 1680.3 | 2616 |

At n=10/batch32, rank-one takes 14.170 s/1,000 steps on V100 and 6.127 s on A100 (2.31x faster). Tensor takes 181.206 s and 75.147 s respectively. Rank-one is about 13x faster than tensor on V100.

At n=10 on V100, batches 32/128/512 process approximately 2,258/3,926/4,417 sampled rows per optimizer second. Batch128 achieves 89% of batch512's throughput with a 4.10 GiB allocator peak instead of 16.11 GiB. Batch32 profiles used beta0=10000 and metrics every 100 steps; batch128/512 profiles used beta0=10240000 and metrics every 1000 steps. Optimizer timing excludes those metric evaluations. At batch512, a 10,000-step run took about 19.3 optimizer minutes on V100 and 9.46 minutes on A100; the A100 recovery runs used a different penalty scale, so this is not the matched hardware comparison above.

The n=10 truth factor occupies 16 MiB and took 3.716 seconds to generate. No observation vector is stored; enumerating its 63,403,380,965,376 float64 observations would require 472,392 GiB (461.32 TiB) before overhead.

**Where time goes**

On V100 at n=10/batch32, isolated median calls took 6.228 ms for generated targets plus the rank-one measurement gradient and 8.114 ms for exact TP. With tensor measurements the first figure was 173.962 ms and TP was 8.003 ms. These include dispatch/synchronization and do not add up exactly to fused-scan time. Host noise preparation took about 1.95 ms per 100-step chunk; CPU prefetch was not justified by that measurement.

**Recovery experiments**

All are fresh runs with initialization/sampling/channel/noise seeds zero. Metrics use the same fixed 512-row sample within each qubit size; fidelity and TP use complete factors. The relative-noise controls use sigma=0.05*2**(5-n), preserving the n=5 ratio of noise standard deviation to mean outcome probability. Batch sizes, penalty scales, step counts, and devices differ as shown; these are exploratory configurations, not isolated causal tests or universal optima.

| n | GPU | sigma | Batch | Steps | beta0 | Final fidelity % | TP | Sampled loss | Truth loss on same sample |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | V100 | 0.05 | 32 | 50000 | 640000 | 6.12239e-05 | 1.15716 | 0.001155583 | 0.001134995 |
| 8 | V100 | 0.00625 | 32 | 50000 | 640000 | 0.00281455 | 0.258992 | 3.297706e-05 | 1.77343e-05 |
| 8 | A100 | 0.00625 | 512 | 50000 | 1e+07 | 17.8883 | 0.981267 | 2.32051e-05 | 1.77343e-05 |
| 10 | V100 | 0.05 | 512 | 10000 | 1.024e+07 | 6.91451e-05 | 5.09137 | 0.001163411 | 0.00115646 |
| 10 | V100 | 0.0015625 | 512 | 10000 | 1.024e+07 | 6.97473e-05 | 0.292949 | 2.441254e-06 | 1.129356e-06 |
| 10 | A100 | 0.05 | 512 | 10000 | 1e+09 | 5.65101e-05 | 30.5377 | 0.001159975 | 0.00115646 |
| 10 | A100 | 0.0015625 | 512 | 10000 | 1e+09 | 0.000501607 | 17.1069 | 2.025642e-06 | 1.129356e-06 |

Truth loss is computed from the exact fixed metric-sample noise recipe, up to floating-point rounding. It is not a convergence certificate. Smoothed objectives and gaps are not compared across penalty schedules.

An allocated CPU probe measured U0 gradient norms using a bounded streamed batch. At n=10/batch512, the unscaled TP gradient norm was 42,560; measurement norms were 0.000174 at sigma=.05 and 0.000142 at sigma=.0015625. Thus beta0=10,240,000 made TP approximately 24–29x larger at initialization. The subsequent beta0=1e9 tests reduce that initial ratio to roughly 0.25–0.30. This is a sampled-gradient observation at U0, not a statement about all iterations. Increasing beta0 weakens TP throughout the schedule.

**Interpretation and next experiment**

Generating observations on demand removes the 24**n storage bottleneck. Exact TP remains a substantial compute cost, but neither GPU memory nor target generation prevented n=10 execution. Increasing the GPU speed alone does not resolve the observed recovery failure. Fixed sigma=.05 also becomes much larger relative to the signal as n grows; the smaller-noise controls are distinct experiments and must not be presented as recovery at the original noise level.

The original n=5 runs used 320,000 sampled rows for 1,024 complex factor entries. The n=10 batch512/10k runs use 5.12 million sampled rows for 1,048,576 entries: 64x fewer draws per entry. Repeated samples and optimizer weighting matter, so this ratio is descriptive, not a sample-complexity bound. Matching that ratio would take 640,000 batch512 steps, roughly 10.1 A100 optimizer hours at the measured rate, without guaranteeing recovery. A useful next investigation is convergence versus sample budget, step schedule, and TP schedule, calibrated first on the improving n=8 configuration before a long n=10 run. Repeating short near-zero-fidelity n=10 runs across seeds would not yet validate a promising recovery setting.

![Saved fidelity trajectories for the relative-noise controls](recovery.png)

**Reproducibility and files**

Raw reports are in the adjacent profile directories; complete scalar traces are in traces.json, and normalized tables are in summary.json. WORK_LOG.md records all job IDs, failed/canceled attempts, output locations, and environment handling. The reviewed uncommitted changes are preserved in source.patch. Ruche's original datasets/results, staged scripts/gen_data.py, and .gitignore edits were preserved.
