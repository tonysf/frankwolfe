# On-demand QPT experiment record — 2026-09-12

Base commit: ba85bff73c306ee5f12b97aa3a24910851f585ae, branch stochastic-frames.
No commit or push. Existing local results and remote staged scripts/gen_data.py
and .gitignore edits preserved. The final source patch has been applied to the
main Ruche checkout; experiments use an immutable v4 snapshot under results.

## Implementation

Explicit synthetic-noisy mode: compact Haar truth and local operators, fixed
row-indexed SplitMix64/Box–Muller Gaussian noise on CPU, truth targets evaluated
on GPU inside each bounded optimizer step. This is a new noise realization;
stored datasets and optimizer/default schedules are unchanged. The existing
rank-one measurement backend is selected explicitly in performance/recovery
runs. TP remains exact. Source patch: source.patch.

## Validation

Local: 414 passed, 56 skipped (optional dependencies unavailable).
Ruche v4: 470 passed; 7 on-demand checks passed with GPU trajectory tests on V100.
A100 gpu12: 7 on-demand checks passed with GPU trajectory tests.

## Jobs and directories under /gpfs/workdir/silvetian/frankwolfe/results/qpt

- 1886824, on_demand_validation_20260912_01: failed before tests (pytest absent).
- 1886881, on_demand_profile_20260912_02: 458 passed, 12 failed; snapshot omitted
  shell scripts, and legacy tests used removed JAX experimental.enable_x64.
  Offline pytest wheels were installed only into this job's test_deps directory.
- 1886896, on_demand_profile_20260912_03: canceled while queued (zero elapsed)
  after catching an error in the compatibility shim. Corrected and retested.
- 1886898, on_demand_profile_20260912_04: COMPLETED, 17m39s, V100 gpu04.
  Full validation; n=6,8,10 x tensor/rank-one; 2 timing repeats + separate memory
  repeat each; 1,000 steps, batch32, beta0=10000, tau=10/20/40; component timings.
- 1886904, on_demand_a100_20260912_01: failed GPU initialization on gpu13.
- 1886907, on_demand_a100_probe_20260912_01: gpu13 CUDA_ERROR_UNKNOWN directly
  and through srun. Preserved Slurm device visibility; no driver/env changes.
- 1886910, on_demand_a100_20260912_02: COMPLETED, 7m14s, A100 gpu12.
  Same n10 data/settings as V100 profile. Excluded gpu13; CUDA init succeeded.
- 1886905, on_demand_recovery_n8_20260912_01: COMPLETED, 2m18s, V100 gpu04.
  n8 rank-one, 50k steps, batch32, beta0=640000, tau20. Fixed sigma=.05
  versus relative sigma=.00625, same truth and standardized row noise.
- 1886912, on_demand_n10_batches_20260912_01: COMPLETED, 49m10s, V100 gpu04.
  V100, batch128/512 profiles (1k steps, 2 timings + memory each), then n10
  batch512 10k-step fixed/relative-noise runs. beta0=10240000, tau40.
  Relative sigma=.0015625. One-hour cap.
- 1886915, on_demand_gradient_probe_20260912_01: GPU probe canceled while queued;
  diagnostic moved to CPU to avoid waiting for a GPU.
- 1886916, on_demand_gradient_cpu_20260912_01: COMPLETED, 55s, node001.
  Initial TP/measurement gradient norms at n5/6/8/10, batch32/512, both noise
  conditions; streamed batches of 32 on 2 CPUs, 8G RAM.
- 1886938, on_demand_weaker_tp_20260912_01: COMPLETED, 24m11s, A100 gpu12.
  A100 excluding gpu13; n8 relative noise, batch512, 50k steps, beta0=1e7;
  n10 both noise conditions, batch512, 10k steps, beta0=1e9. 45-minute cap.

All submissions used fixed new output directories (mkdir without -p), quoted
sbatch heredocs, export=NONE/propagate=NONE, and environment loading inside jobs.
Autotuning disabled throughout. CUDA_VISIBLE_DEVICES preserved.

## Final conclusions

The 24**n observation-storage wall is removed. n10 truth generation: 3.716s;
zero observation-array bytes; truth factor 16MiB. n10 rank-one optimizer:
V100 14.170s/1k, A100 6.127s/1k. V100 tensor: 181.206s/1k, A100 75.147s/1k.
V100 batch32 rank-one allocator peak: 1168.326MiB; sampled driver VRAM 2508MiB;
worker peak RSS 877.145MiB. These are distinct memory definitions.

Initial gradient probe at n10/batch512: unscaled TP norm 42560; measurement
norm .000174 (fixed noise) or .000142 (relative). beta0=10240000 therefore
makes TP about 24–29x larger at U0. This is one sampled gradient at U0, not
an objective-ratio argument or a claim about the entire trajectory. The weaker
penalty comparison uses beta0=1e9. Hardware/batch changes must be disclosed when
comparing recovery; execution capability is separate from successful recovery.

Final weaker-TP recovery: n8 relative noise, batch512/50k, fidelity
0.1788826596 (17.8883%), TP 0.9812671. n10 relative noise, batch512/10k,
fidelity 0.00000501607238 (0.000501607%), TP 17.1069153. The n10 fixed-noise
counterpart has fidelity 0.000000565101123 and TP 30.5377092. High-fidelity
n10 reconstruction has not been achieved. These exploratory comparisons also
change batch/penalty/device, and do not isolate a causal effect.

n10 V100 batch128: 32.6012s/1k, JAX peak 4194.9MiB, sampled driver VRAM
10700MiB. Batch512: 115.9052s/1k, JAX peak 16491.5MiB, sampled driver VRAM
24702MiB. A100 batch512 recovery: 567.576s/10k optimizer, about 585s runner.

Final scalar collector saved 17 completed trajectories to remote
results/qpt/on_demand_traces_final_20260912.json and local traces.json.
Raw profile/recovery reports and the gradient probe were copied locally;
no final factors or large archives were downloaded. analyze.py produces
summary.json, report.md, and the visually inspected recovery.png.

Final squeue query for all twelve job IDs returned no entries. sacct confirmed
the last two jobs completed with exit code 0. Remote HEAD remains ba85bff on
stochastic-frames. No commit, push, Git switch, or pull occurred.

SHA-256 values of all twelve changed/new source, test, and documentation files
match between the local and remote main checkouts (local_source_hashes.json).
The report bundle is also saved on Ruche under
results/qpt/on_demand_report_20260912_01.
