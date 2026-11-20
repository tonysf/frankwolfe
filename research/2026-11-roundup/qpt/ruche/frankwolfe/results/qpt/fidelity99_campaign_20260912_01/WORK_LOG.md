# QPT campaign: reach 99% fidelity at ten qubits

The user's requested target is at least 0.99 factor fidelity at n=10, with
larger batches and more iterations authorized. SSH, edits within the Ruche
repository, submissions and monitoring are authorized. No commit/push/switch/
pull. Preserve earlier datasets/results and the user's remote .gitignore and
staged scripts/gen_data.py. Primary noise remains sigma=.05 unless the user
answers the pending noise clarification differently. Reduced noise is a
separate control, never evidence of success at sigma=.05.

## Implementation and validation

New opt-in `product-state` measurement backend factors the existing local
sensing vectors after their Pauli transform, verifies the factorization, and
uses matrix products with O(B*2**n) workspace instead of O(B*4**n). Predictions
and real-Frobenius gradients are unchanged mathematically; no conjugation or
FRAMES/TP formula was changed. Auto/default backend behavior is unchanged.

The direct structured runner now supports --restart-path, --resume,
--fidelity-target .99, and --prefetch. Restarts retain factor, measurement
momentum, sampling RNG and absolute schedule position. Data/configuration and
schedule prefixes are validated. --steps means total steps, not added steps.
Changing batch/radius/backend/past schedules on resume is rejected. Chunk,
metrics and prefetch can change. Results contain segment-local timing and
scalar schedules, with absolute checkpoint steps. Checkpoints never include
unconsumed prefetched rows. Only one additional host chunk is prepared.

Local v3: 428 passed, 56 optional-dependency skips. Ruche v2: 482 passed,
plus 12 GPU tests. v3 full/GPU checks passed in job 1888457: 484 tests plus 13 GPU checks.
Long jobs depend on its successful exit, including the prefetch memory profile.
Frozen v3 source: results/qpt/fidelity99_prefetch_validation_20260912_01/source
under /gpfs/workdir/silvetian/frankwolfe. Hashes: source_v3_sha256.json.

## Jobs and results

All remote directories below are under the Ruche repo's results/qpt.

- 1887087: fidelity99_n8_control_20260912_01, COMPLETED 32m23s. Old immutable
  rank-one backend, n8 reduced sigma=.00625, B512, beta1e7, gamma2/(k+2),
  500k fresh steps. Final fidelity .7297081536, TP .18339126. 1777.23 optimizer s.
- 1887097: fidelity99_product_profile_20260912_01, COMPLETED 6m28s. v1 backend
  validation/profiling. A later conditional cancellation required PENDING;
  it had begun, so it was allowed to complete. Do not label this canceled.
- 1887112: fidelity99_product_profile_20260912_02, COMPLETED 15m38s, A100 gpu15.
  v2 validation, n10 batch profiles, four n8 pilots. Results below.
- 1888457: fidelity99_prefetch_validation_20260912_01, v3 validation and n10
  B65536 prefetch profile, 15-minute cap. An earlier SSH connection died before
  creating this directory or submitting; absence and scheduler history were
  inspected before submitting the actual job. Do not duplicate it.
- 1888459: fidelity99_n10_fixed_20260912_01. Main sigma=.05 run.
- 1888460: fidelity99_n10_relative_20260912_01. Diagnostic sigma=.0015625 run.

The two n10 jobs depend afterok on 1888457. Both use A100, one GPU, 2 CPUs,
12G RAM, 4-hour caps; product-state, complex128, B65536, chunk10, prefetch,
100k total steps maximum, tau40, rho=2/(k+4)^.6, beta=1e9/(k+1)^.25,
gamma=10/(k+10). Metrics every1000, fixed512 rows, metricbatch512, seed12345.
Initialization/sampling seeds0; existing truth/noise archives reused.
They save latest_restart.npz every checkpoint and final.npz on successful end,
and stop at .99 fidelity. Neither starts from truth or an earlier final factor.

All submissions used fixed new mkdir guards, quoted sbatch heredocs,
export=NONE/propagate=NONE, environment loading in the allocation, preserved
CUDA_VISIBLE_DEVICES, disabled GPU autotuning. Excluded ruche-gpu13 because
earlier CUDA initialization failed there. Anaconda module and qpt environment
are unchanged. Test dependencies are isolated under previous results.

## v2 measurements

A100 n10, 100-step profiles, two timing repeats and a separate memory run:

| Batch | Optimizer seconds | JAX peak MiB | Sampled process VRAM MiB |
|---:|---:|---:|---:|
| 512 | .7665 | 240.3 | 696 |
| 8192 | 1.7269 | 651.2 | 1720 |
| 65536 | about 9.49 | 4347.3 | 8888 |

These are different memory definitions; sampled VRAM can miss peaks.
At B65536 CPU sampling/transfer took 5.04s in addition to 9.49s optimizer,
motivating the opt-in prefetch. Its speedup still needs the v3 measurements.

n8 B8192, beta1e7, gamma=a/(k+a), 30k maximum steps, seeds0:

| sigma | a | Final step | Fidelity | TP |
|---:|---:|---:|---:|---:|
| .05 | 2 | 30000 | .00282159923 | 1.9037408 |
| .05 | 10 | 30000 | .57194503208 | 1.3561853 |
| .00625 | 2 | 30000 | .92238871712 | .17709169 |
| .00625 | 10 | 26000 | .99017278638 | .18074091 |

The reduced-noise 99% result took 47.18 optimizer seconds; elapsed includes
CPU preparation, metrics, compilation and checkpoint I/O. This validates n8
recovery under that control, not n10 or original-noise success. The larger
step schedule is promising for the new n10 runs. The old B512/500k control
only reached 73%; more iterations alone did not close the gap.

## Follow-through

Native thread heartbeat reach-99-qpt-fidelity checks every30 minutes and can
extend promising completed runs within the user's authorization. Keep at most
two long production GPU runs active. Inspect exact IDs and checkpoints before
submitting; preserve rolling checkpoints and earlier segment results. Before
claiming the user's n10 target, independently recompute complete-factor
fidelity in an allocation and report TP and the noise level. Pause the
heartbeat once the clarified target is verified.

Validated v3 is now applied to the main Ruche checkout as well as the frozen
snapshot. All seven affected files match local SHA-256 values. Original
versions are backed up under results/qpt/fidelity99_main_backup_20260912_01.
The expected old file hashes were checked before copying. The staged diff and
.gitignore hash were unchanged afterward. No commit or push occurred.

1888457 COMPLETED with exit0 in 3m54s. Prefetched B65536 timing repeats took
9.505/9.497 optimizer seconds per100 steps, 15.285/15.252 runner seconds.
Sampling/transfer blocking time fell to 1.157/1.074s, while actual host
preparation took 5.094/4.814s and overlapped GPU execution. Metric batch size
also changed from32 to512, so the entire runner-time reduction is not solely
attributed to prefetch. Raw v3 reports are in local validation_reports/.

Latest inspected production checkpoint: both n10 jobs running on gpu15,
9000 steps at 2026-09-12 18:00 UTC. Reduced sigma=.0015625: fidelity
.93926792479, TP1.3809691; fixed sigma=.05: fidelity .0000005890571993,
TP24.3129099. No n10 99% result yet.
Keep following both trajectories; do not silently label the control as
primary success. The pending noise clarification can steer the target.
