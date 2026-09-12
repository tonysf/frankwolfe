# Larger-qubit structured QPT experiments

The direct generator creates **new synthetic noisy data** in the compact
format, without first constructing the legacy dense HDF5 arrays. Start with
four and five qubits. This does not resize or replace the three-qubit archive,
and it does not reproduce that archive's particular ground truth or noise.

Run data generation in a CPU Slurm allocation and the benchmark in a GPU
allocation. On Ruche, neither operation belongs on a login node; see the
[Ruche execution policy](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/12_frequently_asked_questions/).
The commands below assume the repository root and an activated environment
with the existing NumPy/JAX dependencies. They create no Slurm files and do
not connect to or submit work on any host by themselves.

## Generate compact noisy data directly

Inside the CPU allocation:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m paper.experiments.qpt_generate_data \
  --n-qubits 4 --noise-std 0.05 --channel-seed 0 --noise-seed 0 \
  --batch-size 1024 \
  --save results/qpt/scaling_data/n4_channel0_noise0.npz

python -m paper.experiments.qpt_generate_data \
  --n-qubits 5 --noise-std 0.05 --channel-seed 0 --noise-seed 0 \
  --batch-size 1024 \
  --save results/qpt/scaling_data/n5_channel0_noise0.npz
```

Existing destinations are rejected, not overwritten. Reuse a completed data
archive for repeated optimizer runs, and choose new output paths when changing
the data configuration. All observations are stored as float64 on the host;
only bounded batches of tensor-product sensing vectors are built during
generation. No global measurement, Pauli, TP, or process-matrix tensor is
created. The truth is stored as one complex128 factor column.

The default guards admit at most 10,000,000 observations and an estimated
512 MiB of generation working storage. They admit these initial four/five-qubit
experiments and reject six qubits before generating arrays. The estimate is a
conservative array budget, **not a measured peak-RSS guarantee**; it excludes
Python/library overhead and should not be used as a scheduler memory request.
The archive is compressed, so its file size differs from uncompressed arrays.

| Qubits | Number of observations | Float64 observation array only |
|---|---:|---:|
| 4 | 331,776 | 2.53 MiB |
| 5 | 7,962,624 | 60.75 MiB |
| 6 | 191,102,976 | 1.42 GiB |

Generation still visits every observation: its work grows with `24**n`, even
though it no longer constructs `24**n` dense matrices. Raising the guards for
larger sizes requires a new memory/runtime plan, not just a larger GPU.

## Scientific model and reproducibility

The model follows the final data-generating formula in the pinned
[QPT_BFW notebook](https://github.com/LeNavil/QPT_BFW/blob/48e9aa80250e8da065734de593afe549f05912ce/qutomo_gt_gen.ipynb):

1. Draw a Haar unitary `H` by QR factorization of a complex Gaussian matrix,
   with the diagonal-phase correction.
2. Expand it in tensor products of `[I, X, -iY, Z] / sqrt(2)`:
   `c[k] = trace(P[k].conj().T @ H)`.
3. Generate the legacy quadratic targets
   `f[s] = vdot(D[s], outer(c, c.conj())).real + noise_std * z[s]`.

The implementation evaluates that last expression without constructing `D[s]`
or `outer(c, c.conj())`. The row order is the original QPT_BFW order, not plain
base-24 order. This preserves the existing model's conjugation convention;
do not replace the quadratic expression with a superficially similar Born
probability calculation for `H` without checking that convention.

The normalized basis makes `||c||_2 = sqrt(2**n)`, and its associated Kraus
matrix is unitary, so the rank-one truth is trace preserving. `tau=10` contains
the truth and the standard initialization for both four and five qubits. The
radius must be reconsidered when scaling further.

Noise is independent additive Gaussian noise, **not shot noise**. It is not
clipped to `[0, 1]`, renormalized, or replaced with noiseless data. The notebook
also contains shot-count code, but its final Gaussian-observation loop
overwrites those counts. `--noise-std 0` deliberately creates a different,
noiseless realization while retaining the stored-observation format.

Channel and noise draws use separately namespaced PCG64 streams: the channel
uses `SeedSequence(channel_seed, spawn_key=(0,))`, and noise uses
`SeedSequence(noise_seed, spawn_key=(1,))`. Thus even the common choice of
zero for both user seeds does not reuse the same random-number sequence.
Repeating a seed
configuration in the same numerical environment reproduces its arrays, and
changing generation batch size does not change the generated arrays. This
does not promise bitwise-identical QR results across NumPy/BLAS versions or
hardware. Metadata record the model, row/basis conventions, seeds, precision,
generation batch size, allocation estimate, library provenance, and numeric
array hashes. Synthetic construction is labeled `verification="constructed"`,
not as a full comparison with an HDF5 file that was never created.

## Structured-only timings and memory measurements

Inside one GPU allocation, with the CPU generation already finished:

```bash
export XLA_FLAGS=--xla_gpu_autotune_level=0

for n in 4 5; do
  python scripts/benchmark_qpt_structured.py \
    --data "results/qpt/scaling_data/n${n}_channel0_noise0.npz" \
    --backends structured \
    --device gpu --precision 64 --measurement-backend tensor \
    --steps 1000 --rank 1 --tau 10 --batch-size 32 --chunk-steps 100 \
    --metrics-every 100 --metric-mode sampled --metric-samples 512 \
    --metric-batch-size 32 --metric-seed 12345 \
    --initialization-seed 0 --sampling-seed 0 --repeats 3 \
    --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
    --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
    --step-scale 2 --step-offset 2 --step-exponent 1 \
    --profile-memory --require-gpu-memory \
    --save "results/qpt/scaling_n${n}_tensor_seed0_001.json" || exit 1
done
```

Use new report paths for another invocation. `--data` means structured-only;
requesting a dense backend with it is rejected. This path never converts to
dense HDF5 and does not assert dense/structured parity. The tensor backend is
explicit so these first scaling tests use the backend examined by the earlier
same-factor diagnostic. Testing `--measurement-backend rank-one` is a separate
performance experiment; record it as such. Autotuning is disabled to retain
the setting that removed cross-process variation in the earlier V100 test;
this does not guarantee bitwise equality on every GPU/software combination.

Each dataset gets three fresh-process timing repeats and a separate memory
profile, with logs and final factors retained. The benchmark reports CPU peak
RSS, JAX allocator counters, and sampled per-process driver VRAM separately.
One memory profile is not a distribution, and sampled VRAM can miss brief
peaks. Missing GPU measurements are an explicit failure with
`--require-gpu-memory`, not a zero-memory result.

Sampled metrics use one fixed independent set of 512 rows per run. They avoid
a full pass over the observations at every checkpoint; measurement loss and
smoothed gap are estimates for that sample, not exact full-data quantities.
TP evaluation remains exact. At different qubit counts even the same metric
seed selects different problems/rows, so compare runtime and memory scaling
separately from reconstruction quality. The generator's compact truth permits
ground-truth fidelity evaluation without a dense process matrix.

Keeping `noise_std=0.05` fixed means equal **absolute** noise, not equal
signal-to-noise ratio across qubit counts. Typical noiseless probabilities
decrease as the outcome count grows. Do not interpret reconstruction-quality
changes across sizes as solely an optimizer effect.

Report actual device identity, precision, backend, seeds, schedules, metric
mode, code revision, compile/update/metric/runner timing, and all memory
definitions with the results. Do not equate old V100 timings with new A100
timings, treat sampled metrics as full metrics, or infer large-n convergence
from a 1,000-step smoke benchmark. First confirm finite completed runs and
memory fit; then plan longer runs and additional channel/noise/optimizer seeds.
