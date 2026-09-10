# Structured QPT on an A100

The structured runner keeps the same factor-space stochastic-FRAMES updates
and the same noisy observations as the original QPT experiment. It replaces
the dense measurement and trace-preservation tensors with local operators,
keeps observations in host RAM, and transfers one chunk of selected targets to
the GPU. The scan retains the current factor and momentum estimate; it emits
scalar gaps rather than a factor at every iteration.

These commands run from the repository root in your existing Ruche GPU
allocation and Python environment. If that environment already ran the dense
experiment, it has the required JAX GPU installation. See [A100 setup](qpt_a100.md)
for a fresh environment. The implementation uses one visible GPU; requesting
four GPUs does not distribute this runner.

## Convert the existing HDF5 once

```bash
python -m paper.experiments.qpt_structured_data \
  --h5 /path/on/ruche/jax_arr_gt_xi_0.05.h5 \
  --save /path/on/ruche/qpt_compact.npz \
  --chunk-size 32
```

Conversion streams the measurement tensor in chunks and validates every row
against the tensor-product model. It also validates the complete Pauli basis.
The compact archive preserves every value of `f_jax_vector`, including the
original noise realization. If `B_jax_tensors` is present, conversion also
validates every block against the structured TP map using bounded slices.
It never reads the dense ground-truth process matrix or modifies the source file.
Full verification needs one pass over the existing dense measurements; run it
once on fast scratch storage. It can run on a CPU node.

The model is specific to the [QPT_BFW generator](https://github.com/LeNavil/QPT_BFW/blob/48e9aa80250e8da065734de593afe549f05912ce/qutomo_gt_gen.ipynb): its local basis is
`[I, X, -iY, Z] / sqrt(2)`, and its input/axis digit order differs from its
outcome digit order. A simple base-24 interpretation of the original row
number would pair the wrong operators with the observations. Conversion
rejects incompatible operators or bases. An explicitly requested
`--verification sampled --verification-samples 256` checks fewer measurement
rows and records incomplete verification in the archive; full verification
is the default. For an archive stored in single precision, explicitly pass
`--rtol 1e-6 --atol 1e-7` to allow its operator-rounding error; the default
tolerances remain `1e-10` and `1e-12`. The archive records the chosen tolerances.
The benchmark command below accepts the same tolerance flags.

## Run the compact experiment

This example specifies the schedules explicitly; match your earlier run's
schedule flags when comparing trajectories:

```bash
python -m paper.experiments.quantum_process_tomography_structured_jax \
  --data /path/on/ruche/qpt_compact.npz \
  --device gpu --precision 64 \
  --steps 10000 --rank 1 --tau 10 --batch-size 32 \
  --chunk-steps 100 --metrics-every 100 \
  --metric-mode sampled --metric-samples 512 --metric-batch-size 32 \
  --initialization-seed 0 --sampling-seed 0 --metric-seed 12345 \
  --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
  --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
  --step-scale 2 --step-offset 2 --step-exponent 1 \
  --save results/qpt_structured.npz \
  --plot results/qpt_structured.png --quiet
```

`--metric-mode sampled` uses one independent, fixed sample at every checkpoint.
Its measurement loss and smoothed gap are estimates for that fixed empirical
sample; the reported gap is not an exact full-data gap. The TP violation and
TP gradient remain exact. Use `--metric-mode full` for a streamed pass over
all original observations at every checkpoint. Full metrics consume bounded
GPU memory but still cost time proportional to the complete measurement set.
Changing metric mode, metric seed, metric batch size, or chunk size does not
change the optimizer's sampling stream.

Compilation is always outside the synchronized optimizer timer. There is no
extra optimizer warmup by default; `--warmup` executes one additional chunk
before timing. This differs from the dense scan runner's complete-run warmup.
`optimizer_seconds` excludes metrics, sampling and transfer; JSON metadata
also records `compile_seconds`, `metric_seconds`, `sampling_transfer_seconds`
and `wall_seconds`. Metric time includes its first JIT compilation. Each
distinct chunk length may require an optimizer compilation.

The result NPZ contains the final complex factor, final momentum estimate,
scalar traces, fixed metric symbols, realized schedules, seeds, device and
library versions, observation mode and verification metadata. It does not
materialize the dense process matrix. Checkpoint factors are disabled by
default; `--store-checkpoints` stores only the requested metric checkpoints
and increases host memory and archive size accordingly. Both data and result
archives load with `allow_pickle=False`.

For a custom Python `x0`, metadata record that it was supplied, its dtype and
a SHA-256 hash, and set the initialization seed to `None`. Keep that `x0` or
enable `store_checkpoints=True` to retain the initial factor for replay; its
hash alone cannot reconstruct it.

```python
from paper.experiments.quantum_process_tomography_structured_jax import load_structured_result

result = load_structured_result("results/qpt_structured.npz")
print(result.metadata["optimizer_seconds"])
print(result.metadata["measurement_backend"])
print(result.final_factor.shape)
```

Converted data intentionally omit the dense `Chi_star_tensor`. Fidelity is
therefore unavailable (`NaN`) unless a compact truth factor is supplied when
constructing `StructuredQPTData` through Python. This does not affect the
optimizer, measurement loss, TP violation, or smoothed gap.

## What changed computationally

Write `d = 2**n` and `N = d**2`. The original measurements occupy
`24**n * N**2` complex numbers, and its trace tensor occupies `N**2 * d**2`.
In complex128 the three-qubit measurement tensor is 864 MiB; the four-qubit
one would be 324 GiB. The local 24-by-4-by-4 bank takes 6 KiB.

The default `--measurement-backend auto` checks whether all local measurements
are positive semidefinite and rank one. For the QPT_BFW bank they are, so the
runner stores 24 vectors of length four and uses
`D_s = h_s h_s.conj().T` to evaluate sensing values and gradients in
`O(batch_size * N * rank)` arithmetic. The selected vectors `h_s` are formed
from local factors in bounded batches. `--measurement-backend tensor` uses
the general axis-by-axis Kronecker application, costing
`O(batch_size * n * N * rank)` and supporting Hermitian local banks beyond
rank one. `--measurement-backend rank-one` requires the rank-one check to pass.

For each factor column, a tensor-product basis transform builds a `d`-by-`d`
Kraus matrix. Exact TP evaluation and its gradient reuse those matrices. They
cost `O(n * N * rank + rank * d**3)` and require no global Pauli-product tensor
or dense `N`-by-`N` process matrix. The code preserves the original objective's
Moreau term `||T(UU†) - I||_F**2 / (2*beta)`, including the factor one half.
Only sampled measurement gradients enter momentum; the exact TP gradient is
added after the momentum update. Sampling TP would change that algorithm and
is not enabled here.

GPU working arrays scale with the factor, a bounded measurement batch, and
the `d`-by-`d` TP matrices. Host arrays still include all stored observations
(`O(24**n)`) and scalar traces/schedules (`O(steps)`). This removes the dense
operator wall, but does not make an existing noisy experiment independent of
the total number of observations. The Python API offers an explicit
`observation_mode="noiseless"` with a compact `truth_factor` for new synthetic
experiments; those targets are generated on demand and are a different data
model. Existing noisy targets are never replaced with noiseless ground truth.

At larger qubit counts, choose a feasible initial point and radius. The
default initializer has Frobenius norm `sqrt(d)`; choosing
`tau >= sqrt(2**n)` is sufficient, whereas `tau=10` need not contain it. The
Python API accepts a custom packed-real feasible `x0`. Reducing batch size or
metric batch size can reduce temporary GPU storage; reducing `chunk_steps`
reduces the transferred sampling chunk while increasing dispatch frequency.

## Benchmark on Ruche

For measured CPU/GPU memory profiling, follow
[Measured QPT comparison](qpt_benchmark.md). It separates unmonitored timing
runs from memory profiles and preserves logs and partial failure reports.

Run this on the same allocated A100 with the original HDF5 file. It validates
and converts that file once, then runs the old and new paths in separate fresh
processes with identical initialization seeds, sampling seeds and schedules.
The default `full` metric mode allows direct comparison of final losses and
gaps. Start with a short run:

```bash
python scripts/benchmark_qpt_structured.py \
  --h5 /path/on/ruche/jax_arr_gt_xi_0.05.h5 \
  --device gpu --precision 64 \
  --steps 1000 --rank 1 --tau 10 --batch-size 32 \
  --chunk-steps 100 --metrics-every 100 \
  --metric-mode full --metric-batch-size 32 --repeats 3 \
  --initialization-seed 0 --sampling-seed 0 \
  --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
  --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
  --step-scale 2 --step-offset 2 --step-exponent 1 \
  --save results/qpt_dense_vs_structured.json
```

The JSON report includes synchronized optimizer timings, optimizer compilation,
metric timings, fresh-process wall time, repeated-run medians, final-factor
discrepancy, and full-metric discrepancies. Dense metrics run on the CPU and
include an eigendecomposition; structured metrics run in streamed batches on
the selected device and omit that dense eigendecomposition. Their metric
times describe these complete diagnostic implementations, not identical
kernels. Use `--metric-mode sampled --metric-samples 512` to measure the new
fixed-sample diagnostic workflow; those loss/gap values are then not directly
comparable to the dense full-data metrics.

Compare both `dense_over_structured_optimizer_ratio` and
`dense_over_structured_runner_wall_ratio`. The former isolates synchronized
device execution and excludes chunk sampling/transfers; the latter includes
compilation, warmup, transfers and diagnostics after data loading. The report
also separates data-loading time and complete fresh-process wall time.

The `memory_estimates` entries describe array sizes, not runtime peaks.
Add `--profile-memory` to run a separate monitored worker for each backend;
its OS RSS, JAX allocator counters, and sampled process VRAM are reported in
`memory_profiles`. Use `--require-gpu-memory` to flag unavailable GPU
measurements. The default `--allocator grow` disables JAX preallocation for
both backends; this differs from earlier unconfigured runs. A CPU
smoke benchmark validates the command and parity, but does not establish A100
speedup. Once the dense baseline no longer fits, `--backends structured` runs
only the compact path while still reporting dense array-size estimates.
The benchmark retains a sibling artifact directory and refuses to overwrite
an existing report or artifact directory; use a new output path for each job.

For regression checks:

```bash
python -m pytest tests/test_qpt_structured_data.py \
  tests/test_qpt_structured_operators.py tests/test_qpt_structured_jax.py -q
```

The tests compare the complete small-run trajectories, momentum estimates,
and full metrics against dense JAX for one/two qubits and rank one/two, as
well as fixed-sample noisy metrics, chunk independence, scalar-only scan
outputs, portable persistence and command-line execution.
