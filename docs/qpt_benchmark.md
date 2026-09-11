# Measured QPT comparison

This benchmark compares the original dense runner and the structured runner
on the same GPU and original noisy HDF5 data. It records timing separately
from memory profiling and never connects to a cluster or submits work.

## Run the benchmark

Run from the repository root in an existing GPU allocation with the QPT
environment active. Allocation and environment setup are managed separately.
Choose a new output path for each comparison:

```bash
python -u scripts/benchmark_qpt_structured.py \
    --h5 ../QPT_BFW/jax_arr_gt_xi_0.05.h5 \
    --device gpu --precision 64 \
    --backends dense structured --repeats 3 \
    --steps 1000 --rank 1 --tau 10 --batch-size 32 \
    --chunk-steps 100 --metrics-every 0 \
    --metric-mode full --metric-batch-size 32 \
    --initialization-seed 0 --sampling-seed 0 \
    --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
    --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
    --step-scale 2 --step-offset 2 --step-exponent 1 \
    --allocator grow --profile-memory --memory-poll-ms 100 \
    --require-gpu-memory \
    --artifacts-dir results/qpt/benchmark_001/artifacts \
    --save results/qpt/benchmark_001/result.json
```

Use the recorded device kind to confirm the actual hardware. Do not combine
V100 and A100 samples in the same timing comparison.

The HDF5 file is read, not modified. The benchmark performs one full streamed
structure verification and conversion per invocation and reports conversion
time separately. Both paths use the original noisy targets, initialization and
sampling seeds zero, rank one, radius ten, batch size 32, and your previous
power schedules. The command runs 1,000 steps with three fresh-process
timing repetitions per backend, followed by a separate memory-profile run of
each backend. Timing order alternates between repetitions; persistent JAX
compilation caching is disabled for both paths. The structured chunk size is
100. Full-data metrics are evaluated at the initial and final checkpoints
only (`--metrics-every 0`),
reducing diagnostic overhead while retaining comparable final loss and gap.

## Find the report and logs

The example command creates:

```text
results/qpt/benchmark_001/
  result.json    # measured comparison, estimates, and interpretation
  artifacts/    # compact data and per-worker logs, arrays, and reports
```

The benchmark refuses to overwrite an existing report or artifact directory.
Use a different output path for each comparison. On a worker failure,
inspect the partial report and persistent worker logs; do not
interpret an incomplete comparison as a successful speedup measurement.
An abrupt scheduler kill, node failure, or `SIGKILL` can interrupt output;
check the scheduler's accounting as well.

Use `result.json` to interpret the comparison, plus the per-worker logs under
`artifacts/` when diagnosing a failure. There is no need to copy the original
HDF5 file or large worker arrays to interpret the summary.

The main JSON fields to inspect are:

- `status` should be `"complete"`, and `parity_passed` should be `true` for a
  successful two-backend comparison. `parity` contains a final-factor and
  full-loss/full-gap check for every successful timing pair.
- `runs` contains every worker result with its `backend`, `purpose`, `status`,
  `device_kind`, `phases`, `memory`, and persistent log paths. `purpose` is
  `"timing"` or `"memory"`; `memory_profiles` also lists the latter explicitly.
- `medians.dense` and `medians.structured` aggregate successful **timing** runs
  only. Ratios are `dense_over_structured_optimizer_ratio`,
  `dense_over_structured_runner_wall_ratio`, and
  `dense_over_structured_process_wall_ratio`.
- `memory_ratios` reports separate dense/structured CPU RSS, JAX allocator
  high-water, and sampled process-VRAM ratios when the corresponding
  measurements exist for both profiles. These are one-profile comparisons,
  not repeated-run memory medians.

## What the measurements mean

The JSON keeps different memory definitions separate. They must not be
treated as interchangeable:

| Measurement | What it establishes | Limitation |
| --- | --- | --- |
| `memory.cpu_process_peak_rss_bytes` in each worker | Peak host RAM accounted by the worker OS | Not GPU memory; includes imports, data, compilation, diagnostics, and factor saving |
| `memory.jax_allocator.stats.peak_bytes_in_use`, when supported | High-water allocation tracked by the selected device's allocator | Not necessarily total process VRAM or allocator reservation |
| `memory.external_sampler.gpu.sampled_peak_process_bytes` | Largest observed worker VRAM use from `nvidia-smi`, including GPU context and reservation | A sampled lower bound on peak; short spikes between polls can be missed |
| Top-level `memory_estimates` and structured worker `compiled_memory_estimate_bytes` | Storage implied by shapes/dtypes or compiler analysis | Estimates, not measured runtime peaks |

The top-level `converter_parent_peak_rss_bytes` is separate: it is the
coordinator process's **lifetime host-RAM high-water through conversion and
compact-file saving**, including its earlier imports. It is not a
conversion-only increment, a worker peak, or an aggregate job peak. Neither
adding process high-water marks nor taking their maximum reconstructs the
simultaneous host-RAM peak of the complete allocation. The optional Linux
`memory.external_sampler.cpu` entries additionally record sampled worker RSS
and `/proc` high-water observations, useful if a worker exits abruptly before
it can save its own OS counter.

Unsupported counters and unavailable samples are explicitly unavailable, not
zero. `--require-gpu-memory` makes the job fail if no usable device allocator
peak or attributable per-process GPU-memory sample is obtained, instead of
silently producing another host-RAM-only benchmark. Worker artifacts remain
available for diagnosing that failure. A cluster may restrict `nvidia-smi`
process visibility or expose different allocator counters; this strict check
makes that visible before drawing conclusions.

Memory profiling runs in its own fresh processes and is excluded from the
three-run timing medians. Polling every 100 ms can be noticeable for such a
fast optimizer. The memory run therefore provides a whole-worker memory
measurement (including loading, compilation, warmup, and metrics), not a
claim that every reported allocation was needed by a single optimizer step.
The JAX high-water counter, when available, can catch allocations that the
external polling misses; it still has a different scope from total VRAM.
Check `memory.external_sampler.sampling.actual_start_gap` and
`gpu_query_latency` for the observed cadence: 100 ms is the requested interval,
not a guaranteed sampling frequency. The GPU entry also reports valid sample
counts, failed queries, missing PID matches, and partial observations.

Both paths explicitly use `--allocator grow`, with JAX GPU preallocation
disabled and its default BFC allocator growing as needed. This is intentional:
JAX's usual large startup reservation can obscure differences in required
memory. It also means these timings and observed VRAM use are from a
different allocator configuration than an earlier unconfigured run. Compare
the two backends **within this job**, not directly against the earlier
1.421-second structured result. See
[JAX GPU-memory allocation documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
for the distinction between preallocation and the platform allocator.

For time, use the following scopes:

| Report fields | Scope |
| --- | --- |
| Top-level `conversion_seconds`, `compact_save_seconds` | Once-per-job verified conversion and compact-data serialization, separately |
| Worker `phases.python_import_seconds`, `device_initialization_seconds`, `data_load_seconds` | Timed scientific-module imports, selected-device startup, and backend-specific data loading |
| Worker `optimizer_seconds`, `optimizer_compile_seconds`, `metric_seconds` | Synchronized updates, explicit optimizer compilation, and diagnostics; optimizer timing excludes warmup and structured sampling/transfers |
| Structured worker `phases.setup_seconds`, `initial_transfer_seconds`, `warmup_seconds`, `sampling_transfer_seconds` | Runner configuration, initial device data, first-chunk warmup, and each chunk's sampling/transfer |
| Dense worker `phases.runner_setup_seconds`, `scan_setup_and_warmup_seconds`, `post_optimizer_transfer_and_preparation_seconds` | Runner setup including initial transfers; scan setup plus native full-run warmup; host history/checkpoint preparation after updates |
| Worker `runner_wall_seconds` | Runner call including setup, warmup, compilation, updates, transfer, and diagnostics; excludes data loading |
| Worker `process_wall_seconds` | Parent-measured fresh-process span, including interpreter startup, imports, device initialization, loading, running, factor/report saving, and shutdown |
| Top-level `benchmark_wall_seconds` | Coordinator's measured script span, including conversion and all timing/profile runs |

Worker `phases.factor_save_seconds` records the common final-factor save.
`worker_wall_seconds` is the narrower in-worker measured span;
`worker_unattributed_seconds` records time within that span outside the named
phases. `runner_other_seconds` for structured workers and
`runner_finalize_seconds` for dense workers account for runner bookkeeping.
No plot is generated in this comparison. Environment activation and scheduler
startup are outside the Python timings and must be measured separately if
needed. Do not add nested wall times together or compare a timing run's
process wall time with a
monitored profile run's process wall time.

Full metrics use the same observations, but the diagnostic implementations
differ: dense diagnostics run on the CPU and include a process-matrix
eigendecomposition; structured diagnostics run streamed batches on the GPU
without that eigendecomposition. The legacy dense warmup is a complete scan,
whereas structured warmup is one chunk. Both are excluded from optimizer
timings and included in their runner wall times. These implementation
differences should accompany any end-to-end speedup claim.

First check device identity, successful completion, and final-factor/full-loss/
full-gap agreement. Then compare repeated timing medians and the **same memory
metric** from the two profile runs. The benchmark has no established V100 or
A100 speedup until it has actually run on that hardware. If the dense backend
runs out of memory, its failure is a capacity result, not a finite measured
speedup. A later `--backends structured` experiment can characterize the new
path alone but cannot supply a measured dense baseline.

If repeated runs disagree despite matching seeds, use the
[QPT reproducibility diagnostic](qpt_reproducibility.md) to separate
same-executable replay, fresh-process compilation, and fixed-input operator
agreement before changing tolerances or interpreting long trajectories.
