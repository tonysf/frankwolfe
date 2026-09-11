# QPT full-data, same-factor diagnostic

Use this check when dense and structured optimization runs finish at
different factors, even after their repeated execution has stabilized. It
asks whether the implementations evaluate the same function at the same
factor, separately from whether they followed the same numerical trajectory.

This is a CPU-only diagnostic of saved results. It does not import JAX, use a
GPU, rerun optimization, reconvert HDF5 data, change the optimizer, or relax
the original benchmark's parity test.

## Run in a CPU allocation

On Ruche, run the full-data check inside a CPU Slurm allocation, not on
`ruche01` or `ruche02`. The documented `cpu_short` partition supports CPU jobs
up to one hour. No GPU allocation is needed. Keep NumPy/BLAS threads within
the allocated CPU count and retain output on the work filesystem, not a CPU
node's temporary directory. Manage submission separately from Git.
[Ruche partitions](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/07_slurm_partitions_description/),
[Ruche job management](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/).

With the QPT environment active and the repository as working directory
**inside the allocation**, evaluate the saved tensor benchmark:

```bash
python -u scripts/diagnose_qpt_same_factor.py \
    --benchmark-report results/qpt/benchmark_tensor_noauto_1883052/result.json \
    --output-dir results/qpt/same_factor_1883052_001 \
    --batch-size 32 --rtol 1e-10 --atol 1e-12 \
    --max-reference-mib 128
```

Choose a new output directory each time; existing output is not overwritten.
The source report, compact data, and saved factors are read-only and
fingerprinted. A source benchmark may have failed its parity test: that is
the case this diagnostic is intended to investigate, not an instruction to
discard its results.

The source must identify fully verified compact data with stored
observations, 64-bit precision, and full-data metrics. All required timing
runs must have completed successfully; a failed worker is not usable merely
because the overall benchmark saved a report. The original HDF5 file is not
needed. The dense reference
reconstructs matrices from the verified compact operator bank; this is not an
independent repetition of the earlier HDF5 conversion verification.

For a copied result bundle, retain the report plus `data.npz` and all saved
timing-run factor files, preserving their
`timing_BACKEND_REPEAT.npy` names. Use `--artifacts-dir` to identify the
actual copied artifact directory explicitly. The diagnostic must not silently
substitute an unrelated nearby data file for a missing recorded path.

## What is compared

1. Every saved timing repetition after the first is compared with the first
   repetition of its backend. Every matched dense/structured pair is compared
   using raw factors, gauge-aligned factors, and process matrices `U U†`. For rank one,
   gauge alignment removes a physically irrelevant global phase.
2. Every distinct saved factor is evaluated on **all** stored observations
   using both explicit dense Kronecker measurement matrices and the existing
   structured NumPy operators. Measurement matrices are streamed in batches;
   the full dense dataset is not materialized. The trace-preservation
   reference uses the global dense penalty representation.
3. Both evaluations include full measurement loss and gradient,
   trace-preservation residual and gradient, the linear minimization result,
   and the full smoothed gap. They use the benchmark's final smoothing value
   at index `steps - 1`. The gap uses the full gradient, not the optimizer's
   stochastic momentum estimate.
4. Saved final loss and gap scalars are checked against both common CPU
   evaluations at the corresponding saved factor. Cross-evaluation at dense
   and structured final factors distinguishes a difference in the functions
   from a difference in the points where they were measured.

The comparison tolerances describe this new diagnostic; they do not change
the source benchmark or turn its failed parity test into a pass. CPU
evaluation is not a GPU-kernel replay and does not establish bitwise GPU
agreement.

## Bounds and interpretation

The default guards are `--max-measurements 1000000`, `--max-input-mib 128`,
and `--max-reference-mib 128`: at most one million measurements, 128 MiB of
input-file storage plus NPZ members' declared uncompressed sizes, and 128 MiB
of estimated dense-reference storage. The reference bound depends on
dimensions and batch size; smaller batches reduce measurement-matrix working
storage but not the global penalty
representation. These are workload/allocation estimates, not a guarantee of
total process RAM. Exceeding a guard or missing required data is an explicit
diagnostic failure, never a passing or silently sampled full-data check.

Inspect `report.json` for completion, provenance, comparisons, and findings:

- `source` and `settings` record input fingerprints, inherited verification,
  the effective backend, and terminal smoothing value.
- `evaluations` contains each distinct factor's full-data comparisons and
  links to `evaluation_*.npz` arrays, named `dense__METRIC` and
  `structured__METRIC`. Identical saved factors share one evaluation.
- `factors` retains every timing run and its saved-metric versus CPU checks;
  `repeatability` compares each backend's repetitions with its first.
- `cross_backend` contains matched-pair geometry and
  `common_evaluator_metrics`; `findings` summarizes the individual checks.

A completed diagnostic can report disagreements; distinguish execution
status from numerical findings. Preserve the partial report if execution
fails. Read per-metric comparisons rather than only the aggregate boolean:
linear minimization atoms can be nonunique, especially at higher rank, so an
atom-only difference is not equivalent to a loss, gradient, or gap mismatch.

If the two formulations agree at each identical factor, while final factors
and their common loss/gap values differ, the evidence supports a trajectory
difference rather than disagreement between the tested CPU formulations.
If common CPU values disagree with the saved GPU-run scalars at the same
factor, investigate the original metric calculation or GPU numerics. If the
CPU formulations themselves disagree, inspect that local algebra/data
discrepancy before interpreting performance.

Close same-factor agreement is not proof of identical optimization paths,
global correctness, or acceptable reconstruction quality. Process-matrix
and objective differences provide more meaningful context than a raw factor
distance alone; any application-level acceptance threshold needs its own
justification. Do not loosen tolerances or publish revised speedup claims
solely to make the report pass.

Small local CPU tests exercise input handling, full-data aggregation, and
report generation. They do not certify GPU behavior or Ruche performance.
