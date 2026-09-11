# QPT reproducibility diagnostic

Use this diagnostic when repeated structured runs disagree despite matching
initial-factor and sampled-batch fingerprints. It investigates numerical
agreement; it does not change the optimizer, relax benchmark tolerances, or
establish new speedup or memory claims.

## Run in an existing allocation

Activate the QPT environment and enter the repository in your own GPU job or
allocation. Keep job submission and cluster setup outside Git. This script
does not connect to a cluster or submit a job.

For the saved tensor-backend experiment:

```bash
python -u scripts/diagnose_qpt_reproducibility.py \
    --benchmark-report results/qpt/benchmark_1865396/result.json \
    --output-dir results/qpt/repro_1865396_001 \
    --processes 3 --replays 3 --device gpu
```

Choose a new output directory each time. Retain the benchmark report and its
verified compact-data artifact; the diagnostic reuses them instead of
regenerating observations or reconverting the original HDF5 file.

The benchmark configuration supplies the measurement backend, precision,
seeds, schedules, rank, radius, batch size, step count, and chunk size. For
job `1865396`, that means the tensor backend in 64-bit precision, the full
1,000-step plan, and 100-step chunks. The prepared input bundle freezes the
initial factor, observations, and complete sampling plan for all workers.

The default experiment launches three fresh worker processes. Each compiles
the structured chunk and executes three complete trajectories using that
same compiled executable. Every replay resets the factor to the frozen
initial value and the momentum estimator to zero. Within a trajectory,
chunk state carries forward normally. The comparisons therefore separate
repeated execution of one executable from independent compilations in fresh
processes. At least two replays are required. With only one process, the
cross-process comparison is unavailable rather than passed.

All source chunks must have the same length so each worker can reuse one
compiled shape. A partial final chunk or metric checkpoints that produce
different chunk lengths are rejected; the 1,000-step/100-step source above
is supported.

Before GPU work, the script checks estimated frozen-plan array storage
against `--max-plan-mib` (64 MiB by default), and estimated per-worker
trajectory array storage against `--max-trace-mib` (128 MiB). Exceeding either
guard stops the diagnostic. These estimates do not bound total RAM or VRAM;
device copies, temporary allocations, and compilation need additional memory.

## Read the artifacts

The output directory contains:

```text
report.json        # overall completion and comparison findings
input_bundle.npz   # frozen inputs shared by every worker
input_manifest.json # frozen configuration and input/source fingerprints
worker_*/          # persistent logs, reports, trajectories, and compiler artifacts
```

Workers capture factors and momentum at chunk boundaries, including
initialization, plus the per-step gaps. Fingerprints record the actual shared
inputs and, when supported, compiler IR. A compiler
IR hash is evidence about that representation, not a hash of the final
machine-code binary: equal IR hashes do not prove identical executable code,
and different hashes do not by themselves prove a mathematical difference.

Start with `findings` in the overall report: `inputs_match`, `devices_match`,
the `same_executable_exact`/`same_executable_allclose` pair, the corresponding
`cross_process_exact`/`cross_process_allclose` pair, and
`cpu_reference_status`. Exact equality and tolerance-based agreement are
reported separately. Inspect `workers` and `across_process_comparisons` for
the detailed evidence; `frozen_inputs` records shared-input fingerprints.

After replay execution, selected identical factors are used for measurement
and penalty gradient probes against a CPU dense-operator reference. The
probe points include initialization and selected chunk boundaries. The
reference has a default cap of 128 MiB of estimated dense-reference storage; a
skipped or unavailable reference is explicitly incomplete evidence, not a
passing comparison. This cap is not a bound on total worker RAM or GPU memory.

A diagnostic can complete successfully while finding discrepancies.
`report.json` distinguishes completed diagnostics with findings from worker
or coordinator execution errors. Inspect the findings, not only the exit
status. For a failed worker, retain its logs and the partial overall report.

## Interpret the comparisons

| Observation | What it establishes | What remains unresolved |
| --- | --- | --- |
| Identical inputs produce different chunk states when replaying one compiled executable | Variation occurs during repeated execution or state handling | Which operation causes it, and whether the source is a kernel or a diagnostic/implementation defect |
| Each process repeats consistently, but fresh processes disagree | The variation is associated with something that differs between processes, potentially compilation or runtime setup | Compilation is not proven to be the cause merely because it occurred separately |
| Gradients disagree at the same factor | There is a local numerical or implementation discrepancy to inspect before interpreting long trajectories | Its source and significance; compare absolute and relative errors |
| Gradient probes agree closely, but trajectories separate later | Agreement holds at the tested points; accumulated sensitivity is a plausible explanation | The probes do not prove every step correct or establish roundoff as the sole cause |

Chunk-boundary trajectories locate the earliest **recorded** discrepancy;
they do not identify the first divergent update within a chunk. Factor
differences alone can also reflect phase conventions, so examine gradient,
momentum, and gap evidence together rather than treating one array norm as a
complete correctness verdict.

The frozen measurement bank also controls differences from host-side
rank-one eigendecomposition, when that backend is selected. The diagnostic
changes memory layout by keeping the complete plan and collecting chunk
states. Operator probes are separately compiled after the replays, not
extractions of the fused scan executable. Therefore, a stable diagnostic
does not rule out variability in the original benchmark, and probe agreement
does not prove the fused scan uses identical numerical operations.

Do not weaken tolerances or publish revised performance comparisons merely
because a discrepancy is small at one probe. Keep the original benchmark
results as provisional until its agreement failure is understood.

For a local smoke check, use a small local benchmark report with
`--device cpu`. CPU execution can test input freezing, replay comparisons,
and report generation; it does not validate V100 GPU reproducibility.
