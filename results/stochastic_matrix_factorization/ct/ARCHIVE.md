# CT experiment evidence

This archive preserves the CT experiment files from
`/home/tony/frames_ct_2detect_20261104` on Elite. Copied files retain their original
bytes and paths relative to that directory. Absolute machine paths inside the
files are historical provenance, not paths expected on another machine.

The current experiment uses nonnegative factors inside a product of **spectral
norm balls**. The original root-level experiment and `lmo_initialization/` used
Frobenius norm balls and are obsolete pilot implementations. Later stages also
contain superseded baseline parameters and comparisons; their presence records
the experiment's history, not additional current results. No experiment removing
the spectral constraints has been run or included here.

The current result is
[`gradient_comparison_summary.json`](spectral_balls/expanded/gradient_budget/report_stage_k96ldd86/gradient_comparison_summary.json).
PALM and SPRING–SARAH were continued without retuning to the same 69,120
gradient-view budget per direction as ProxSARAH. Curvature views remain recorded
separately, and the original comparison charging both costs is retained. The
[`gradient_budget/README.md`](spectral_balls/expanded/gradient_budget/README.md)
documents this final stage. The published PDF and its portable LaTeX package are
in [`paper/experiments/stochastic_ct`](../../../paper/experiments/stochastic_ct/).

## Contents and verification

- [`archive_manifest.json`](archive_manifest.json) records every copied file's
  original size and SHA-256, plus every compressed archive and member. Originals
  were checked again after export. All frozen source hashes referenced by the
  retained run metadata have corresponding source files in this archive.
- [`reported_runs.json`](reported_runs.json) indexes the **448 physical runs**
  used in the current figures. Full-batch SFRAMES has 28 runs; each of the other
  five plotted variants has 84. The index is generated from the frozen summaries
  and metadata, not an additional experimental result.
- Plain run records contain metadata and final results for those 448 runs and
  their **168 distinct continuation origins**. These include raw loss,
  feasibility, optimizer parameters, work counters and provenance hashes.
- [`reported_traces.tar.gz`](reported_traces.tar.gz) contains the 448 original
  `trace.json` files, losslessly compressed with their original relative paths.
  They contain 272,300 recorded observations, of which 272,132 are within the
  final gradient budget. Native overshooting observations are preserved; the
  report applies its documented budget filter.
- [`historical_summaries.tar.gz`](historical_summaries.tar.gz) contains eight
  large predecessor summaries/statistics files. Their member paths and hashes
  are listed in the manifest. Current gradient-budget summary/statistics remain
  directly readable under `spectral_balls/expanded/gradient_budget/`.

From the repository root, verify this archive and the older factorization
archive with:

```sh
python3 scripts/verify_stochastic_matrix_archive.py
```

To inspect a compressed original without extracting it:

```sh
tar -tzf results/stochastic_matrix_factorization/ct/reported_traces.tar.gz
tar -xOzf results/stochastic_matrix_factorization/ct/historical_summaries.tar.gz \
  spectral_balls/expanded/baseline_step_rules/report_stage_ca3r3ct9/corrected_summary.json
```

The tarballs contain only regular files with relative paths. They can be
extracted into a separate scratch directory when native trajectories or
predecessor summaries are needed for analysis. Their bytes are sufficient to
inspect the recorded curves and regenerate the plotted statistics without
loading the CT dataset. The preserved campaign/report scripts still contain
their original host paths and validation assumptions; they are historical
source snapshots rather than a portable one-command campaign launcher.

Raw/preprocessed CT arrays, calibration arrays, optimizer checkpoints,
environments, downloaded dependencies and papers, caches, logs, fixture output,
duplicate rendered reports and unrelated historical per-run files are omitted.
Small data sidecars, acquisition/preprocessing code, source snapshots,
configurations, selections, validation records and relevant failed/superseded
stage audits remain included. The original files and complete checkpoints
remain on Elite. This archive supports result inspection and figure
regeneration; resuming optimization requires those external data/checkpoints.
