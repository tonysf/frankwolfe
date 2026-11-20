# Gromov–Wasserstein experiment roundup

Collected from Elite and Ruche on 6 November 2026. This is a source and compact-evidence archive, not a full result-data backup or a new experiment run.

- `gw_benchmark_experiment_20260927/` contains the current Elite benchmark kernels, tests, campaign extensions, configurations, selected reports/figures, and compact scalar evidence.
- `ruche_snapshots/` stores content unique to 20 named Ruche campaign source snapshots and selected compact historical summaries. Files identical to another archived file are represented by hash-checked pointers in `inventory.json`.
- `inventory.json` maps every selected original path to its stored bytes and SHA256, records exclusions, original locations, source identities, and transfer scope. `local_git_history.tsv` and `local_git_status.txt` preserve the original local repository's revision history and dirty/untracked status at collection. Elite was at `7b22612` with no Git remote.
- `raw_artifact_backlog.json` records the limits of raw-result coverage. The partial Ruche metadata inspection observed 2,810 array files totaling 323,288,673,474 pathname bytes. This is not a count of unique bytes missing from Elite: retries and replicas can duplicate content, and no full checksum comparison was performed. The existing approximately 60 GB Elite experiment tree is not evidence of a complete Ruche backup.

## Read the existing evidence

Start with the archived [experiment account](gw_benchmark_experiment_20260927/reports/EXPERIMENT_REPORT.md), [method definitions](gw_benchmark_experiment_20260927/reports/METHOD_NAMES.md), [stochastic comparison](gw_benchmark_experiment_20260927/reports/stochastic_primary/report.md), and [reproduction notes](gw_benchmark_experiment_20260927/EXPERIMENT.md). Historical statements in these files describe their original experiment stage.

The [3 November SHARE summary](gw_benchmark_experiment_20260927/reports/share_progress_20261103/summary.json) explicitly describes a best completed two-seed tuning setting without fresh-seed confirmation and without per-loss timestamps. Those limitations remain. At initial live inspection on 6 November, job **2297800 was RUNNING** and **2297801 was PENDING**, both associated with `share_resumable_20261018_c73a754/source`. Their changing result/checkpoint files were not copied as completed evidence. No jobs were submitted, cancelled, or modified.

## Reconstruct before rerunning a snapshot

Historical sources and job scripts were deduplicated; their original runnable relative layout is restored by this standard-library utility. From this directory:

```bash
python3 restore_layout.py
python3 restore_layout.py --origin elite --out /tmp/gw-elite-selected
python3 restore_layout.py --origin ruche --out /tmp/gw-ruche-selected
```

The first command verifies all selected file hashes/sizes. The other commands require an absent or empty destination, verify the content, and materialize every selected original relative path, including duplicate source/script dependencies. They do not recreate excluded data, environments, dense exports, arrays, checkpoints, logs, or full downloaded manuscripts. The output roots correspond respectively to the original Elite and Ruche experiment roots. Use a different empty directory if these example destinations already contain files.

The current Elite source entry points include `run_benchmark.py`, `run_smoothing.py`, `run_native_sampled.py`, and the campaign helpers in `extensions/`. Consult each campaign's existing configuration and README for its method and data requirements. Dataset preparation is described by `make_instances.py`, `prepare_development_data.py`, and the original protocol; datasets must be obtained or restored separately. Five expanded original task/config manifests are retained as explicit size-cap exceptions (7.6 MB total), because the current launchers/tests depend on those task definitions. Other dense historical exports remain omitted as recorded in the inventory. Scalar summaries alone cannot reconstruct every original loss trajectory or final coupling.

The dependency freezes are observations from their original environments. The Elite freeze records a CPU PyTorch build; `ruche_snapshots/environment.ruche.lock.txt` records the Ruche CUDA build and some nonportable package URLs. Use the campaign's relevant freeze and dependency recipe after checking the intended machine rather than installing an unrelated freeze verbatim. Existing absolute `/home/tony/...` and `/gpfs/workdir/silvetian/...` paths in source/configuration/provenance are preserved as evidence and must be adapted in a copied configuration before reuse. On Ruche, run computational preparation, tests, benchmarks, and data processing inside Slurm allocations. Source restoration and verification do not execute a benchmark.

## Collection verification

The remote transfer used an explicit reviewed file list (3,243 files, 31,275,568 bytes before final curation) and SSH tar streaming. Archive member count/sizes matched the metadata selection. SHA256 hashes were computed from copied bytes on Elite; no expensive cluster-side rehash was performed. Code/data originals and active source directories were preserved. Original scientific reports are archived evidence; no benchmark results were regenerated for this roundup.
