# Historical ORL and Yale-B factorization experiments

This is a curated archive of `/home/tony/frames_factorization_20261104` on Elite, originating from Git commit `488c9ba8220a5215b87dd09bdaf4b1b7fafb5fd1`. It preserves the earlier face-image factorization experiments. The later CT/tomographic experiments have a different data model, constraints, implementations, and budget accounting; their results should not be combined with these runs.

The copied files retain their original bytes and relative paths. The two source symlinks `spring_step_ablation/core.py` and `spring_step_ablation/setup_problem.py` have been materialized as ordinary files with identical contents. Numerical methods, parameter names, diagnostics, and historical claims have not been updated to match later experiments. The original source directory was not modified.

## Problem and recorded campaigns

The data matrix contains unit-norm image columns: ORL64 has shape `4096 × 400`, with factorization rank 25; Extended Yale-B32 has shape `1024 × 2414`, with rank 49. The objective is `||UVᵀ − A||²_F / (2mn)`. Each factor is constrained to an operator-norm ball. The additional constraints are entrywise nonnegativity of `U` and a global entrywise ℓ₁ bound on `V`; `V` may be signed. This is the declared spectral/convex-sparsity adaptation in the original [README.txt](README.txt), not the original SPRING paper's ℓ₀ problem.

The main campaign records 26 completed development runs at 20,000 iterations and 12 completed confirmations at 50,000 iterations. [selection.json](selection.json) retains all development scores and the frozen selection rule. Per-run metadata and results retain initial settings, source hashes, final raw and repaired losses, feasibility, gradient counts, and the historically recorded timing fields. The separate author-code controls use a different problem and remain under `runs/original_spring_*`; their source-port departures are described in [provenance/spring_original_settings.json](provenance/spring_original_settings.json).

**The practical SPRING step-size ablation did not complete.** Both local-curvature arms failed when Dykstra projection reached its 2,000-iteration limit without attaining its numerical target. ORL's saved progress reports iteration 646 and its last probe is at 600. Yale-B's saved progress reports 5882, while its saved trace and report reach 5900; this inconsistency is preserved, not repaired. These records are failed historical diagnostics and must not be presented as completed practical-SPRING comparisons. Their small traces, progress records, frozen manifest, validation record, and report summary are retained. [archive_summary.json](archive_summary.json) records the failure evidence and hashes of the excluded logs.

No Newton–Schulz benchmark source or result was found in this source directory, or in the scoped filename search under `/home/tony/experiments`. This archive therefore contains no Newton–Schulz timing result.

## Contents and exclusions

- Authored solvers, runners, data preparation, plotting code, and tests are preserved, including the distinct SPRING ablation implementation.
- Configurations, selected parameters, all main-run terminal results and metadata, original-control scalar histories, small ablation traces, and compact report summaries are preserved.
- Data/source manifests, algorithm audits, historical package versions, and validation records preserve the experiment's provenance. The original pytest XML is summarized in `archive_summary.json`: its recorded result was 29 tests, with no failures, errors, or skips. Tests were not rerun as part of this archive operation.
- Datasets, checkpoints, main-run trace payloads, environments, caches, logs, locks, third-party code, downloaded paper PDFs, generated figures/PDFs, and the obsolete large status snapshot are omitted. Their exclusions are inventoried in [archive_manifest.json](archive_manifest.json); original copies remain outside this Git archive.

`archive_manifest.json` lists each included file's path, byte size, and SHA-256. Original file entries also identify their source-relative path; generated archival documentation/audit entries are explicitly marked. Excluded ordinary files have hashes and categories. Excluded Git/cache directory inventories and the omitted data-directory symlink have no content hash. The manifest does not hash itself. No compressed archive is used.

Archival verification checked every copied file against its source, parsed the retained Python and JSON, and verified that all 38 main runs and both ablation runs name the matching numerical source hashes. This is evidence preservation, not a fresh numerical replication.

## Reproduction dependencies

Use a separate working copy for any rerun: the historical runners write into their own project tree, and several records retain original absolute paths. This small archive is not a resumable checkpoint bundle, and its plotting scripts cannot reconstruct every old trajectory without restoring the omitted main-run traces.

The recorded environment is Python 3.12.14 with NumPy 2.4.0, SciPy 1.17.0, and Matplotlib 3.10.8. Tests additionally used pytest 9.0.2, CVXPY 1.9.3, and Clarabel 0.11.1. See [provenance/environment.json](provenance/environment.json) for the original host and threading settings. No environment is bundled or installed by this archive.

[provenance/data_manifest.json](provenance/data_manifest.json) records the original data URLs, hashes, dimensions, preprocessing, and processed-file identities. `prepare_data.py --spring-reproduction` also requires the excluded author SPRING archive and its bundled saved initial factors; [provenance/source_manifest.json](provenance/source_manifest.json) supplies its pinned download URL and SHA-256. Restore the expected `data/processed/*_unitnorm.npz` files and the ablation's original `data -> ../data` relationship, or adapt paths in a separate reproduction copy. Restoring the datasets or third-party archive may be necessary before the historical preparation/runners can execute.

Optimizer-time values are retained as historical evidence. As the original documentation states, the runs shared a CPU host and do not establish a fair isolated throughput comparison.
