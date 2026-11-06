# Stochastic FRAMES experiment archive, 6 November 2026

This directory preserves seven local experiment campaigns collected from Elite.
It contains their numerical source and tests, frozen configurations and audits,
source/data checksums, compact run receipts, and selected authored reports and
figures. The original campaign directories were left untouched. No campaign was
rerun and no scientific result was independently revalidated during collection.

| Campaign | Contents |
| --- | --- |
| `frames_stochastic_toy_20261014` | Scalar demonstration, exact short-horizon output calculation, code, configuration, table and figures. |
| `frames_stochastic_l1ls_1024_20261014` | Least-squares demonstration, plotting code, metadata, scalar table and figures. |
| `frames_stochastic_nonconvex_1024_20261014` | Nonconvex example and zero-initialization variant, authored note, diagnostics and tables. |
| `frames_stochastic_rows_1024_20261014` | Row-sampling example, eight-seed and longer-iteration variants, authored notes, diagnostics and tables. |
| `frames_factorization_20261104` | Spectral factorization campaign, data preparation, author-code control port, tests, run configurations, original audits, plots and spring-step ablation. |
| `frames_ct_2detect_20261104` | 2DeteCT pilot, initialization variants, spectral-ball comparison, expanded cohort, dense trajectories, joint tuning/confirmation, report and manuscript-section sources. |
| `frames_matrix_completion_20261104` | Synthetic convex matrix-completion comparison, reference-solver code, tests, design checks, frozen protocol, scalar/curve tables and authored report. |

## Reading and provenance

`CURATION_MANIFEST.json` lists every included file with its staged and original
SHA-256. Its exclusion inventory identifies each omitted ordinary artifact and
symlink; entries for Git metadata, caches and downloaded/vendor directories cover
their entire subtree. Data arrays and checkpoints have individual checksums but
are not distributed. The manifest also records original Git history/status,
source-module link targets, and files containing historical absolute paths.

The factorization campaign had its own local Git repository: `main` at
`488c9ba` (full hash in the manifest), with two commits on 4 November
and no remote. Its per-run configurations and spring-step ablation also included
untracked local work. The other six campaign roots had no `.git` directory.
This archive captures the bytes present at collection, including that untracked
work; it does not imply those bytes were previously published.

For the three November campaigns, `RUN_EVIDENCE.jsonl` gathers per-run result,
failure and control receipts. Every record names the original relative path and
its SHA-256. Configurations, numerical-source/data hashes, statuses and scalar
endpoints are preserved. Raw trace lists and long index sequences are omitted;
their lengths and JSON-content checksums are recorded. These files support
inspection of archived endpoints, not replay of the original checkpoints.
Existing report summaries, selections and validation receipts are historical
evidence, not checks performed by this collection.

Full fetched research papers, extracted manuscript text, downloaded third-party
source trees, datasets, array outputs, checkpoints, caches, transient logs and
process status files are omitted. Source URLs and recorded hashes remain in
the campaign provenance. Redundant rendered page previews and some duplicate
PNG/PDF figures are omitted. Internal Python-module symlinks were copied as
regular source files so their imports remain available. Data, calibration and
run-directory symlinks were omitted and are inventoried.

Personal attachment-directory identifiers were replaced by `SOURCE_MANUSCRIPT/`
in copied text. This is the only automatic text transformation. Original hashes
remain available. Existing machine paths under `/home/tony` describe historical
research paths; the manifest's path inventory identifies them for porting.

## Preparing a rerun

Treat these as archived campaign programs: inspect and copy a campaign into a
fresh writable run directory before launching it. Many programs write reports,
selection files or manifests beside their source, acquire locks, resume local
checkpoints, or expect frozen data. The collected summaries are not a complete
runtime directory. Rebuild missing raw outputs before calling report generators
that read `runs/*/trace.json`, checkpoint files, or calibration arrays.

Use a separate Python environment. Existing `requirements.txt` files preserve
the recipes supplied with the October examples and matrix-completion campaign.
`requirements-captured.txt` in factorization and CT records direct package
versions recoverable from original provenance; it is not a complete transitive
lock or a freshly tested installation. Preserve the per-campaign environment
rather than combining the recorded NumPy versions. The original November
scripts used Python 3.12.14 on Linux and one BLAS thread per worker.

Before a rerun:

1. **Paths and CPUs.** Replace historical interpreters such as
   `/home/tony/zofw_experiments/.venv/bin/python` with the new environment.
   Review the `hardcoded_path_inventory` in the manifest. Adapt pinned CPU lists
   `(0, 2, 4, 6)` to the target machine, and review worker counts. Reruns on a
   cluster require that site's scheduler and storage rules.
2. **October source manuscripts.** Several demonstrations read a manuscript
   solely to record its SHA-256. Supply the original document at a chosen local
   path and update `SOURCE`; it was intentionally excluded from this public
   archive. The toy example's path is now the explicit placeholder
   `SOURCE_MANUSCRIPT/Nonsmooth_FW-7.pdf`. Other examples retain their historical
   `gw_benchmark_experiment_20260927/provenance/manuscript.txt` reference.
3. **Factorization data and controls.** Read `README.txt`,
   `provenance/data_manifest.json` and `provenance/source_manifest.json`.
   `prepare_data.py` contains the public face-data download/preprocessing recipe.
   The independent SPRING author-code control also expects files from the
   recorded GitLab archive/commit; restore those separately. Data and external
   code are not bundled. Restore the ablation's `data` directory reference to
   the campaign's prepared data. The materialized `core.py` and
   `setup_problem.py` need no source symlinks.
4. **CT data and initialization.** `prepare_data.py` records the 2DeteCT
   Zenodo/mirror source and preprocessing. Compact per-slice JSON provenance is
   included under `data/`; its numerical arrays are omitted. ASTRA CPU
   `line_fanflat` projection was used. Prepare the data at the expected relative
   locations for each variant, and restore their shared data/calibration
   directory references from the symlink inventory. Expanded calibration can
   be regenerated with the captured source. Dense/joint confirmation programs
   additionally compare against original-run traces; regenerate or separately
   restore those traces before validation. A scalar receipt is not a replacement.
5. **Matrix completion.** `reference.py` deterministically creates the frozen
   synthetic instance and solves the numerical reference; its default radius
   is 40. Arrays and reference duals were excluded, while their checksums and
   scalar reference metadata remain. Review `README.txt`, the design checks
   and `provenance/interpretation_correction.json`. Its report builder also
   refers to the historical row-sampling example; redirect that reference to
   the sibling archived campaign or newly regenerated output.
6. **Reports.** TeX and selected required figures are preserved in their
   original relative layout. A TeX installation and the packages named by each
   document are needed to rebuild them. Reports depending on omitted numerical
   outputs require a fresh campaign run or a separate private artifact restore.
   `spectral_balls/report_template.tex` is a generator template: its figure
   reference is resolved from the generated `reports/` directory, where
   `ct_sframes_figures.pdf` is included, rather than from the template directory.

The original notes distinguish theorem statistics, last-iterate diagnostics,
pilot timings and adapted controls. Read those qualifications with the figures.
This collection makes no new comparative-performance or convergence claim.
