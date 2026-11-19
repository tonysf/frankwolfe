# Stochastic CT paper section

This bundle captures the completed **19 November 2026 gradient-budget comparison** of SFRAMES, SPRING–SARAH, ProxSARAH and PALM on 2DeteCT. It contains the exact published [PDF](report/sframes_ct_section.pdf), LaTeX section, embedded bibliography, and two mean ±1 SD trajectory figures. The endpoint comparison uses 69,120 gradient views per direction, with curvature work recorded separately. There are 448 physical trajectories, including 168 completed PALM/SPRING continuations.

The [public reproducibility appendix](CT_REPRODUCIBILITY.md) links to the repository's [archived experiment evidence](../../../results/stochastic_matrix_factorization/ct/archive_manifest.json). The [manifest](manifest.json) records the original and bundled SHA-256 digests. All report files are byte-identical to the published versions. Only the appendix's relative file links were relocated; its [original bytes](provenance/CT_REPRODUCIBILITY.source.md) are retained.

## Validate

From the repository root:

```bash
python -m paper.experiments.stochastic_ct.bundle --check --require-archive
```

To validate this directory independently of the experiment archive, omit `--require-archive`. Validation uses Python's standard library and checks every captured file, the two figure dependencies, embedded references, and local document links. It does not run an optimizer or access the network.

## Rebuild the PDF

Install `latexmk` and a LaTeX distribution providing the standard packages used by the driver (`fontenc`, `lmodern`, `geometry`, `amsmath`, `amssymb`, `booktabs`, `graphicx`, and `hyperref`). With those dependencies installed locally, the report builds offline using its bundled figures; raw CT data, ASTRA, NumPy and optimizer checkpoints are not required.

```bash
python -m paper.experiments.stochastic_ct.bundle --build
```

The new PDF is written to `build/report/sframes_ct_section.pdf` below this directory, leaving the captured PDF unchanged. Use `--output-dir PATH` to select another new or empty output directory. The compiler log is saved in that directory. Compiler versions and PDF metadata can change the rebuilt PDF's byte hash without changing the report's scientific content.

The same commands work directly as `python paper/experiments/stochastic_ct/bundle.py ...`. This is a document rebuild; reproducing the optimization runs requires the data preparation and numerical environment described in the appendix and archived campaign notes.
