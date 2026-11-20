# Stochastic CT paper section

This bundle compares SFRAMES, SPRING–SARAH, ProxSARAH and PALM with nonnegative-only alternating NNLS, multiplicative updates and joint L-BFGS-B on 2DeteCT. The latter three methods have no spectral constraints. The two mean ±1 SD figures and endpoint table cover 532 physical trajectories at a 69,120-view cap per direction. All joint L-BFGS-B line-search work is charged; historical curvature work remains separate.

The [PDF](report/sframes_ct_section.pdf), LaTeX, figures and [public appendix](CT_REPRODUCIBILITY.md) are bundled. [Joint L-BFGS-B evidence](../../../results/stochastic_matrix_factorization/ct_joint_lbfgsb/archive_manifest.json), [NNLS/MU evidence](../../../results/stochastic_matrix_factorization/ct_nonnegative/archive_manifest.json), and [original evidence](../../../results/stochastic_matrix_factorization/ct/archive_manifest.json) retain frozen sources, settings and trajectories. The [manifest](manifest.json) records hashes and predecessor provenance. The gradient-estimator explanation is preserved.

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
