# Examples

This directory contains notebooks and scripts used to exercise the Frank-Wolfe
implementations. Paper-specific figure entrypoints live in `../paper`.

## Experiment Helpers

- `nonneg_matrix_factorization.py`, `trend_filtering_mf.py`, and
  `l1_splitting_nonconvex.py` define the primary experiment models.
- `exponential_inconsistent_experiment.py`,
  `pathological_inconsistent_experiment.py`, and
  `semialgebraic_inconsistent_experiment.py` generate additional
  inconsistent-constraint examples.

Run scripts from the repository root, for example:

```bash
python -m examples.nonneg_matrix_factorization
```

Generated local figure and paper-snippet files are ignored by git.
