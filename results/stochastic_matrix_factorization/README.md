# Stochastic matrix factorization experiments

This archive records the experiments incorporated into the `stochastic-frames`
branch on November 19, 2026. It preserves the numerical sources, configurations,
selection records, validation, and reported evidence from the original Elite
experiment directories. It does not change the algorithms in the reusable
`frank_wolfe` package.

## Current CT comparison

- [Paper section and PDF](../../paper/experiments/stochastic_ct/README.md)
- [Reproducibility supplement](../../paper/experiments/stochastic_ct/CT_REPRODUCIBILITY.md)
- [CT archive and exclusions](ct/ARCHIVE.md)
- [Latest continuation protocol](ct/spectral_balls/expanded/gradient_budget/README.md)
- [Completion audit](ct/spectral_balls/expanded/gradient_budget/validation_summary.json)

The completed comparison uses 14 slices from six acquisition groups, two
training-view counts, and three stochastic repetitions. SFRAMES, SPRING–SARAH,
ProxSARAH, and PALM fit a rank-64 factorization to real 2DeteCT measurements.
The reported problem includes nonnegative factors and separate spectral-norm
bounds. The spectral bounds are modeling restrictions introduced for the
Frank–Wolfe formulation, not physical constraints of CT.

The latest campaign extends all 84 PALM and 84 SPRING–SARAH runs to 69,120
gradient forward views and 69,120 gradient adjoint views. Curvature work is
recorded separately. The two figures show weighted means and one standard
deviation from native per-iteration observations. Optimizer time is not a
comparison metric. The CT archive includes the 448 physical trajectories used
by those figures in a lossless compressed archive.

The proposed comparison without spectral bounds, including multiplicative
updates or alternating nonnegative least squares, has **not** been run. It is
not part of these results.

## Earlier factorization study

The [ORL/Yale-B archive](factorization/ARCHIVE.md) preserves the preceding
image-factorization experiments and SPRING step-size ablation. Its problem,
constraints, method settings, and historical timing measurements differ from
the CT study. Do not combine those results as if they came from the current
protocol. Superseded CT campaigns are similarly retained as provenance and
identified in the CT archive index.

## Integrity and reproduction

From the repository root, verify all archived source and result bytes with:

```bash
python scripts/verify_stochastic_matrix_archive.py
```

The verifier also streams through compressed trace members and checks their
original hashes; it does not extract them or run any experiments. The paper
bundle has separate instructions for validating and rebuilding the PDF from
the included figures without downloading CT data.

Each study's `archive_manifest.json` records original relative paths, sizes,
and SHA-256 hashes. Frozen source and provenance files remain byte-identical;
absolute Elite paths inside them are historical execution records, not
portable installation locations. In particular, campaign scripts that encode
those paths are archived execution recipes, not a claim that every historical
launcher can run unchanged in a fresh checkout.

Raw datasets, cached preprocessed arrays, calibration arrays, optimizer
checkpoints, environments, and caches stay outside Git. Source download and
preprocessing instructions and available data hashes are retained. Resuming
an original run requires its separately retained checkpoint; the Git archive
alone is not a complete backup of the execution directories. No off-machine
backup of the excluded artifacts is asserted by this archive.
