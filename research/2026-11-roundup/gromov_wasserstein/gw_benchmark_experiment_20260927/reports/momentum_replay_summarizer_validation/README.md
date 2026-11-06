# Replay reporter parser validation

This folder contains parser/layout evidence only, not campaign replay results.
Eight focused tests run with:

```bash
.venv/bin/python -m pytest -q test_summarize_momentum_replays.py
```

`constructed_parser_fixture/` is a manufactured parser test, not an optimization trajectory. Its layout plot is explicitly labeled **NOT EXPERIMENT DATA**. It must never be supplied as campaign replay evidence.

Four real, existing tiny CPU verification/CSV pairs from utility tests (sketch product, exact randomized, row projection randomized, and archived pilot source) also passed `validate_verified` with three checkpoints each; no optimizer was rerun. Their original pytest temporary directory was subsequently rotated away, so those proof files are not represented as archived here.

The real three frozen manifests were parsed against intentionally absent temporary roots: all 796 records remained missing, no figures were generated, and `--require-complete` exited 1. Production invocation pairs each repeated `--manifest` with its corresponding downloaded `--root` in order and requires a fresh `--out` directory. Later documentation commits are compatible because utility source is validated at each receipt's controller revision using `git show`.

The final report-only population enhancement has a ninth focused test. The new denominator case and affected missing-record case passed (2 passed, 7 deselected):

```bash
.venv/bin/python -m pytest -q test_summarize_momentum_replays.py -k 'prescribed_plot_denominators or missing_prescribed_receipt'
```

This reads frozen scalar manifests, never held-out arrays: exact product comparisons prescribe five synthetic instances per family and one BMMC instance. Missing runs and windows reduce observed counts without reducing those denominators. Every target has prescribed/observed run and instance counts and `coverage_complete` in the JSON and `curve_summaries.csv`, including wholly missing targets. Incomplete populations use open plot markers. Markdown links the generated figures.

Regenerate the explicitly manufactured layout fixture with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python reports/momentum_replay_summarizer_validation/plot_fixture/manual_plot_fixture.py
```

The plotted scalar values are constructed only for visual QA; the fixture watermark says **NOT EXPERIMENT DATA**. The regenerated PNG was inspected for readable captions, open/filled markers, and legend placement.
