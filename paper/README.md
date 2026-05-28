# Paper Figure Scripts

This directory contains only the paper-facing figure-generation entrypoints.
They reuse experiment helpers from `examples/` and algorithms from
`frank_wolfe/`.

- `generate_main_figures.py` generates the main matrix-factorization and splitting
  figures.
- `generate_nonintersecting_linf_figures.py` generates the inconsistent
  nonintersecting L-infinity example.
- `generate_trend_filtering_trajectory_figure.py` generates the SCAD/MCP
  trend-filtering trajectory comparison.

Run scripts from the repository root, for example:

```bash
python -m paper.generate_main_figures
python -m paper.generate_nonintersecting_linf_figures
python -m paper.generate_trend_filtering_trajectory_figure
```

Generated figures are ignored by git unless explicitly added for a release.
