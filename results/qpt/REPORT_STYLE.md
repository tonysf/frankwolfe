# QPT report style

User preference confirmed on 2026-09-23: retain the visual and report style of
`output/pdf/qpt_exponent_comparison_20260922.pdf` for future updates.

- Landscape pages, DejaVu typography, dark blue headings, light table stripes,
  and readable scientific axes and captions.
- Use solid curves for both old and new runs. Distinguish old blue (`#176B9A`)
  and new orange (`#BF651D`) by color; do not use dashed new-run curves.
- Dotted reference lines may identify the 99% threshold or a sample budget.
- Put fidelity, sampled measurement loss, and TP residual side by side against
  iterations; show equal sampled-row budgets when they inform the comparison.
- Include endpoint tables, exact schedule scales/offsets/exponents/caps, batch
  versus dataset size, noise and seed scope, and verification/provenance.
- Plot saved scalar evidence without fitting or smoothing the trajectories.
  State logarithmic scales explicitly and use readable numeric ticks.
- Preserve earlier reports and visually inspect final rendered pages.

Reusable builders and source figures:
`results/qpt/exponents_20260922_01/make_comparison_plots.py` and
`results/qpt/exponents_20260922_01/make_comparison_pdf.py`.
