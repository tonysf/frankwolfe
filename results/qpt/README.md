# Structured stochastic-FRAMES QPT experiment record

The campaign directories retain the saved scalar trajectories, configurations,
source hashes, independent verification reports, work logs, and report builders.
Paths in historical records refer to the machine where the experiment ran; they
are provenance, not portable launch commands. Failed attempts remain labelled.

Shareable reports are in [`output/pdf`](../../output/pdf):

- `qpt_coauthor_report_20260915.pdf`: recovery experiments through the original
  successful eight-qubit run.
- `qpt_exponent_comparison_20260922.pdf`: twelve old-versus-requested exponent
  comparisons, using solid colored trajectories.
- `qpt_n8_scale_tuning_20260923.pdf`: four eight-qubit scale comparisons with the
  requested exponents and the validated host-preparation optimization.
- `qpt_n8_followups_20260923.pdf`: late-trajectory analysis and the proposed
  fixed continuation of the best requested-exponent run.

The eight-qubit scale campaign is `n8_scale_tuning_20260923_01`. Its best final
factor fidelity was 98.7689% at 148,000 steps with step scale 0.5 and momentum
scale 0.6. The follow-up proposal is a continuation to 200,000 total steps, not
a new completed result. Consult its separate execution record for later status.
Fidelity is a full-factor overlap diagnostic, separate from measurement loss
and trace-preservation feasibility. These comparisons use one dataset and seed.

The ordinary source tree contains the on-demand observation mode, product-state
backend, prefetching and restart support. The later host-preparation optimization
remains an isolated experiment: `n8_scale_tuning_20260923_01/stage_source.py`,
`qpt_batch_preparation.py`, and `source_deployment.json` record its construction
and hashes. Synchronizing the repository does not replace the frozen Ruche
`source_fast` snapshot used by that campaign.

Large NPZ datasets, factor/restart archives, source bundles, generated figures,
transient logs and historical automation files are preserved in the separate
checksum-verified experiment backup rather than Git. Scalar evidence and source
recipes in this directory accompany the four final PDFs in Git. Do not regenerate
datasets or overwrite existing runs to reconstruct a missing artifact.

Future reports should follow [REPORT_STYLE.md](REPORT_STYLE.md).
