# Gradient-budget continuation of PALM and SPRING–SARAH

This experiment isolates gradient work from the cost of estimating block curvature. It continues all 84 published PALM runs and all 84 published SPRING–SARAH runs to a common cap of **69,120 gradient forward views and 69,120 gradient adjoint views**, matching the nominal cap used by ProxSARAH. ProxSARAH and all SFRAMES runs are reused unchanged. No optimizer parameters are retuned.

## Continuation and work accounting

Each continuation restores the original final factors, optimizer iteration, recursive gradient estimates and all random-generator states from its validated checkpoint. It preserves the complete original trace as an exact prefix, including the last update that crossed the former inclusive view cap. The stopping condition is changed; the update equations, batch sizes, power iterations, refresh rules and projection tolerances are unchanged. All observations remain cumulative from initialization.

For either projection direction,

$$
W_{\mathrm{gradient}}=W_{\mathrm{total}}-W_{\mathrm{curvature}}.
$$

`forward_views` and `backward_views` retain their original meaning: actual total operator views, including curvature. `curvature_forward_views` and `curvature_backward_views` record Hessian work separately. New rows additionally store `gradient_forward_views` and `gradient_backward_views`; readers derive the same values for unchanged prefix rows. `gradient_calls` counts literal joint or block gradient evaluations. `oracle_calls` additionally counts Hessian applications. Loss evaluations used only for diagnostics and the shared initialization are outside optimizer work; optimizer timing is not compared.

The final native update is retained even when it crosses the gradient cap, but endpoint and first-hit comparisons exclude it. PALM reaches this cap after 96 updates with 360 views and 32 updates with 1,080 views. Its twelve full-data Hessian applications and two full-data block gradients per update give total work seven times gradient work. SPRING's update count and total/gradient ratio depend on the recorded full-gradient refreshes.

## Comparison and figures

The existing endpoint table uses the new gradient-budget endpoints for PALM and SPRING; it retains all SFRAMES and ProxSARAH entries and recomputes the best-value bolding. Its aggregation remains the median over repetitions, then slices within acquisition groups, then groups. Paired changes between the old and extended endpoints are computed before the same aggregation. Raw loss, repaired loss, held-out loss, reference error and raw feasibility remain separate criteria.

The same two figures retain four metric rows and three work axes: iterations, gradient calls and gradient views. They show one arithmetic mean and one descriptive ±1 standard-deviation band for every method, with equal acquisition-group weight and equal slice/repetition weights within groups. All 448 physical traces are represented; the deterministic full-batch SFRAMES arm contributes one run per slice. Work-axis summaries use the last recorded iterate at each work value, with step curves, no linear interpolation, and the fixed cohort's common observed range. The bands describe variation across slices and repetitions, not confidence intervals.

`plot_trajectories.py` copies the finalized renderer from `../trajectory_mean_sd/mean_sd_plots.py`. Only the work-axis names/labels, the corresponding linear-gradient-view axis conditions, and its module description change. The original renderer and its source hash are recorded in the publication audit. Means, SDs, weights and grids are saved in `statistics_360.json` and `statistics_1080.json` beside the staged report.

This is **not an equal-total-work comparison**. The report gives the paired old-to-new loss changes and the separately measured curvature cost in a short paragraph. The derived summary and reproducibility appendix retain the old inclusive-budget endpoints, complete work decomposition and first-hit comparisons. The previous numerical summaries remain immutable.

## Reproducibility and publication

`specification.json`, `evaluation_plan.json`, `protocol.json` and `implementation_manifest.json` freeze the cohort, original checkpoint/file hashes, copied numerical code and continuation semantics. `validation_summary.json` requires all 168 continuations, exact original prefixes, dense native observations, unchanged step rules, correct total/gradient/curvature counters and native endpoints within the new cap. A numerical or operational failure halts the campaign; evaluation results are not filtered.

`update_report.py` publishes only after that validation and its bounded report fixture pass. It starts from the published mean/SD report and the corrected all-method and joint-SFRAMES summaries. It replaces only PALM and SPRING endpoints and traces, recomputes their paired and target statistics, and preserves the SFRAMES-versus-ProxSARAH comparison. The paper retains two figures and one table.

Publication stages and compiles the report, verifies all inputs, backs up current report assets and four predecessor publication audits, and journals replacement. The predecessor audits gain a gradient-budget successor link and current asset hashes; their scientific anchors, historical summaries, original asset hashes and existing successor links remain intact. The historical feasibility-result appendix marker is byte-identical, so replaying its upstream publisher cannot restore obsolete current results. Recovery accepts only the recorded old or new versions of each file; unrelated edits stop replay.

`report_update.json` links the new `gradient_comparison_summary.json`, the unchanged joint-SFRAMES summary, plotted statistics, exact backups, and the publication chain.
