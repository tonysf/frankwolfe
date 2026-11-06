# Completed BMMC scaling campaign

**Scientific interpretation corrected on 2 October 2026:** the completed momentum-KL comparisons are exploratory tests against a custom comparator. They do not supply the requested comparison against a published stochastic algorithm under its stated theory. Their measurements are retained; the primary-baseline designation is withdrawn. See [the correction](COMPARISON_SCOPE_CORRECTION.md). Replacement GPU results are not yet available.

**All 288 tuning runs and 40 final runs completed with valid saved outputs on Ruche A100 GPUs. Transport KL-proximal gradient with momentum achieved lower final feasible GW loss in all 20 paired comparisons at 300 solver seconds.**

Both methods use the same q=128 sketch gradient and momentum schedule. The table gives median final float64 GW loss across five fresh seeds; lower is better. FW denotes **Smoothed row FW + momentum (sketch gradient)**; KL denotes **Transport KL-proximal gradient + momentum (sketch gradient)**.

| Cells/view | Initialization | FW loss | KL loss | KL reduction relative to FW |
|---:|---|---:|---:|---:|
| 3,000 | product | 0.149619 | 0.145843 | 2.52% |
| 3,000 | randomized | 0.149589 | 0.145859 | 2.49% |
| 6,224 | product | 0.156729 | 0.152488 | 2.71% |
| 6,224 | randomized | 0.156968 | 0.152442 | 2.88% |

FW performs about 3.4 times as many outer updates. It has lower median feasible loss at the recorded 30-second checkpoints for all four groups; at 100 seconds it still leads at 6,224 cells, while KL leads at 3,000. By 300 seconds KL leads at both sizes and both starts. These checkpoint comparisons use actual completed observations, without interpolation.

The independently tuned KL step is eta=0.1 at both sizes, compared with inherited eta=10 in the original 1,000-cell study. Randomized final runs now also receive 300 seconds rather than the earlier 30. The change in ranking cannot be attributed to size alone. Equal solver time does not mean equal oracle work or equal total process time.

All saved rounded couplings passed validation. The ten final KL runs at 6,224 cells recorded 4–21 intermediate inner stopping misses each; these remain visible in the complete report. FW raw iterates have larger column-marginal residuals, so the comparison uses the common rounded output. This is one fixed biological dataset with optimizer-seed repetitions and a limited tuning grid.

[Complete report and figures](bmmc_scaling_final/report.md), [verified collection and paired comparisons](bmmc_scaling_final/completion.json), and [frozen selection](bmmc_scaling_final/selection.json). Seven downloaded files matched their Ruche SHA-256 checksums. The original dense coupling archives remain on Ruche. Final report generated 2026-09-28T21:22:35.578315+00:00.

| Stage | Slurm job | Work |
|---|---:|---|
| Corrected pilot | 1968437 | Two stochastic methods at 3000, then 6224 subject to validity/resources |
| Eligibility | 1968447 | Verify both pilot sizes and frozen data before GPU tuning |
| Tuning | 1968448 | 24 array tasks; 288 runs; at most 3 A100 tasks concurrently |
| Selection | 1968449 | Freeze independent selections per (method, size) |
| Final comparisons | 1968450 | 20 paired array tasks; 40 runs; at most 3 A100 tasks concurrently |
| Report | 1968451 | Audit outputs, aggregate results and render time/probe figures |

Compare **Smoothed row FW + momentum (sketch gradient)** with **Transport KL-proximal gradient + momentum (sketch gradient)** at 3000 and 6224 cells per view. Each method gets 12 tuning candidates, three development seeds and both product/randomized starts, with 60 solver seconds per run. Selections use the median final float64 rounded loss across all six development cells, and any invalid cell disqualifies its candidate. The shared estimator, q=128 and momentum schedule rho(k)=0.3/(k+1)^0.5 stay fixed.

Final comparisons use five fresh seeds 200–204, both starting conditions and 300 solver seconds for both methods. Each two-method pair runs sequentially on the same A100 allocation; method order alternates. The pilot's seed 100 does not enter final summaries. Inner stopping failures remain visible. This is independent, equal-budget tuning of the actual momentum methods; it supersedes the original inherited-KL tuning policy.

Nominal solver time is **8.13 GPU-hours**, excluding pilot, startup, exact diagnostics and final validation. Saved couplings are expected to occupy about 63 GB. GPU tasks request one A100, two CPUs, 32 GiB and a two-hour ceiling, with at most three campaign tasks running together. CPU selection/reporting have a one-hour ceiling; actual allocation accounting is separate from these solver budgets. Original study accounting remains separate.

Pilot 1968346 ended after 534 allocation seconds because environment validation exceeded its 180-second child timeout, before data preparation or either optimizer. That failure is preserved. Retry 1968437 allows 600 seconds with diagnostic stack dumps; a diagnostic stack trace located the startup delay inside PyTorch import; the retry subsequently completed. No inference about solver speed or numerical behavior is drawn from this startup failure. All GPU campaign jobs also cap environment validation at 600 seconds and save startup failure evidence.

The [prospective design](../extensions/bmmc_scaling_campaign_20260928/README.md) specifies candidates, selection, provenance, failures and interpretation. All 21 selection/saved-plan tests, syntax checks and the reporter's missing-output smoke passed. The staged source is `53cc4e9d853250e8719997c7c0073ad052432f30`; the unchanged optimizer runtime is `b5851980646be55a148b083a89cfbf6153dac77e`. Every transferred file checksum passed. [Machine-readable submission record](BMMC_SCALING_CAMPAIGN.json).

Remote output root: `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/bmmc_campaign_53cc4e9`. The final report is preserved under `reports/report.md` there and copied locally above. A partial or failed campaign is explicitly reported as incomplete; missing outputs are not replaced with zero or counted as successful.

These sizes are nested subsets of one fixed BMMC dataset, separately normalized by size. Five optimizer seeds measure stochastic variability on that dataset. Equal solver time is not equal oracle work, and the limited tuning grid does not establish globally optimal hyperparameters. A larger problem is a test of the proposed scaling advantage, not a guaranteed favorable outcome.
