# Completed SNARE-seq and BMMC stochastic comparison

**Scientific interpretation corrected on 2 October 2026:** the completed momentum-KL comparisons are exploratory tests against a custom comparator. They do not supply the requested comparison against a published stochastic algorithm under its stated theory. Their measurements are retained; the primary-baseline designation is withdrawn. See [the correction](COMPARISON_SCOPE_CORRECTION.md). Replacement GPU results are not yet available.

**All 432 tuning runs, 60 final outputs and six component profiles completed and validated on Ruche. Transport KL-proximal gradient with momentum achieved lower feasible GW loss in all 30 matched final seed pairs at 300 solver seconds.**

The table gives median final float64 GW loss across five fresh seeds; lower is better. FW means **Smoothed row FW + momentum (sketch gradient)**; KL means **Transport KL-proximal gradient + momentum (sketch gradient)**. Both use the same q=128 gradient estimator and momentum schedule, with independent equal-budget tuning.

| Dataset and cells/view | Initialization | FW loss | KL loss | KL reduction relative to FW |
|---|---|---:|---:|---:|
| BMMC 1,000 | product | 0.153347 | 0.151063 | 1.49% |
| BMMC 1,000 | randomized | 0.153311 | 0.150872 | 1.59% |
| SNARE 3,000 | product | 0.093187 | 0.089069 | 4.42% |
| SNARE 3,000 | randomized | 0.093183 | 0.089154 | 4.32% |
| SNARE 10,000 | product | 0.093738 | 0.093465 | 0.29% |
| SNARE 10,000 | randomized | 0.093742 | 0.093479 | 0.28% |

FW leads on BMMC at the recorded 10- and 30-second checkpoints; KL leads by 100 seconds. On SNARE 3,000, FW has a small median lead at 10 seconds, while KL leads at 30/100/300 seconds. On SNARE 10,000, KL leads at all four recorded checkpoints. FW executes more outer updates, but this does not yield a lower final objective under the prescribed budget.

Independent tuning selected KL eta=.1 on BMMC and SNARE 10,000, and eta=1 on SNARE 3,000. The retuned BMMC result reverses the earlier comparison using inherited eta=10. Changed seeds and execution conditions mean this is not an isolated causal estimate of the step-size effect. These grids and budgets do not establish a universal method ranking.

Exact-cell matching on SNARE remains near or below the uniform/chance baseline for both methods: all eight SNARE method/start medians are at or below 1/n. The chance top1 baseline is 0.0333% at 3,000 and 0.01% at 10,000. Normalized true-match rank medians range about 0.5002–0.5539, compared with 0.5 for uniform weights. Lower GW loss therefore does not demonstrate successful recovery of the true cell pairing. No claim about cell-type transfer is made because it was not measured.

All final couplings passed feasibility and float64 validation. Eight final KL runs recorded intermediate inner stopping misses; these do not invalidate the saved outputs and remain reported. There are no missing final runs or infrastructure/provenance failures. All six separate component profiles passed their source/parameter checks; their extra synchronization means they are explanatory measurements rather than primary throughput estimates.

[Complete report and figures](cross_dataset_final/report.md), [verified collection and paired comparisons](cross_dataset_final/completion.json), and [frozen parameter selection](cross_dataset_final/selection.json). All 15 collected files matched their Ruche SHA-256 checksums. Dense coupling archives remain on Ruche. The final report was generated at `2026-09-30T03:18:11.864916+00:00`. FAUST remains deferred as requested.

## Preserved design and execution record

The completed study uses canonical BMMC n=1,000 and paired SNARE-seq adult cortex at n=3,000 and n=10,000. Of 10,309 intersecting cell barcodes, 10,301 passed the fixed joint QC. Preparation job 1975469 completed in 182 seconds. FAUST is deferred as the user requested; its licensed training archive is still needed.

Compare **Smoothed row FW + momentum (sketch gradient)** with **Transport KL-proximal gradient + momentum (sketch gradient)**. Both retain the same q=128 gradient estimator, momentum rule, theoretical FW exponents, precision and full unsampled GW objective. The underlying optimizer kernels are byte-identical to the earlier runtime. Independent tuning uses 12 candidates per method/instance, three development seeds and two starts, for **432 tuning runs**. Finals use five fresh seeds and two starts, for **60 runs at 300 solver seconds**, with 10/30/100/300-second checkpoints. Each two-method pair shares its allocation and initial plan.

New diagnostics report tie-aware correspondence accuracy and true-match ranks on SNARE, observed attainment of common prespecified loss targets, and separate synchronized component-profiling companions. BMMC's original independent cell subsets have no complete bijective correspondence, so its matching accuracy is not applicable. Labels never enter solver updates or selection. The profiler changes synchronization and does not supply primary timing rankings.

| Stage | Slurm job |
|---|---:|
| freeze | 1975487 |
| pilot | 1975488 |
| gate | 1975489 |
| tuning | 1975490 |
| select | 1975491 |
| final | 1975492 |
| profile | 1975493 |
| report | 1975494 |

The chain is configuration freeze→A100 feasibility/resource pilot→CPU eligibility audit→tuning→selection→finals→profiling→report. The report also runs after final/profile termination to preserve missing or failed coverage. Scientific/provenance/resource gates prevent the full campaign from starting after an invalid pilot. Arrays allow at most 3 concurrent A100 tasks. Nominal solver time is **12.2 GPU-hours**, excluding pilot, setup, diagnostics, validation and profiling. The coupling archive is estimated at 145 GB; available Ruche work storage was checked before submission.

All **61 active CPU tests** passed, plus 46 controller/report subtests; independent review covered update preservation, diagnostic noninterference, pilot gates and all stage interfaces. An additional 15 tests cover the deferred FAUST preparer. Source/controller revision `f93f8079fb9322b49d3de88ff2eabc167a9276aa`; new diagnostic runtime `ace667162dd3c82da2ff9712301f0019c9f63715`. The minimal 184,320-byte controller/runtime payload passed remote checksums and runtime clean-commit verification. Original results were not changed.

Preparation job 1975469 had aged out of Slurm's live dependency table by submission; the rejected attempt submitted no jobs and is preserved. Its completion and bound preprocessing manifest were independently verified, and the allocated freeze stage audits its outputs directly. All subsequent dependencies were accepted and verified in the initial scheduler snapshot.

[Prospective design](../extensions/cross_dataset_20260929/README.md), [SNARE sources and preparation](../extensions/cross_dataset_20260929/SNARE_DATA.md), [test/provenance summary](../extensions/cross_dataset_20260929/validation.json), [submission record](CROSS_DATASET_CAMPAIGN.json).

Remote output root: `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/cross_dataset_20260929/campaign_f93f807`. The final report is preserved at `reports/report.md` under that root and copied locally above. The actual dataset/preparation and run manifests, failures, source bindings and original dense outputs are retained there. No ranking is inferred from submission or a successful pilot.
