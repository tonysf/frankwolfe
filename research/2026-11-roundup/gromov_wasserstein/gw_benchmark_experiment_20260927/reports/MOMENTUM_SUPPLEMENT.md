# Prospectively frozen momentum-KL supplement

Protocol §8B asks for a separately named momentum-matched stochastic KL-prox variant after the plain estimator comparison. This comparator was absent from the original campaign. The [supplemental configuration](../configs/momentum_ablation.json) was frozen before the authoritative final selection and held-out execution. The original six tuning grids, tuning manifest, final budgets and generator remain byte-for-byte unchanged.

The variant inherits the selected stochastic-KL step size, step exponent, marginal tolerance and inner cap, and the selected smoothing momentum constant rho0. It uses the same fixed q=128 trace-sketch oracle, beta=1/2, first-gradient initialization and first-step reuse convention. It has no column penalty and receives no extra tuning. The composite identifier `kl_cXX_rho_cYY` records these two inherited choices; it is not a seventh tuned candidate grid or a theorem claim. The completed tuning selected stochastic-KL `c09` (eta=10, eta_power=0, inner_tol=0.0001, inner cap=1000) and smoothing `c08` (rho0=0.3), giving the frozen supplemental identifier `kl_c09_rho_c08`. [Selection evidence](STAGE_C_SELECTION.md).

| Supplemental phase | Base comparison phase | Coverage | Solver budget | Checkpoints |
|---|---|---:|---:|---|
| momentum_product | final_product | 55 runs: 10 held-out synthetic datasets plus one fixed BMMC dataset, five stochastic seeds each | 300 s/run | 1,3,10,30,100,300 s |
| momentum_randomized | final_randomized | Same 55 instance/seed cells with the five matched independent initialization seeds | 30 s/run | 1,3,10,30 s |

The extra nominal solver budget is 18,150 seconds (5.0417 GPU-hours), plus normal diagnostic and execution overhead. Base-panel budgets remain unchanged. Keep the supplemental phase and provenance in reporting while using `comparison_phase` to show its curve alongside the corresponding base panel.

[prepare_momentum_ablation.py](../prepare_momentum_ablation.py) requires the authoritative selection, exact tuning manifest, and already frozen base product/randomized manifests. It verifies the base campaign/tuning/selection hashes and selected eligible tuning audit rows, validates every inherited stochastic-KL cell against the original prospective generator, and clones the instance hashes, parameters, seed pair, initialization, budget and checkpoint grid. Only method identity and momentum are added. No dataset is opened. Missing selections, stale provenance, incomplete or altered base cells, and existing output files are rejected.

The production manifests are frozen as [momentum_product.json](../configs/momentum_final_manifests/momentum_product.json) and [momentum_randomized.json](../configs/momentum_final_manifests/momentum_randomized.json). Their 110 runs were submitted with the 613 base final runs in **array 1961580**, using runtime source `af7b7c549351b81de2aa886b987bc3e56ffeff9c`. **Submission is complete; optimization completion and final validation are pending.** [Submission record](final_submission.json).

To reproduce the manifests into a new output directory using the frozen inputs:

```bash
.venv/bin/python prepare_momentum_ablation.py \
  --selected configs/stage_c_selection.json \
  --tuning-manifest configs/stage_c_manifests/tuning.json \
  --base-product-manifest configs/stage_c_final_manifests/final_product.json \
  --base-randomized-manifest configs/stage_c_final_manifests/final_randomized.json \
  --out-dir PATH/new_supplemental_manifests
```

The [production audit](final_manifest_audit/production_selection.json) independently recomputed all 72 candidate scores from the 1,164 validated tuning outputs and confirmed the six selected minima. It checked all 723 final cells, including exact inheritance for all 110 momentum cells, without reading held-out arrays or running optimizers. The tuning archive contains 13,422 checksum-verified files with no omissions or initialization/provenance issues. Fifteen focused generator tests also pass with dataset reads prohibited: 55+55 matching, budget/seed/parameter inheritance, untouched source documents, real-runner CLI compatibility, and stale/ineligible/missing-cell rejection. [Tests](../test_momentum_ablation_manifest.py).

The preceding independent [base-manifest audit](final_manifest_audit/result.json) exercised all 12 candidate IDs per tunable method through 12 fabricated in-memory selection scenarios, checked 7,356 specification cells, and confirmed base counts 253/330/30 and disjoint synthetic development/final instance names. It read no datasets and generated no real final manifests.

The larger-instance Stage D gate remains provisionally closed pending completion and validation of the selected final comparison. The submitted manifests and earlier smoke checks do not constitute final performance evidence.
