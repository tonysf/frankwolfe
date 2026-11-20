# Frozen Stage C parameter selection

All **1,164 prescribed development trials** completed, and all **1,164 actual final rounded outputs** passed validation. All 72 candidates were eligible. The full archive contains 13,422 checksum-verified files; its remote checksum manifests were identical before and after transfer. Source, manifest, initial-plan, receipt and coverage audits passed.

Selection was frozen before final manifests were generated or held-out optimization began. Each method uses the minimum flat median of final float64 rounded loss divided by the same instance’s canonical product-plan loss; exact score ties use the candidate ID. The score combines clean/noisy development instances and fixed BMMC for the primary methods. PG is tuned separately on n=128, so its score is not a same-instance ranking against the primary methods.

| Method | Candidate | Selected constants | Development score |
|---|---|---|---:|
| smoothing-fw | c08 | gamma0=1, rho0=0.3, mu_scale=10 | 0.9999257695 |
| full-kl | c06 | eta_power=0, eta=10, inner_tol=0.0001 | 0.01029233721 |
| stochastic-kl | c09 | eta_power=0, eta=10, inner_tol=0.0001 | 0.956526118 |
| native-sampled-gw | c08 | batch=128, epsilon=0.01 | 0.8195876492 |
| smoothing-row-projection | c10 | gamma0=1, rho0=1, eta_power=0.25, mu_scale=10, eta_coefficient=1 | 0.9992804392 |
| pg | c09 | eta_power=0, eta_coefficient=100, inner_tol=0.0001 | 0.004439106433 |

For smoothing, `mu_scale=m*mu0`. Row projection uses `eta0=eta_coefficient*mu0/n`; PG uses `eta0=eta_coefficient/n²`. Custom stochastic methods retain q=128. Exact-gradient smoothing inherits the selected smoothing constants unchanged. Momentum-KL inherits selected stochastic-KL settings and smoothing rho0=0.3 with fixed beta=1/2; it receives no additional tuning.

The tuning runs comprise 714 ordinary completions, 200 completions with custom inner stopping failures, and 250 completions with native warnings. There were 6,584 known custom inner stopping failures and 4,619 recorded warnings. Native inner counts remain unknown. The largest summed marginal-L1 residual of an actual rounded output was 3.2103473e-07. No time checkpoints were missing, and no native run ended before its budget.

Some selected constants lie on grid boundaries; these choices are winners of the prescribed finite grid, not globally best hyperparameters. The selected smoothing score is close to the product-plan score. These 10-second development observations do not replace the pending longer-budget held-out comparison.

**Final plan:** 253 product-start runs at 300 seconds, 330 randomized-start runs at 30 seconds, 30 small-PG comparison runs at 30 seconds, plus 55 product and 55 randomized momentum-KL ablations. Total: 723 runs and 29.125 nominal solver GPU-hours, with overhead measured separately.

Selection SHA256: `4cfa1ef2a2ea127ba33987378b7a89c1922cfc6bedd1a433696136641c616fe7`.

[Selection JSON](../configs/stage_c_selection.json) · [Full tuning report](stage_c_tuning/report.md) · [Candidate scores](stage_c_tuning/tuning_candidate_scores.png) · [Collection verification](../ruche_results/tuning_aaeb5fa_collection/collection_verification.json)
