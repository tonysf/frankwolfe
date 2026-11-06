# Supplementary momentum replay report

**complete_verified**: 796/796 prescribed replays terminal; 796 verified, 0 terminal unverified. Provenance valid: True. This does not change original optimization status or establish full-study completion.

At original output K>=1 primary error pairs actual d_(K-1) with grad f(P_(K-1)); stale tracking uses grad f(P_K). K=0 uses d0/P0. No terminal oracle extension.

Verified observed checkpoints: 3714; explicitly missed original windows in verified runs: 4. Missing or rejected replays contribute no error curves.

Summed controller elapsed cost: 50104.8 seconds across 796 terminal replays. Worker totals and their nested subtotals are retained in JSON and are not added again. Original solver/output clocks remain the diagnostic observation labels.

| Replay phase | Prescribed | Terminal | Verified | Unverified |
|---|---:|---:|---:|---:|
| tuning | 504 | 504 | 504 | 0 |
| pilot | 6 | 6 | 6 | 0 |
| finals | 286 | 286 | 286 | 0 |

Synthetic plots: within-instance median over prescribed stochastic/initialization runs, then across-instance median/IQR. BMMC: fixed-instance stochastic/initialization median/IQR. Deterministic duplicate starts are not new replicates. Observed and prescribed run/instance counts and coverage_complete are saved per target, including empty windows. Open markers show incomplete populations; partial populations may differ.

Supplementary diagnostics only; no performance ranking, selection changes, retuning or theorem-rate claim. Original timing and optimization outcomes stay authoritative.

All prescribed records, failures and integrity issues are retained in [runs.csv](runs.csv) and [report.json](report.json).

Figures:
- [final_product_momentum_errors.png](final_product_momentum_errors.png)
- [final_product_momentum_errors.pdf](final_product_momentum_errors.pdf)
- [final_randomized_momentum_errors.png](final_randomized_momentum_errors.png)
- [final_randomized_momentum_errors.pdf](final_randomized_momentum_errors.pdf)

Window populations and statistics: [curve_summaries.csv](curve_summaries.csv).
