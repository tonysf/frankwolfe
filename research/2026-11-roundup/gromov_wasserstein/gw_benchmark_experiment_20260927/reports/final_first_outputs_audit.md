# First selected final outputs: bounded independent audit

**Passed: exactly three completed trials**, the first terminal output from array 1961580 tasks 0, 1, and 2. All are selected smoothing-FW c08 on the clean held-out n=512 instance `metric512_s2000`, with stochastic seeds 100, 101, and 102. This is an early integrity check, not completion of the 723-run study or a comparative ranking.

The separate preview contains 15 trial files with identical remote-before, remote-after, and local checksums, plus atomic ledger and dispatch-receipt snapshots. Source `af7b7c549351b81de2aa886b987bc3e56ffeff9c`, frozen selection, manifest/dispatch bindings, canonical input hashes, matched initial plans, A100 precision settings and all six checkpoint opportunities passed.

Saved raw and rounded float32 plans were promoted directly to NumPy float64 on one local CPU. The objective uses each plan’s actual row/column sums; neither plan was repaired or altered. All plans are finite and nonnegative. No optimizer ran, no planted correspondence was read, and no setting was changed.

| Seed | CPU raw loss | CPU rounded loss | Raw column L1 | Rounded combined marginal L1 | Solver / output time (s) | Max loss difference |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.0437658612734 | 0.0437675616036 | 0.0224009843 | 6.17525302e-08 | 300.001047 / 300.001833 | 2.22e-16 |
| 101 | 0.0437691661692 | 0.0437690559722 | 0.0224713866 | 6.11755127e-08 | 300.001786 / 300.002588 | 2.22e-16 |
| 102 | 0.0437701490082 | 0.0437670579484 | 0.0212170947 | 5.75510057e-08 | 300.000324 / 300.001052 | 4.44e-16 |

Every run is `completed`, reached its 300-second solver budget, and reported zero failure events, inner stopping failures, or underflow steps. Updates / stochastic gradients / q=128 probe totals respectively: seed 100: 160,857 / 160,857 / 20,589,696; seed 101: 160,225 / 160,225 / 20,508,800; seed 102: 169,572 / 169,572 / 21,705,216. Exact diagnostic work, timing components, memory and full counters are retained in the JSON.

CPU versus recorded CUDA float64 objective differences are at most 4.44e-16; every marginal residual differs by at most the 1e-12 audit tolerance. Raw infeasibility remains visible and is not presented as feasible quality.

[Full audit](final_first_outputs_audit.json) · [Transfer verification](../ruche_results/final_first_outputs_preview/transfer_verification.json) · [Reproducible audit source](../ruche_results/final_first_outputs_preview/audit_saved_plans.py)
