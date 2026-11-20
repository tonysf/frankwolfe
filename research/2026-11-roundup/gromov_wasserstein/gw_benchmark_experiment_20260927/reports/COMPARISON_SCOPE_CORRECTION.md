# Correction to the stochastic benchmark's scientific interpretation

Recorded 2 October 2026 following the user's clarification of the intended comparison.

The completed momentum-KL comparisons are exploratory experiments against a custom comparator. They do not establish that the manuscript algorithm performs worse than a published stochastic algorithm run under its convergence theorem. Equal gradient estimators, momentum coefficients, tuning budgets and hardware did not establish that kind of comparison. The earlier presentation of this comparator as the primary scientific baseline was inappropriate for the requested study.

The recorded losses, timings, failures and provenance remain valid measurements of the recorded implementations. Preserve both favorable and unfavorable results as exploratory evidence; the correction concerns their interpretation and eligibility for the primary comparison. Downloaded, checksum-verified reports and frozen remote releases remain historical artifacts. This correction supersedes their primary-baseline designation wherever it appears.

## Classification of completed work

| Recorded experiment | Correct interpretation |
|---|---|
| `stochastic-kl-momentum`, including later independently tuned BMMC, SNARE and SHARE-seq studies | Custom stochastic entropic mirror descent with gradient EMA; exploratory comparator without a verified theorem covering its recorded schedules and inexact solves. |
| Practical large-data `smoothing-fw` runs | Manuscript algorithm with fixed theoretical exponents and tuned prefactors, permitted by the asymptotic requirements as clarified by the user. Current rounded GW loss remains an empirical performance metric. |
| Dedicated `a100_theory` study | Literal displayed FW schedules and every-iterate true smoothed-gap statistic, on the recorded small synthetic instance; no published stochastic competitor was run in this study. |

The user clarified that tuning FW prefactors is valid because its step-size requirements are asymptotic. Retain this tuning; do not require unit gamma/rho prefactors or rerun FW solely to restore them. The dedicated unit-prefactor study is one recorded setting, not a fairness requirement. Separately, the proof's statistic is an average of true pre-update smoothed FW gaps; current rounded GW loss is an empirical metric and should not be described as that statistic. See [the existing manuscript mapping](THEORY_MAPPING.md).

## Replacement primary baseline

Use stochastic entropic mirror descent from [Fatkhullin and He, AISTATS 2024](https://proceedings.mlr.press/v238/fatkhullin24a.html), equation (2), with the assumptions, step restrictions and output rule of Theorem 4.3. The entropy geometry is explicitly covered there. The mathematical GW specialization and implementation status are recorded in [the replacement specification](../extensions/published_stochastic_20261002/README.md).

MSBPG from Ding, Li and Toh is the closest algebraic relative of the old EMA comparator, but its theorem is not a retrospective certificate for those runs. Matching an algorithm family or an acronym is insufficient; the implemented schedules, geometry, inner solve and returned statistic must match the cited result.

The replacement study must use each published algorithm within its own theoretical conditions. Prefactor tuning is allowed for both methods within those conditions; forcing shared momentum or step rules is not required. Record each theorem's output statistic separately from common practical metrics such as feasible GW loss, oracle work and elapsed time. Those empirical metrics remain useful for the performance comparison. Primary eligibility is determined from the paper and assumptions before new outcomes are observed, never from which method wins. The old heuristic may remain a clearly labeled secondary ablation.

No replacement GPU results are available as of this correction. Do not reuse historical momentum-KL losses as results for the published baseline, or describe a reference component as an executed campaign.
