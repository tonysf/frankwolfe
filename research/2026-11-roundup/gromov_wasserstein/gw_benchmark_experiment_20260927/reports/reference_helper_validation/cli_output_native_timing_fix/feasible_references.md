# Feasible references

Descriptive final-output references only; no recomputation, repair, checkpoint search, parameter tuning or optimizer selection.

Source scope: parser-smoke. Input report SHA256: `a1e4ebeac1b8735a3fd68f12407e439f66c6d5923d8eb242a9771387db32c5f4`.

Clean synthetic instances retain known f*=0 exactly. Their best observed values are recorded separately, including negative cancellation. Other references are observed feasible values, not certified optima. Mixed budgets remain explicit and do not define a fair optimizer ranking.

| Instance | Reference kind | Reference value | Best observed float64 loss | Eligible / expected | Budgets (s) | Source run | Source phase / method / budget (s) | Source solver / output time (s) | Rounded both-L1 |
|---|---|---:|---:|---:|---|---|---|---:|---:|
| toy16.npz | known_noiseless_optimum | 0 | 0 | 4/4 | 0.029999999999999999 | final_product/fw/c00/toy16/s100_i200 | final_product / fw / 0.029999999999999999 | 0.030189313925802708 / 0.03036495391279459 | 0 |

The minimum spans the explicitly listed included budgets and phases; it is not a matched-budget algorithm comparison.
Recorded output availability may precede final total solver time when a native epilogue returns the same previously repaired plan; both clocks are retained.
A lower independently evaluated feasible output requires a newly archived reference report; negative differences are never clipped.
