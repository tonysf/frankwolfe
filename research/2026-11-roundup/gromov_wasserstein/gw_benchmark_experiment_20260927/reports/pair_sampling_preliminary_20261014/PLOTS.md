# Preliminary pair-sampling plots

Three completed datasets, three methods and three stochastic seeds: **27/27 included runs**. SHARE-seq remains excluded.

[Download both figures as a two-page PDF](pair_sampling_preliminary.pdf)

[Feasible GW loss](feasible_gw_loss.png) · [PDF](feasible_gw_loss.pdf) · [Relative loss reduction](relative_loss_reduction.png) · [PDF](relative_loss_reduction.pdf)

Lines show the median and bands the full three-seed range, not confidence intervals. Curves preserve current logged outputs, without best-so-far selection.

**Clock:** solver time plus the current repair only. Diagnostics, preparation, logging and final validation are excluded. Actual final total elapsed time is listed below. Time curves linearly interpolate sparse checkpoints within each method’s three-seed overlap; they do not identify when changes occurred between checkpoints.

**Precision:** curves retain the online full-objective values. Hollow endpoints and this table use the independent final FP64 reevaluation. The endpoint CSV records every signed FP64-minus-online difference.

| Dataset | Method | Final FP64 loss, median [min, max] | Reduction from start, median (%) | Final output time, median (min) | Actual total elapsed, median (min) |
|---|---|---:|---:|---:|---:|
| Synthetic metric data | FraMES | 0.035093099 [2.4328537e-06, 0.035113631] | 20.03 | 154.8 | 155.3 |
| Synthetic metric data | Entropic SMD (fresh gradient) | 0.043876615 [0.043876614, 0.043876616] | 0.01583 | 198.8 | 199.2 |
| Synthetic metric data | MSBPG (theorem parameters) | 0.043883451 [0.043883451, 0.043883451] | 0.0002527 | 205.4 | 205.8 |
| BMMC | FraMES | 0.15840799 [0.1583286, 0.15845191] | 1.634 | 24.82 | 25.02 |
| BMMC | Entropic SMD (fresh gradient) | 0.15781056 [0.15781044, 0.1578139] | 2.005 | 72.35 | 72.55 |
| BMMC | MSBPG (theorem parameters) | 0.1610392 [0.1610392, 0.1610392] | 0.000278 | 99.9 | 100.1 |
| SNARE-seq | FraMES | 0.092188256 [0.091996492, 0.092216234] | 2.003 | 33.14 | 33.49 |
| SNARE-seq | Entropic SMD (fresh gradient) | 0.09313 [0.09312952, 0.093132083] | 1.002 | 143.7 | 144.1 |
| SNARE-seq | MSBPG (theorem parameters) | 0.094072151 [0.09407215, 0.094072151] | 0.0003617 | 222.9 | 223.4 |

Fixed-prefactor comparison: three stochastic methods use the same pair-sampling oracle and pair budget. Real-data prefactors were previously selected using Rademacher sketches; no retuning or loss-based stopping. This is not a ranking of independently retuned best configurations.
Independent sampled baseline output and last-rounded GW loss are separate. Neither loss establishes global optimality or stationarity. Fixed 1e-7 proximal accuracy has no claimed inexact-outer theorem.

Input report SHA256: `4d1b9ce9112194482745d0400467572335ec78308f0e5ce6261007f86dfbb5fe`.

Data: [curves](curve_data.csv), [observations](observations.csv), [endpoints](endpoints.csv), [provenance and summary](plot_summary.json).
