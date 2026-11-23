# Primary stochastic comparison

Compare **Smoothed row FW + momentum (sketch gradient)** with **Transport KL-proximal gradient + momentum (sketch gradient)**. Both use the same q=128 Rademacher sketch and momentum recurrence. There are 110 corresponding instance/seed/start cells, with 220 valid final outputs. This comparison was included before held-out execution; the presentation selects methods by oracle/momentum compatibility, not by which one wins.

[Matching audit](../STOCHASTIC_MATCH_AUDIT.md) · [method definitions](../METHOD_NAMES.md) · [complete original results](../stage_c_final_named/report.md)

Matching covers instance, initialization, stochastic seed, gradient estimator/batch, iteration-indexed momentum schedule and initialization convention, GPU model/application libraries/CUDA runtime/precision, output rounding and prescribed solver-time budget. Twenty-five pairs differ in Ruche OS/driver revisions and 64 ran on different hosts. The complete machine environment is not identical in every pair. The same iteration-indexed schedule does not imply equal momentum coefficients at a given wall time. Constraint handling and the associated step/smoothing parameters differ by algorithm.

**Scope limits:** equal solver time is not equal oracle work: KL inner iterations reduce the number of gradient calls performed within a time budget. The work-axis panels display the actual recorded probe counts; sparse time checkpoints do not establish equal-probe-budget performance and are not interpolated to pretend otherwise. Momentum-KL inherits the plain-KL step settings without an independent tuning grid; this is not a comparison of independently optimal hyperparameters.

Final rounded losses below are float64 evaluations of the saved outputs. Entries give median [Q1, Q3]. Product cases have 300-second budgets; randomized cases have 30-second budgets. Actual output times and current-checkpoint losses are in the plots and source JSON. Synthetic summaries weight each of five problems equally after within-problem medians; BMMC is one fixed dataset.

| Start | Data | Method | Final rounded GW loss |
|---|---|---|---:|
| Product | BMMC n=1000 | Smoothed row FW + momentum (sketch gradient) | 0.15303 [0.15276, 0.15307] |
| Product | BMMC n=1000 | Transport KL-proximal gradient + momentum (sketch gradient) | 0.15359 [0.15348, 0.1539] |
| Product | Clean synthetic n=512 | Smoothed row FW + momentum (sketch gradient) | 0.043765 [0.043634, 0.043767] |
| Product | Clean synthetic n=512 | Transport KL-proximal gradient + momentum (sketch gradient) | 0.036214 [0.036211, 0.036267] |
| Product | Noisy synthetic n=512 | Smoothed row FW + momentum (sketch gradient) | 0.040018 [0.039911, 0.040028] |
| Product | Noisy synthetic n=512 | Transport KL-proximal gradient + momentum (sketch gradient) | 0.033092 [0.033017, 0.03316] |
| Randomized | BMMC n=1000 | Smoothed row FW + momentum (sketch gradient) | 0.15596 [0.15595, 0.156] |
| Randomized | BMMC n=1000 | Transport KL-proximal gradient + momentum (sketch gradient) | 0.16165 [0.16153, 0.16279] |
| Randomized | Clean synthetic n=512 | Smoothed row FW + momentum (sketch gradient) | 0.043776 [0.043647, 0.043787] |
| Randomized | Clean synthetic n=512 | Transport KL-proximal gradient + momentum (sketch gradient) | 0.03778 [0.037671, 0.037845] |
| Randomized | Noisy synthetic n=512 | Smoothed row FW + momentum (sketch gradient) | 0.040035 [0.039921, 0.040037] |
| Randomized | Noisy synthetic n=512 | Transport KL-proximal gradient + momentum (sketch gradient) | 0.034551 [0.034487, 0.034569] |

At the recorded endpoints, momentum-KL gives lower median loss for both synthetic families under each start condition; smoothed row FW gives lower median loss on fixed BMMC under each start condition. This mixed, configuration-specific result is not a uniform winner, significance claim or oracle-complexity result.

| Method | Valid outputs | Runs with inner stopping failures | Known inner stopping failures | Warnings |
|---|---:|---:|---:|---:|
| Smoothed row FW + momentum (sketch gradient) | 110/110 | 0 | 0 | 0 |
| Transport KL-proximal gradient + momentum (sketch gradient) | 110/110 | 103 | 57996 | 0 |

A valid rounded output is not a certificate that every inner KL solve met its stopping tolerance. Exact-gradient methods are kept in the separate original diagnostic/context archive and are absent from these figures. Plain sketch-gradient KL, the independently tuned row-projection variant and native Sampled GW have additional estimator/momentum/tuning differences and are secondary comparisons.

Figures show checkpoint medians with horizontal and vertical IQRs. Synthetic IQRs are across problem-level medians; BMMC IQRs are across repeats on its one fixed dataset. Open markers denote incomplete checkpoint populations.

Figures:

- [final_product_quality.png](final_product_quality.png)
- [final_product_quality.pdf](final_product_quality.pdf)
- [final_product_feasibility.png](final_product_feasibility.png)
- [final_product_feasibility.pdf](final_product_feasibility.pdf)
- [final_randomized_quality.png](final_randomized_quality.png)
- [final_randomized_quality.pdf](final_randomized_quality.pdf)
- [final_randomized_feasibility.png](final_randomized_feasibility.png)
- [final_randomized_feasibility.pdf](final_randomized_feasibility.pdf)
