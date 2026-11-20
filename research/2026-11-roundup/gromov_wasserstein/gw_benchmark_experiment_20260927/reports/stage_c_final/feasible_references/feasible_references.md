# Feasible references

Descriptive final-output references only; no recomputation, repair, checkpoint search, parameter tuning or optimizer selection.

Source scope: campaign. Input report SHA256: `39730329a43e14f0ce3af063f8263bb5b6b45e6acc45696ccbff164813d9086c`.

Clean synthetic instances retain known f*=0 exactly. Their best observed values are recorded separately, including negative cancellation. Other references are observed feasible values, not certified optima. Mixed budgets remain explicit and do not define a fair optimizer ranking.

| Instance | Reference kind | Reference value | Best observed float64 loss | Eligible / expected | Budgets (s) | Source run | Source phase / method / budget (s) | Source solver / output time (s) | Rounded both-L1 |
|---|---|---:|---:|---:|---|---|---|---:|---:|
| bmmc1000.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.14992996986905682 | 0.14992996986905682 | 59/63 | 30, 300 | final_randomized/full-kl/c06/bmmc1000/s102_i900102 | final_randomized / full-kl / 30 | 30.023742658202536 / 30.024406377226114 | 1.2465284978595108e-07 |
| metric128_noisy_s2000.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041039918206919879 | 0.00041039918206919879 | 3/3 | 30 | small_pg/full-kl/c06/metric128_noisy_s2000/s100_i100 | small_pg / full-kl / 30 | 30.002788945683278 / 30.003400563611649 | 0 |
| metric128_noisy_s2001.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041641767967504073 | 0.00041641767967504073 | 3/3 | 30 | small_pg/full-kl/c06/metric128_noisy_s2001/s100_i100 | small_pg / full-kl / 30 | 30.002486045472324 / 30.003097997396253 | 0 |
| metric128_noisy_s2002.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00042104325500758044 | 0.00042104325500758044 | 3/3 | 30 | small_pg/full-kl/c06/metric128_noisy_s2002/s100_i100 | small_pg / full-kl / 30 | 30.000335810473189 / 30.000938078504987 | 0 |
| metric128_noisy_s2003.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041783665828409866 | 0.00041783665828409866 | 3/3 | 30 | small_pg/full-kl/c06/metric128_noisy_s2003/s100_i100 | small_pg / full-kl / 30 | 30.000281044398434 / 30.000890470342711 | 0 |
| metric128_noisy_s2004.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041903417034605539 | 0.00041903417034605539 | 3/3 | 30 | small_pg/full-kl/c06/metric128_noisy_s2004/s100_i100 | small_pg / full-kl / 30 | 30.003789794398472 / 30.004446599399671 | 0 |
| metric128_s2000.npz | known_noiseless_optimum | 0 | 2.2204460492503131e-15 | 3/3 | 30 | small_pg/full-kl/c06/metric128_s2000/s100_i100 | small_pg / full-kl / 30 | 30.003735326696187 / 30.00439427071251 | 0 |
| metric128_s2001.npz | known_noiseless_optimum | 0 | 1.9984014443252818e-15 | 3/3 | 30 | small_pg/full-kl/c06/metric128_s2001/s100_i100 | small_pg / full-kl / 30 | 30.002900094608776 / 30.003516910597682 | 0 |
| metric128_s2002.npz | known_noiseless_optimum | 0 | 1.9984014443252818e-15 | 3/3 | 30 | small_pg/full-kl/c06/metric128_s2002/s100_i100 | small_pg / full-kl / 30 | 30.000962528749369 / 30.001578337745741 | 0 |
| metric128_s2003.npz | known_noiseless_optimum | 0 | 1.7763568394002505e-15 | 3/3 | 30 | small_pg/full-kl/c06/metric128_s2003/s100_i100 | small_pg / full-kl / 30 | 30.00130152143538 / 30.001888889470138 | 0 |
| metric128_s2004.npz | known_noiseless_optimum | 0 | 1.9984014443252818e-15 | 3/3 | 30 | small_pg/full-kl/c06/metric128_s2004/s100_i100 | small_pg / full-kl / 30 | 30.002847742172889 / 30.003472673124634 | 0 |
| metric512_noisy_s2000.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.0004170625071502343 | 0.0004170625071502343 | 62/63 | 30, 300 | final_product/full-kl/c06/metric512_noisy_s2000/s100_i100 | final_product / full-kl / 300 | 300.0015405183658 / 300.00218510429841 | 0 |
| metric512_noisy_s2001.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041424927777011789 | 0.00041424927777011789 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_noisy_s2001/s100_i100 | final_product / full-kl / 300 | 300.00018411909696 / 300.00083992711734 | 0 |
| metric512_noisy_s2002.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041676930219392005 | 0.00041676930219392005 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_noisy_s2002/s100_i100 | final_product / full-kl / 300 | 300.00263842474669 / 300.00332210375927 | 0 |
| metric512_noisy_s2003.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041563493824625652 | 0.00041563493824625652 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_noisy_s2003/s100_i100 | final_product / full-kl / 300 | 300.00026087334845 / 300.00094959430862 | 0 |
| metric512_noisy_s2004.npz | descriptive_observed_feasible_reference_not_certified_optimum | 0.00041601614987962776 | 0.00041601614987962776 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_noisy_s2004/s100_i100 | final_product / full-kl / 300 | 300.00080896413419 / 300.0026146491291 | 0 |
| metric512_s2000.npz | known_noiseless_optimum | 0 | 3.1086244689504383e-15 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_s2000/s100_i100 | final_product / full-kl / 300 | 300.0023355971789 / 300.00298089010175 | 0 |
| metric512_s2001.npz | known_noiseless_optimum | 0 | 3.1086244689504383e-15 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_s2001/s100_i100 | final_product / full-kl / 300 | 300.00150589854456 / 300.00226280547213 | 0 |
| metric512_s2002.npz | known_noiseless_optimum | 0 | 3.1086244689504383e-15 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_s2002/s100_i100 | final_product / full-kl / 300 | 300.00124614161905 / 300.00192685658112 | 0 |
| metric512_s2003.npz | known_noiseless_optimum | 0 | 3.1086244689504383e-15 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_s2003/s100_i100 | final_product / full-kl / 300 | 300.00253927521408 / 300.00323426828254 | 0 |
| metric512_s2004.npz | known_noiseless_optimum | 0 | 3.1086244689504383e-15 | 63/63 | 30, 300 | final_product/full-kl/c06/metric512_s2004/s100_i100 | final_product / full-kl / 300 | 300.00104829517659 / 300.0017338671023 | 0 |

The minimum spans the explicitly listed included budgets and phases; it is not a matched-budget algorithm comparison.
Recorded output availability may precede final total solver time when a native epilogue returns the same previously repaired plan; both clocks are retained.
A lower independently evaluated feasible output requires a newly archived reference report; negative differences are never clipped.
