# Stage A baseline validation

CPU n=16, five outer steps; settings prespecified, not selected winners; no performance conclusion

Loss and gradients agree with the installed POT squared-loss conventions on three feasible plans: True

| Method | Status | Raw smooth loss | Rounded feasible loss | Raw marginal L1 | Rounded marginal L1 | Original FW gap after rounding |
|---|---|---:|---:|---:|---:|---:|
| reference_fw | passed | 0 | 0 | 0 | 0 | 0 |
| reference_kl | passed | 0.098558679 | 0.098558679 | 1.36e-11 | 1.6e-16 | 0.00754950154 |
| reference_stochastic_kl | passed | 0.100371953 | 0.100371953 | 4.09e-16 | 8.33e-17 | 0.00607739252 |
| reference_pg | passed | 0.100276292 | 0.100276292 | 1.04e-16 | 5.55e-17 | 0.00589294424 |
| pot_ppa | passed | 0.098558679 | 0.098558679 | 1.27e-12 | 7.63e-17 | 0.00754950154 |
| pot_sampled | passed | 0.0507231862 | 0.0507231864 | 2.56e-09 | 6.25e-17 | 0.0129016678 |
| pot_unregularized | passed | 0 | 0 | 0 | 0 | 0 |

These short CPU checks do not rank methods, select parameters, validate GPU behavior, or establish convergence rates.

Every output is independently evaluated with actual marginals. Raw and rounded arrays, configurations, native logs, warnings, and installed POT sources are retained.

The proposed method and theorem-specific checks are outside this baseline-only validation; see the separate method validation records.
