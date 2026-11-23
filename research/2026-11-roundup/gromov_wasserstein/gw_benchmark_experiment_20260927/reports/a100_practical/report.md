# Initial 30-second A100 practical comparison

Three development instances and one stochastic seed per method, using prespecified settings and a common solver-time checkpoint grid. This is an initial practical pilot, not a replicated final study or a completed parameter-selection exercise.

Panel status: **complete_with_numerical_issues**. Successful run records: 18/18. Missing checkpoint observations: 0.

All points are current rounded outputs at measured times. Missing windows remain missing. A terminal native early stop appears only at its actual time; no trajectory is extended to the nominal budget.

| Instance / method | Status | Stop reason / issues |
|---|---|---|
| metric512_s1001 / smoothing-fw | completed | — |
| metric512_s1001 / smoothing-fw-exact | completed | — |
| metric512_s1001 / full-kl | completed_with_inner_failures | — |
| metric512_s1001 / stochastic-kl | completed_with_inner_failures | — |
| metric512_s1001 / native-sampled-gw | completed_with_warnings | requested solver budget reached |
| metric512_s1001 / fw | completed | — |
| metric512_noisy_s1001 / smoothing-fw | completed | — |
| metric512_noisy_s1001 / smoothing-fw-exact | completed | — |
| metric512_noisy_s1001 / full-kl | completed_with_inner_failures | — |
| metric512_noisy_s1001 / stochastic-kl | completed_with_inner_failures | — |
| metric512_noisy_s1001 / native-sampled-gw | completed_with_warnings | requested solver budget reached |
| metric512_noisy_s1001 / fw | completed | — |
| bmmc1000 / smoothing-fw | completed | — |
| bmmc1000 / smoothing-fw-exact | completed | — |
| bmmc1000 / full-kl | completed_with_inner_failures | — |
| bmmc1000 / stochastic-kl | completed | — |
| bmmc1000 / native-sampled-gw | completed_with_warnings | requested solver budget reached |
| bmmc1000 / fw | completed | — |

Checkpoint opportunities use one common requested grid. Actual observation times are reported without moving an improvement backward to a target time.

| Instance / method | Target 1s | Target 3s | Target 10s | Target 30s | Terminal solver s |
|---|---|---|---|---|---|
| metric512_s1001 / smoothing-fw | observed at 1.0008s, iteration 527 | observed at 3.0003s, iteration 1635 | observed at 10.002s, iteration 5521 | observed at 30.002s, iteration 16613 | 30.002 |
| metric512_s1001 / smoothing-fw-exact | observed at 1.0011s, iteration 750 | observed at 3.0001s, iteration 2310 | observed at 10s, iteration 7795 | observed at 30.001s, iteration 23496 | 30.001 |
| metric512_s1001 / full-kl | observed at 1.0018s, iteration 250 | observed at 3.0582s, iteration 485 | observed at 10.003s, iteration 1383 | observed at 30.001s, iteration 6515 | 30.001 |
| metric512_s1001 / stochastic-kl | observed at 1.0012s, iteration 231 | observed at 3.0041s, iteration 707 | observed at 10.005s, iteration 2033 | observed at 30.008s, iteration 4683 | 30.008 |
| metric512_s1001 / native-sampled-gw | observed at 1.1134s, iteration 2 | observed at 3.177s, iteration 7 | observed at 10.204s, iteration 25 | observed at 30.331s, iteration 65 | 30.331 |
| metric512_s1001 / fw | observed at 1.0006s, iteration 234 | observed at 3.002s, iteration 805 | observed at 10s, iteration 2783 | observed at 30.002s, iteration 8450 | 30.002 |
| metric512_noisy_s1001 / smoothing-fw | observed at 1.0004s, iteration 528 | observed at 3.0004s, iteration 1642 | observed at 10.001s, iteration 5545 | observed at 30.001s, iteration 16712 | 30.001 |
| metric512_noisy_s1001 / smoothing-fw-exact | observed at 1.0003s, iteration 745 | observed at 3.001s, iteration 2307 | observed at 10.001s, iteration 7777 | observed at 30s, iteration 23440 | 30 |
| metric512_noisy_s1001 / full-kl | observed at 1.0021s, iteration 273 | observed at 3.0368s, iteration 513 | observed at 10.002s, iteration 1501 | observed at 30.001s, iteration 7052 | 30.001 |
| metric512_noisy_s1001 / stochastic-kl | observed at 1.0003s, iteration 216 | observed at 3.0037s, iteration 662 | observed at 10.002s, iteration 1942 | observed at 30.007s, iteration 4492 | 30.007 |
| metric512_noisy_s1001 / native-sampled-gw | observed at 1.1738s, iteration 4 | observed at 3.2637s, iteration 9 | observed at 10.108s, iteration 24 | observed at 30.409s, iteration 64 | 30.409 |
| metric512_noisy_s1001 / fw | observed at 1.005s, iteration 134 | observed at 3.0036s, iteration 458 | observed at 10.001s, iteration 1593 | observed at 30.004s, iteration 4839 | 30.004 |
| bmmc1000 / smoothing-fw | observed at 1.0006s, iteration 420 | observed at 3.0014s, iteration 1491 | observed at 10.001s, iteration 5252 | observed at 30.001s, iteration 16006 | 30.001 |
| bmmc1000 / smoothing-fw-exact | observed at 1.0011s, iteration 638 | observed at 3.0008s, iteration 2104 | observed at 10s, iteration 7252 | observed at 30.001s, iteration 21962 | 30.001 |
| bmmc1000 / full-kl | observed at 1.0169s, iteration 47 | observed at 3.0426s, iteration 81 | observed at 10.081s, iteration 145 | observed at 30.021s, iteration 245 | 30.021 |
| bmmc1000 / stochastic-kl | observed at 1.0154s, iteration 78 | observed at 3.0131s, iteration 182 | observed at 10.021s, iteration 442 | observed at 30.036s, iteration 894 | 30.036 |
| bmmc1000 / native-sampled-gw | observed at 1.0412s, iteration 3 | observed at 3.0367s, iteration 8 | observed at 10.21s, iteration 22 | observed at 30.006s, iteration 60 | 30.006 |
| bmmc1000 / fw | observed at 1.2549s, iteration 1 | observed at 3.0647s, iteration 5 | observed at 10.038s, iteration 24 | observed at 30.416s, iteration 78 | 30.416 |

Final losses use the actual saved raw and rounded plans promoted to float64, without a second repair. Raw smooth loss is not a feasible objective. Feasibility is reported for the rounded output to arithmetic tolerance.

| Instance / method | Raw loss64 | Rounded loss64 | Raw row L1 | Raw column L1 | Raw column L2 | Rounded row L1 | Rounded column L1 |
|---|---|---|---|---|---|---|---|
| metric512_s1001 / smoothing-fw | 0.043866 | 0.043857 | 1.1243e-07 | 0.084961 | 0.0043861 | 2.6404e-08 | 9.5734e-08 |
| metric512_s1001 / smoothing-fw-exact | 0.043869 | 0.043858 | 2.3758e-07 | 0.067417 | 0.0034415 | 2.2922e-08 | 1.996e-07 |
| metric512_s1001 / full-kl | 3.1086e-15 | 3.1086e-15 | 0 | 0 | 0 | 0 | 0 |
| metric512_s1001 / stochastic-kl | 0.043551 | 0.043551 | 5.1218e-07 | 2.842e-07 | 1.6436e-08 | 3.0254e-08 | 3.3891e-08 |
| metric512_s1001 / native-sampled-gw | 0.036433 | 0.036434 | 5.1926e-08 | 5.4823e-05 | 4.6812e-06 | 2.8678e-08 | 2.6534e-08 |
| metric512_s1001 / fw | 3.1086e-15 | 3.1086e-15 | 0 | 0 | 0 | 0 | 0 |
| metric512_noisy_s1001 / smoothing-fw | 0.040127 | 0.040105 | 1.4712e-07 | 0.083787 | 0.0043277 | 2.79e-08 | 1.0172e-07 |
| metric512_noisy_s1001 / smoothing-fw-exact | 0.040117 | 0.040106 | 2.6333e-07 | 0.06753 | 0.0034473 | 8.8476e-09 | 1.8859e-07 |
| metric512_noisy_s1001 / full-kl | 0.000416 | 0.000416 | 0 | 0 | 0 | 0 | 0 |
| metric512_noisy_s1001 / stochastic-kl | 0.03992 | 0.03992 | 4.707e-07 | 2.7255e-07 | 1.5945e-08 | 2.9562e-08 | 3.6183e-08 |
| metric512_noisy_s1001 / native-sampled-gw | 0.03318 | 0.03318 | 5.0399e-08 | 5.1393e-05 | 4.6854e-06 | 2.742e-08 | 2.7183e-08 |
| metric512_noisy_s1001 / fw | 0.032692 | 0.032692 | 0 | 0 | 0 | 0 | 0 |
| bmmc1000 / smoothing-fw | 0.16732 | 0.16753 | 2.7988e-07 | 0.12784 | 0.0048739 | 7.402e-08 | 1.0939e-07 |
| bmmc1000 / smoothing-fw-exact | 0.15525 | 0.15588 | 8.4197e-07 | 0.036129 | 0.0014862 | 7.3801e-08 | 8.0613e-08 |
| bmmc1000 / full-kl | 0.15158 | 0.15158 | 7.1134e-07 | 1.7808e-07 | 6.95e-09 | 6.3036e-08 | 6.4406e-08 |
| bmmc1000 / stochastic-kl | 0.15993 | 0.15993 | 6.8962e-07 | 2.497e-07 | 9.9489e-09 | 6.6129e-08 | 6.957e-08 |
| bmmc1000 / native-sampled-gw | 0.15173 | 0.15173 | 6.7616e-08 | 6.7974e-08 | 2.533e-09 | 8.2787e-08 | 8.2842e-08 |
| bmmc1000 / fw | 0.15102 | 0.15102 | 4.7497e-08 | 4.7497e-08 | 1.502e-09 | 4.7497e-08 | 4.7497e-08 |

Recorded method parameters are development settings, not selected winners. Native Sampled GW epsilon and conditional-sample counts have their own meanings; they are not equivalent to KL step sizes or trace-sketch probe counts.

| Instance / method | Parameters |
|---|---|
| metric512_s1001 / smoothing-fw | {"batch": 128, "gamma0": 1.0, "m_mu0": 1.0, "momentum_exponent": 0.5, "mu0": 0.001953125, "oracle": "sketch", "rho0": 1.0, "smoothing_exponent": 0.25, "step_exponent": 0.75} |
| metric512_s1001 / smoothing-fw-exact | {"batch": null, "gamma0": 1.0, "m_mu0": 1.0, "momentum_exponent": 0.5, "mu0": 0.001953125, "oracle": "exact", "rho0": 1.0, "smoothing_exponent": 0.25, "step_exponent": 0.75} |
| metric512_s1001 / full-kl | {"batch": null, "eta": 10.0, "eta_power": 0.0, "inner_max": 1000, "oracle": "exact", "tolerance_marginal_l1": 1e-06} |
| metric512_s1001 / stochastic-kl | {"batch": 128, "eta": 1.0, "eta_power": 0.0, "inner_max": 1000, "oracle": "sketch", "tolerance_marginal_l1": 1e-06} |
| metric512_s1001 / native-sampled-gw | {"epsilon": 0.1, "max_iter": 1000000, "native_inner_settings_unchanged": true, "native_samples": 128, "sample_semantics": "native conditional samples; not comparable to trace-sketch probes"} |
| metric512_s1001 / fw | {"line_search": "exact quadratic", "lmo": "exact SciPy assignment"} |
| metric512_noisy_s1001 / smoothing-fw | {"batch": 128, "gamma0": 1.0, "m_mu0": 1.0, "momentum_exponent": 0.5, "mu0": 0.001953125, "oracle": "sketch", "rho0": 1.0, "smoothing_exponent": 0.25, "step_exponent": 0.75} |
| metric512_noisy_s1001 / smoothing-fw-exact | {"batch": null, "gamma0": 1.0, "m_mu0": 1.0, "momentum_exponent": 0.5, "mu0": 0.001953125, "oracle": "exact", "rho0": 1.0, "smoothing_exponent": 0.25, "step_exponent": 0.75} |
| metric512_noisy_s1001 / full-kl | {"batch": null, "eta": 10.0, "eta_power": 0.0, "inner_max": 1000, "oracle": "exact", "tolerance_marginal_l1": 1e-06} |
| metric512_noisy_s1001 / stochastic-kl | {"batch": 128, "eta": 1.0, "eta_power": 0.0, "inner_max": 1000, "oracle": "sketch", "tolerance_marginal_l1": 1e-06} |
| metric512_noisy_s1001 / native-sampled-gw | {"epsilon": 0.1, "max_iter": 1000000, "native_inner_settings_unchanged": true, "native_samples": 128, "sample_semantics": "native conditional samples; not comparable to trace-sketch probes"} |
| metric512_noisy_s1001 / fw | {"line_search": "exact quadratic", "lmo": "exact SciPy assignment"} |
| bmmc1000 / smoothing-fw | {"batch": 128, "gamma0": 1.0, "m_mu0": 1.0, "momentum_exponent": 0.5, "mu0": 0.001, "oracle": "sketch", "rho0": 1.0, "smoothing_exponent": 0.25, "step_exponent": 0.75} |
| bmmc1000 / smoothing-fw-exact | {"batch": null, "gamma0": 1.0, "m_mu0": 1.0, "momentum_exponent": 0.5, "mu0": 0.001, "oracle": "exact", "rho0": 1.0, "smoothing_exponent": 0.25, "step_exponent": 0.75} |
| bmmc1000 / full-kl | {"batch": null, "eta": 10.0, "eta_power": 0.0, "inner_max": 1000, "oracle": "exact", "tolerance_marginal_l1": 1e-06} |
| bmmc1000 / stochastic-kl | {"batch": 128, "eta": 1.0, "eta_power": 0.0, "inner_max": 1000, "oracle": "sketch", "tolerance_marginal_l1": 1e-06} |
| bmmc1000 / native-sampled-gw | {"epsilon": 0.1, "max_iter": 1000000, "native_inner_settings_unchanged": true, "native_samples": 128, "sample_semantics": "native conditional samples; not comparable to trace-sketch probes"} |
| bmmc1000 / fw | {"line_search": "exact quadratic", "lmo": "exact SciPy assignment"} |

Time components in seconds. The quality axis includes solver work plus the repair for that output. Diagnostics, float64 validation, serialization, and process overhead are reported separately; no uncounted best-output selection is performed.

| Instance / method | Prepare | Native instrumentation prep | Solver | All repairs | Diagnostics | Validation64 | Serialization | Other overhead | Total instrumented |
|---|---|---|---|---|---|---|---|---|---|
| metric512_s1001 / smoothing-fw | 0.87856 | — | 30.002 | 0.02575 | 2.499 | 0.0086432 | 0.01172 | 0.16198 | 33.587 |
| metric512_s1001 / smoothing-fw-exact | 0.77776 | — | 30.001 | 0.026773 | 1.1445 | 0.0121 | 0.012731 | 0.22265 | 32.197 |
| metric512_s1001 / full-kl | 0.86113 | — | 30.001 | 0.026762 | 0.65724 | 0.011065 | 0.053685 | 0.063131 | 31.674 |
| metric512_s1001 / stochastic-kl | 0.83339 | — | 30.008 | 0.026299 | 0.92041 | 0.0083759 | 0.012806 | 0.046237 | 31.856 |
| metric512_s1001 / native-sampled-gw | 1.0212 | 0.058035 | 30.331 | 0.025115 | 0.44057 | 0.0083577 | 0.015394 | 0.025877 | 31.925 |
| metric512_s1001 / fw | 0.7573 | — | 30.002 | 0.024786 | 0.51067 | 0.0075395 | 0.012744 | 0.082221 | 31.398 |
| metric512_noisy_s1001 / smoothing-fw | 0.81952 | — | 30.001 | 0.02561 | 1.3067 | 0.0085014 | 0.01268 | 0.15746 | 32.331 |
| metric512_noisy_s1001 / smoothing-fw-exact | 0.93622 | — | 30 | 0.027446 | 0.96725 | 0.011101 | 0.010919 | 0.22473 | 32.178 |
| metric512_noisy_s1001 / full-kl | 0.81954 | — | 30.001 | 0.025494 | 0.61831 | 0.0074687 | 0.059075 | 0.068299 | 31.599 |
| metric512_noisy_s1001 / stochastic-kl | 0.80644 | — | 30.007 | 0.027288 | 0.89683 | 0.01105 | 0.011297 | 0.044427 | 31.804 |
| metric512_noisy_s1001 / native-sampled-gw | 0.91456 | 0.024516 | 30.409 | 0.025347 | 0.36953 | 0.0088876 | 0.014144 | 0.023626 | 31.79 |
| metric512_noisy_s1001 / fw | 0.81276 | — | 30.004 | 0.024593 | 0.49907 | 0.011675 | 0.01195 | 0.050305 | 31.415 |
| bmmc1000 / smoothing-fw | 1.0007 | — | 30.001 | 0.025747 | 0.037845 | 0.071092 | 0.01801 | 0.15447 | 31.309 |
| bmmc1000 / smoothing-fw-exact | 0.77825 | — | 30.001 | 0.026577 | 0.036552 | 0.025633 | 0.020393 | 0.21099 | 31.099 |
| bmmc1000 / full-kl | 0.86949 | — | 30.021 | 0.025705 | 0.055421 | 0.018568 | 0.021208 | 0.0031558 | 31.014 |
| bmmc1000 / stochastic-kl | 0.84988 | — | 30.036 | 0.026225 | 0.055726 | 0.012978 | 0.030973 | 0.0103 | 31.022 |
| bmmc1000 / native-sampled-gw | 0.88135 | 0.033705 | 30.006 | 0.02615 | 0.068137 | 0.021808 | 0.020999 | 0.024484 | 31.083 |
| bmmc1000 / fw | 0.79835 | — | 30.416 | 0.024591 | 0.056485 | 0.015735 | 0.018198 | 0.001612 | 31.331 |

Native snapshot/callback instrumentation is outside its solver clock. The separate native callback duration overlaps repairs and diagnostics and must not be added to those components again; the full instrumentation record is preserved in report.json.

Peak memory covers the instrumented process, including diagnostic and validation work. Unavailable native counters are blank rather than inferred from another method's sample definition.

| Instance / method | RSS MiB | GPU allocated MiB | GPU reserved MiB | Updates | Oracle calls | Sketch probes | Inner iterations | Inner failures | Underflow steps | Max underflow entries |
|---|---|---|---|---|---|---|---|---|---|---|
| metric512_s1001 / smoothing-fw | 998.57 | 55.02 | 66 | 16613 | 16613 | 2.1265e+06 | 0 | 0 | 0 | 0 |
| metric512_s1001 / smoothing-fw-exact | 999.7 | 55.02 | 66 | 23496 | 23496 | 0 | 0 | 0 | 0 | 0 |
| metric512_s1001 / full-kl | 997.17 | 56.02 | 66 | 6515 | 6515 | 0 | 83320 | 29 | 5925 | 2.6163e+05 |
| metric512_s1001 / stochastic-kl | 1004.3 | 56.02 | 66 | 4683 | 4683 | 5.9942e+05 | 91620 | 34 | 0 | 0 |
| metric512_s1001 / native-sampled-gw | 1048.8 | 55.021 | 70 | 65 | — | — | — | — | — | — |
| metric512_s1001 / fw | 996.92 | 57.024 | 68 | 8450 | 8450 | 0 | 0 | 0 | 0 | 0 |
| metric512_noisy_s1001 / smoothing-fw | 1001.4 | 55.02 | 66 | 16712 | 16712 | 2.1391e+06 | 0 | 0 | 0 | 0 |
| metric512_noisy_s1001 / smoothing-fw-exact | 985.89 | 55.02 | 66 | 23440 | 23440 | 0 | 0 | 0 | 0 | 0 |
| metric512_noisy_s1001 / full-kl | 1011.2 | 56.02 | 66 | 7052 | 7052 | 0 | 90560 | 65 | 6407 | 2.6163e+05 |
| metric512_noisy_s1001 / stochastic-kl | 1014.3 | 56.02 | 66 | 4492 | 4492 | 5.7498e+05 | 84250 | 38 | 0 | 0 |
| metric512_noisy_s1001 / native-sampled-gw | 1051.8 | 55.021 | 70 | 64 | — | — | — | — | — | — |
| metric512_noisy_s1001 / fw | 996.46 | 57.024 | 68 | 4839 | 4839 | 0 | 0 | 0 | 0 | 0 |
| bmmc1000 / smoothing-fw | 982.62 | 121.63 | 156 | 16006 | 16006 | 2.0488e+06 | 0 | 0 | 0 | 0 |
| bmmc1000 / smoothing-fw-exact | 957.67 | 121.63 | 156 | 21962 | 21962 | 0 | 0 | 0 | 0 | 0 |
| bmmc1000 / full-kl | 891.25 | 124.52 | 156 | 245 | 245 | 0 | 1.1736e+05 | 24 | 192 | 2.601e+05 |
| bmmc1000 / stochastic-kl | 909.17 | 124.52 | 176 | 894 | 894 | 1.1443e+05 | 1.199e+05 | 0 | 821 | 18697 |
| bmmc1000 / native-sampled-gw | 916.62 | 120.71 | 156 | 60 | — | — | — | — | — | — |
| bmmc1000 / fw | 1015 | 129.27 | 156 | 78 | 78 | 0 | 0 | 0 | 0 | 0 |

Every recorded failure/warning event is preserved in report.json, including events between checkpoints. The table groups identical event types/messages to keep this report readable.

| Instance / method | Event | Count | First iteration | Last iteration | Details |
|---|---|---|---|---|---|
| metric512_s1001 / full-kl | inner_tolerance_not_attained | 29 | 436 | 539 | Full event records are preserved in report.json |
| metric512_s1001 / full-kl | exponentiation_underflow_logstate_retained | 5925 | 591 | 6515 | Full event records are preserved in report.json |
| metric512_s1001 / stochastic-kl | inner_tolerance_not_attained | 34 | 1170 | 4671 | Full event records are preserved in report.json |
| metric512_s1001 / native-sampled-gw | native warning | 59 | — | — | UserWarning: Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`. |
| metric512_noisy_s1001 / full-kl | inner_tolerance_not_attained | 65 | 457 | 597 | Full event records are preserved in report.json |
| metric512_noisy_s1001 / full-kl | exponentiation_underflow_logstate_retained | 6407 | 646 | 7052 | Full event records are preserved in report.json |
| metric512_noisy_s1001 / stochastic-kl | inner_tolerance_not_attained | 38 | 1042 | 4478 | Full event records are preserved in report.json |
| metric512_noisy_s1001 / native-sampled-gw | native warning | 58 | — | — | UserWarning: Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`. |
| bmmc1000 / full-kl | inner_tolerance_not_attained | 24 | 14 | 124 | Full event records are preserved in report.json |
| bmmc1000 / full-kl | exponentiation_underflow_logstate_retained | 192 | 54 | 245 | Full event records are preserved in report.json |
| bmmc1000 / stochastic-kl | exponentiation_underflow_logstate_retained | 821 | 72 | 894 | Full event records are preserved in report.json |
| bmmc1000 / native-sampled-gw | native warning | 49 | — | — | UserWarning: Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`. |

This single-seed development pilot supplies no error bars, fitted rates, tuned algorithm ranking, global-optimality claim, or GPU/CPU speedup conclusion. Current feasible quality and raw feasibility remain separate. The BMMC instance receives no biological interpretation. No winner is selected.

Figures:

- [current_rounded_objective.png](current_rounded_objective.png)
- [current_rounded_objective.pdf](current_rounded_objective.pdf)
- [rounded_original_fw_gap.png](rounded_original_fw_gap.png)
- [rounded_original_fw_gap.pdf](rounded_original_fw_gap.pdf)
- [raw_feasibility.png](raw_feasibility.png)
- [raw_feasibility.pdf](raw_feasibility.pdf)
