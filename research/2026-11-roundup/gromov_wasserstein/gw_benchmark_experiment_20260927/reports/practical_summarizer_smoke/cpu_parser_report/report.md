# Practical-run parser smoke using preserved CPU test outputs

This is an output-parser/plotting test using preserved small CPU runs and a separate fixture plan. These records are not A100 Stage C results.

Panel status: **complete_initial_pilot**. Successful run records: 4/4. Missing checkpoint observations: 0.

All points are current rounded outputs at measured times. Missing windows remain missing. A terminal native early stop appears only at its actual time; no trajectory is extended to the nominal budget.

| Instance / method | Status | Stop reason / issues |
|---|---|---|
| toy16 / smoothing-fw | completed | — |
| toy16 / full-kl | completed | — |
| toy16 / stochastic-kl | completed | — |
| toy16 / fw | completed | — |

Checkpoint opportunities use one common requested grid. Actual observation times are reported without moving an improvement backward to a target time.

| Instance / method | Target 0.01s | Target 0.03s | Terminal solver s |
|---|---|---|---|
| toy16 / smoothing-fw | observed at 0.010196s, iteration 29 | observed at 0.030253s, iteration 92 | 0.030253 |
| toy16 / full-kl | observed at 0.01011s, iteration 9 | observed at 0.030745s, iteration 28 | 0.030745 |
| toy16 / stochastic-kl | observed at 0.010919s, iteration 9 | observed at 0.030634s, iteration 26 | 0.030634 |
| toy16 / fw | observed at 0.010169s, iteration 33 | observed at 0.030189s, iteration 100 | 0.030189 |

Final losses use the actual saved raw and rounded plans promoted to float64, without a second repair. Raw smooth loss is not a feasible objective. Feasibility is reported for the rounded output to arithmetic tolerance.

| Instance / method | Raw loss64 | Rounded loss64 | Raw row L1 | Raw column L1 | Raw column L2 | Rounded row L1 | Rounded column L1 |
|---|---|---|---|---|---|---|---|
| toy16 / smoothing-fw | 0.10175 | 0.1004 | 5.5511e-17 | 0.15153 | 0.043809 | 0 | 1.6653e-16 |
| toy16 / full-kl | 0.10062 | 0.10062 | 2.498e-16 | 1.3878e-16 | 5.1926e-17 | 2.7756e-17 | 4.1633e-17 |
| toy16 / stochastic-kl | 0.10057 | 0.10057 | 3.1225e-16 | 1.3184e-16 | 4.6548e-17 | 2.0817e-17 | 9.0206e-17 |
| toy16 / fw | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Recorded method parameters are development settings, not selected winners. Native Sampled GW epsilon and conditional-sample counts have their own meanings; they are not equivalent to KL step sizes or trace-sketch probe counts.

| Instance / method | Parameters |
|---|---|
| toy16 / smoothing-fw | {"batch": 8, "gamma0": 1.0, "m_mu0": 1.0, "momentum_exponent": 0.5, "mu0": 0.0625, "oracle": "sketch", "rho0": 1.0, "smoothing_exponent": 0.25, "step_exponent": 0.75} |
| toy16 / full-kl | {"batch": null, "eta": 0.1, "eta_power": 0.0, "inner_max": 1000, "oracle": "exact", "tolerance_marginal_l1": 1e-09} |
| toy16 / stochastic-kl | {"batch": 8, "eta": 0.1, "eta_power": 0.0, "inner_max": 1000, "oracle": "sketch", "tolerance_marginal_l1": 1e-09} |
| toy16 / fw | {"line_search": "exact quadratic", "lmo": "exact SciPy assignment"} |

Time components in seconds. The quality axis includes solver work plus the repair for that output. Diagnostics, float64 validation, serialization, and process overhead are reported separately; no uncounted best-output selection is performed.

| Instance / method | Prepare | Native instrumentation prep | Solver | All repairs | Diagnostics | Validation64 | Serialization | Other overhead | Total instrumented |
|---|---|---|---|---|---|---|---|---|---|
| toy16 / smoothing-fw | 1.0003 | — | 0.030253 | 0.00073523 | 0.25915 | 0.00058805 | 0.001414 | 0.00039108 | 1.2928 |
| toy16 / full-kl | 0.015531 | — | 0.030745 | 0.00071352 | 0.0016873 | 0.0006274 | 0.0013201 | 0.00018354 | 0.050809 |
| toy16 / stochastic-kl | 0.015615 | — | 0.030634 | 0.00073254 | 0.0017569 | 0.00059904 | 0.0012413 | 0.00018079 | 0.05076 |
| toy16 / fw | 0.015795 | — | 0.030189 | 0.00069548 | 0.0016488 | 0.00060131 | 0.0012311 | 0.00030686 | 0.050468 |

Native snapshot/callback instrumentation is outside its solver clock. The separate native callback duration overlaps repairs and diagnostics and must not be added to those components again; the full instrumentation record is preserved in report.json.

Peak memory covers the instrumented process, including diagnostic and validation work. Unavailable native counters are blank rather than inferred from another method's sample definition.

| Instance / method | RSS MiB | GPU allocated MiB | GPU reserved MiB | Updates | Oracle calls | Sketch probes | Inner iterations | Inner failures | Underflow steps | Max underflow entries |
|---|---|---|---|---|---|---|---|---|---|---|
| toy16 / smoothing-fw | 357.34 | — | — | 92 | 92 | 736 | 0 | 0 | 0 | 0 |
| toy16 / full-kl | 360.07 | — | — | 28 | 28 | 0 | 280 | 0 | 0 | 0 |
| toy16 / stochastic-kl | 361.79 | — | — | 26 | 26 | 208 | 260 | 0 | 0 | 0 |
| toy16 / fw | 361.79 | — | — | 100 | 100 | 0 | 0 | 0 | 0 | 0 |

No failure/warning events were present in the available run records; missing run records cannot establish absence of failures.

This single-seed development pilot supplies no error bars, fitted rates, tuned algorithm ranking, global-optimality claim, or GPU/CPU speedup conclusion. Current feasible quality and raw feasibility remain separate. The BMMC instance receives no biological interpretation. No winner is selected.

Figures:

- [current_rounded_objective.png](current_rounded_objective.png)
- [current_rounded_objective.pdf](current_rounded_objective.pdf)
- [rounded_original_fw_gap.png](rounded_original_fw_gap.png)
- [rounded_original_fw_gap.pdf](rounded_original_fw_gap.pdf)
- [raw_feasibility.png](raw_feasibility.png)
- [raw_feasibility.pdf](raw_feasibility.pdf)
