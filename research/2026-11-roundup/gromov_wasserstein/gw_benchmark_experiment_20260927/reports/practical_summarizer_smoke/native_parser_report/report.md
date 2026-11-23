# Practical-run parser smoke using preserved CPU test outputs

This is an output-parser/plotting test using preserved small CPU runs and a separate fixture plan. These records are not A100 Stage C results.

Panel status: **complete_with_numerical_issues**. Successful run records: 1/1. Missing checkpoint observations: 0.

All points are current rounded outputs at measured times. Missing windows remain missing. A terminal native early stop appears only at its actual time; no trajectory is extended to the nominal budget.

| Instance / method | Status | Stop reason / issues |
|---|---|---|
| toy16 / native-sampled-gw | completed_with_warnings | requested solver budget reached |

Checkpoint opportunities use one common requested grid. Actual observation times are reported without moving an improvement backward to a target time.

| Instance / method | Target 0.1s | Terminal solver s |
|---|---|---|
| toy16 / native-sampled-gw | observed at 0.16202s, iteration 2 | 0.16203 |

Final losses use the actual saved raw and rounded plans promoted to float64, without a second repair. Raw smooth loss is not a feasible objective. Feasibility is reported for the rounded output to arithmetic tolerance.

| Instance / method | Raw loss64 | Rounded loss64 | Raw row L1 | Raw column L1 | Raw column L2 | Rounded row L1 | Rounded column L1 |
|---|---|---|---|---|---|---|---|
| toy16 / native-sampled-gw | 0.071776 | 0.071776 | 5.6699e-08 | 5.5402e-08 | 1.7971e-08 | 2.3135e-08 | 3.0017e-08 |

Recorded method parameters are development settings, not selected winners. Native Sampled GW epsilon and conditional-sample counts have their own meanings; they are not equivalent to KL step sizes or trace-sketch probe counts.

| Instance / method | Parameters |
|---|---|
| toy16 / native-sampled-gw | {"epsilon": 0.1, "max_iter": 1000000, "native_inner_settings_unchanged": true, "native_samples": 8, "sample_semantics": "native conditional samples; not comparable to trace-sketch probes"} |

Time components in seconds. The quality axis includes solver work plus the repair for that output. Diagnostics, float64 validation, serialization, and process overhead are reported separately; no uncounted best-output selection is performed.

| Instance / method | Prepare | Native instrumentation prep | Solver | All repairs | Diagnostics | Validation64 | Serialization | Other overhead | Total instrumented |
|---|---|---|---|---|---|---|---|---|---|
| toy16 / native-sampled-gw | 0.016916 | 0.024403 | 0.16203 | 0.00058297 | 0.001564 | 0.00067894 | 0.0014759 | 0.0010362 | 0.20869 |

Native snapshot/callback instrumentation is outside its solver clock. The separate native callback duration overlaps repairs and diagnostics and must not be added to those components again; the full instrumentation record is preserved in report.json.

Peak memory covers the instrumented process, including diagnostic and validation work. Unavailable native counters are blank rather than inferred from another method's sample definition.

| Instance / method | RSS MiB | GPU allocated MiB | GPU reserved MiB | Updates | Oracle calls | Sketch probes | Inner iterations | Inner failures | Underflow steps | Max underflow entries |
|---|---|---|---|---|---|---|---|---|---|---|
| toy16 / native-sampled-gw | 403.74 | — | — | 2 | — | — | — | — | — | — |

All recorded failure/warning events are retained below, including failures between output checkpoints.

| Instance / method | Iteration | Event | Details |
|---|---|---|---|
| toy16 / native-sampled-gw | — | native warning | {"category": "UserWarning", "filename": "/home/tony/gw_benchmark_experiment_20260927/.venv/lib/python3.11/site-packages/ot/bregman/_sinkhorn.py", "lineno": 666, "message": "Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`."} |
| toy16 / native-sampled-gw | — | native warning | {"category": "UserWarning", "filename": "/home/tony/gw_benchmark_experiment_20260927/.venv/lib/python3.11/site-packages/ot/bregman/_sinkhorn.py", "lineno": 666, "message": "Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`."} |

This single-seed development pilot supplies no error bars, fitted rates, tuned algorithm ranking, global-optimality claim, or GPU/CPU speedup conclusion. Current feasible quality and raw feasibility remain separate. The BMMC instance receives no biological interpretation. No winner is selected.

Figures:

- [current_rounded_objective.png](current_rounded_objective.png)
- [current_rounded_objective.pdf](current_rounded_objective.pdf)
- [rounded_original_fw_gap.png](rounded_original_fw_gap.png)
- [rounded_original_fw_gap.pdf](rounded_original_fw_gap.pdf)
- [raw_feasibility.png](raw_feasibility.png)
- [raw_feasibility.pdf](raw_feasibility.pdf)
