# Practical-run parser smoke using preserved CPU test outputs

This is an output-parser/plotting test using preserved small CPU runs and a separate fixture plan. These records are not A100 Stage C results.

Panel status: **partial**. Successful run records: 1/1. Missing checkpoint observations: 1.

All points are current rounded outputs at measured times. Missing windows remain missing. A terminal native early stop appears only at its actual time; no trajectory is extended to the nominal budget.

| Instance / method | Status | Stop reason / issues |
|---|---|---|
| toy16 / native-sampled-gw | completed | native maximum iterations |

Checkpoint opportunities use one common requested grid. Actual observation times are reported without moving an improvement backward to a target time.

| Instance / method | Target 0.1s | Terminal solver s |
|---|---|---|
| toy16 / native-sampled-gw | MISSED: native_stopped_before_checkpoint_or_no_new_completed_iteration | 0.005409 |

Final losses use the actual saved raw and rounded plans promoted to float64, without a second repair. Raw smooth loss is not a feasible objective. Feasibility is reported for the rounded output to arithmetic tolerance.

| Instance / method | Raw loss64 | Rounded loss64 | Raw row L1 | Raw column L1 | Raw column L2 | Rounded row L1 | Rounded column L1 |
|---|---|---|---|---|---|---|---|
| toy16 / native-sampled-gw | 0.07982 | 0.07982 | 6.245e-17 | 1.3347e-11 | 4.9666e-12 | 4.8572e-17 | 9.7145e-17 |

Recorded method parameters are development settings, not selected winners. Native Sampled GW epsilon and conditional-sample counts have their own meanings; they are not equivalent to KL step sizes or trace-sketch probe counts.

| Instance / method | Parameters |
|---|---|
| toy16 / native-sampled-gw | {"epsilon": 0.1, "max_iter": 1, "native_inner_settings_unchanged": true, "native_samples": 8, "sample_semantics": "native conditional samples; not comparable to trace-sketch probes"} |

Time components in seconds. The quality axis includes solver work plus the repair for that output. Diagnostics, float64 validation, serialization, and process overhead are reported separately; no uncounted best-output selection is performed.

| Instance / method | Prepare | Native instrumentation prep | Solver | All repairs | Diagnostics | Validation64 | Serialization | Other overhead | Total instrumented |
|---|---|---|---|---|---|---|---|---|---|
| toy16 / native-sampled-gw | 0.016624 | 0.022906 | 0.005409 | 0.00059815 | 0.0017334 | 0.00075849 | 0.0014689 | 0.00084974 | 0.050347 |

Native snapshot/callback instrumentation is outside its solver clock. The separate native callback duration overlaps repairs and diagnostics and must not be added to those components again; the full instrumentation record is preserved in report.json.

Peak memory covers the instrumented process, including diagnostic and validation work. Unavailable native counters are blank rather than inferred from another method's sample definition.

| Instance / method | RSS MiB | GPU allocated MiB | GPU reserved MiB | Updates | Oracle calls | Sketch probes | Inner iterations | Inner failures | Underflow steps | Max underflow entries |
|---|---|---|---|---|---|---|---|---|---|---|
| toy16 / native-sampled-gw | 403.9 | — | — | 1 | — | — | — | — | — | — |

No failure/warning events were present in the available run records; missing run records cannot establish absence of failures.

This single-seed development pilot supplies no error bars, fitted rates, tuned algorithm ranking, global-optimality claim, or GPU/CPU speedup conclusion. Current feasible quality and raw feasibility remain separate. The BMMC instance receives no biological interpretation. No winner is selected.

Figures:

- [current_rounded_objective.png](current_rounded_objective.png)
- [current_rounded_objective.pdf](current_rounded_objective.pdf)
- [rounded_original_fw_gap.png](rounded_original_fw_gap.png)
- [rounded_original_fw_gap.pdf](rounded_original_fw_gap.pdf)
- [raw_feasibility.png](raw_feasibility.png)
- [raw_feasibility.pdf](raw_feasibility.pdf)
