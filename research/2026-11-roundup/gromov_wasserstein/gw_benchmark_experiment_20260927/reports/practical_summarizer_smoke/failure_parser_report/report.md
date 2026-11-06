# Practical-run parser smoke using preserved CPU test outputs

This is an output-parser/plotting test using preserved small CPU runs and a separate fixture plan. These records are not A100 Stage C results.

Panel status: **partial**. Successful run records: 0/1. Missing checkpoint observations: 1.

All points are current rounded outputs at measured times. Missing windows remain missing. A terminal native early stop appears only at its actual time; no trajectory is extended to the nominal budget.

| Instance / method | Status | Stop reason / issues |
|---|---|---|
| toy_failure / fw | failed | FloatingPointError: Nonfinite gradient rejected. |

Checkpoint opportunities use one common requested grid. Actual observation times are reported without moving an improvement backward to a target time.

| Instance / method | Target 0.03s | Terminal solver s |
|---|---|---|
| toy_failure / fw | MISSED: run_ended_before_target | 0.00013344 |

Recorded method parameters are development settings, not selected winners. Native Sampled GW epsilon and conditional-sample counts have their own meanings; they are not equivalent to KL step sizes or trace-sketch probe counts.

| Instance / method | Parameters |
|---|---|
| toy_failure / fw | {"line_search": "exact quadratic", "lmo": "exact SciPy assignment"} |

Time components in seconds. The quality axis includes solver work plus the repair for that output. Diagnostics, float64 validation, serialization, and process overhead are reported separately; no uncounted best-output selection is performed.

| Instance / method | Prepare | Native instrumentation prep | Solver | All repairs | Diagnostics | Validation64 | Serialization | Other overhead | Total instrumented |
|---|---|---|---|---|---|---|---|---|---|
| toy_failure / fw | 0.016084 | — | 0.00013344 | 0.00031957 | 0.00039712 | 0 | 0.0013114 | 5.226e-05 | 0.018298 |

Native snapshot/callback instrumentation is outside its solver clock. The separate native callback duration overlaps repairs and diagnostics and must not be added to those components again; the full instrumentation record is preserved in report.json.

Peak memory covers the instrumented process, including diagnostic and validation work. Unavailable native counters are blank rather than inferred from another method's sample definition.

| Instance / method | RSS MiB | GPU allocated MiB | GPU reserved MiB | Updates | Oracle calls | Sketch probes | Inner iterations | Inner failures | Underflow steps | Max underflow entries |
|---|---|---|---|---|---|---|---|---|---|---|
| toy_failure / fw | 361.79 | — | — | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

Every recorded failure/warning event is preserved in report.json, including events between checkpoints. The table groups identical event types/messages to keep this report readable.

| Instance / method | Event | Count | First iteration | Last iteration | Details |
|---|---|---|---|---|---|
| toy_failure / fw | run_rejected | 1 | 0 | 0 | FloatingPointError: Nonfinite gradient rejected. |

This single-seed development pilot supplies no error bars, fitted rates, tuned algorithm ranking, global-optimality claim, or GPU/CPU speedup conclusion. Current feasible quality and raw feasibility remain separate. The BMMC instance receives no biological interpretation. No winner is selected.

Figures:

- [current_rounded_objective.png](current_rounded_objective.png)
- [current_rounded_objective.pdf](current_rounded_objective.pdf)
- [rounded_original_fw_gap.png](rounded_original_fw_gap.png)
- [rounded_original_fw_gap.pdf](rounded_original_fw_gap.pdf)
- [raw_feasibility.png](raw_feasibility.png)
- [raw_feasibility.pdf](raw_feasibility.pdf)
