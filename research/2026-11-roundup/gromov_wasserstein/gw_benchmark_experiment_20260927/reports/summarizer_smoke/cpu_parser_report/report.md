# Parser smoke using existing local outputs

Parser/plotting smoke test using existing recorded local outputs. Directory layout is only a test fixture; these are not A100 measurements or experiment results beyond the original CPU checks.

Panel status: **incomplete**. Completed smoothing runs: 2/2; completed oracle reports: 0/1.

| Expected run/report | Status | Missing evidence or issue |
|---|---|---|
| n16_exact | completed | Legacy final_float64 fields describe fresh float64 repair, not verified actual saved rounded output. |
| n16_sketch_q32 | completed | Legacy final_float64 fields describe fresh float64 repair, not verified actual saved rounded output. |
| oracle n=16 | missing | Expected oracle report is missing. |

Final outputs, reevaluated in CPU float64 using the actual saved rounded plan. Raw loss is the smooth quadratic at an infeasible point; rounded output feasibility is reported separately.

| Run | Raw smooth loss | Actual rounded loss | Raw column L1 | Raw column L2 | Rounded marginal L1 | Original gap at actual rounded output | Full-trajectory mean true gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| n16_exact | 0.12937 | — | 0.7261 | 0.27983 | — | — | 3.2545 |
| n16_sketch_q32 | 0.17587 | — | 1.375 | 0.40171 | — | — | 9.346 |

Time components in seconds. The plotted time-to-output is cumulative solver time plus that checkpoint's repair; diagnostic/serialization overhead is excluded from that axis and shown here. Total elapsed includes the instrumented run. Values are one observed run, not runtime estimates across repetitions.

| Run | Preparation/cache | Solver | Repairs, all checkpoints | Exact diagnostics | Serialization | Final float64 check | Instrumented total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n16_exact | 1.0509 | 0.0027535 | 0.0012251 | 0.28244 | 0.0018268 | 0.0013666 | 1.3409 |
| n16_sketch_q32 | 0.97828 | 0.0025261 | 0.00089859 | 0.22612 | 0.0013991 | 0.00097853 | 1.2105 |

Peak memory includes diagnostics and terminal validation; it is not a solver-only memory measurement.

| Run | Peak process RSS MiB | Peak GPU allocated MiB | Peak GPU reserved MiB | Algorithm oracle calls | Exact-gradient calls | Sketch probes | Row LMO calls | Diagnostic gradients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n16_exact | 340.42 | — | — | 5 | 5 | 0 | 5 | 12 |
| n16_sketch_q32 | 340.86 | — | — | 5 | 0 | 160 | 5 | 12 |

Figure interpretation: plotted objective values are current rounded outputs, never best-selected values. Time plots show only recorded checkpoints. True-gap means include every pre-update iterate and exclude the extra terminal point. With one optimization seed, these are realized trajectory statistics, not estimates of theorem expectations. The exact-gradient ablation retains the same momentum schedule. Signed numerical values are preserved; gap plots use a symmetric logarithmic scale near zero. No slopes are fitted, and no GPU/CPU algorithmic speedup conclusion is drawn.

Figures (PNG and PDF):

- [current_rounded_objective_vs_output_time.png](current_rounded_objective_vs_output_time.png)
- [current_rounded_objective_vs_output_time.pdf](current_rounded_objective_vs_output_time.pdf)
- [true_smoothed_gap_vs_iteration.png](true_smoothed_gap_vs_iteration.png)
- [true_smoothed_gap_vs_iteration.pdf](true_smoothed_gap_vs_iteration.pdf)
- [true_smoothed_gap_vs_probes.png](true_smoothed_gap_vs_probes.png)
- [true_smoothed_gap_vs_probes.pdf](true_smoothed_gap_vs_probes.pdf)
- [full_trajectory_mean_gap.png](full_trajectory_mean_gap.png)
- [full_trajectory_mean_gap.pdf](full_trajectory_mean_gap.pdf)
- [rounded_original_gap_vs_output_time.png](rounded_original_gap_vs_output_time.png)
- [rounded_original_gap_vs_output_time.pdf](rounded_original_gap_vs_output_time.pdf)
- [raw_feasibility_vs_solver_time.png](raw_feasibility_vs_solver_time.png)
- [raw_feasibility_vs_solver_time.pdf](raw_feasibility_vs_solver_time.pdf)
