# Initial A100 diagnostic candidate pilot

One development instance per size, one optimization seed, and prespecified candidate batches. This is a diagnostic candidate pilot, not a completed tuning exercise, held-out final study, convergence proof, or algorithmic speed comparison.

Panel status: **complete_initial_candidate_panel**. Completed smoothing runs: 8/8; completed oracle reports: 2/2.

| Expected run/report | Status | Missing evidence or issue |
|---|---|---|
| n128_exact | completed | — |
| n128_sketch_q8 | completed | — |
| n128_sketch_q32 | completed | — |
| n128_sketch_q128 | completed | — |
| n512_exact | completed | — |
| n512_sketch_q8 | completed | — |
| n512_sketch_q32 | completed | — |
| n512_sketch_q128 | completed | — |
| oracle n=128 | completed | — |
| oracle n=512 | completed | — |

Final outputs, reevaluated in CPU float64 using the actual saved rounded plan. Raw loss is the smooth quadratic at an infeasible point; rounded output feasibility is reported separately.

| Run | Raw smooth loss | Actual rounded loss | Raw column L1 | Raw column L2 | Rounded marginal L1 | Original gap at actual rounded output | Full-trajectory mean true gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| n128_exact | 0.052085 | 0.049991 | 0.57624 | 0.060454 | 7.1274e-08 | 0.00070648 | 7.7785 |
| n128_sketch_q8 | 0.052099 | 0.050012 | 0.60727 | 0.063323 | 5.8944e-08 | 0.00055953 | 11.356 |
| n128_sketch_q32 | 0.051866 | 0.050015 | 0.5908 | 0.061949 | 5.5244e-08 | 0.00060239 | 11.232 |
| n128_sketch_q128 | 0.051685 | 0.050012 | 0.57553 | 0.059894 | 6.6405e-08 | 0.00065214 | 11.168 |
| n512_exact | 0.046593 | 0.043846 | 1.133 | 0.07579 | 7.3605e-08 | 0.0005428 | 23.847 |
| n512_sketch_q8 | 0.048663 | 0.043884 | 1.4412 | 0.092783 | 1.1296e-07 | 0.00019031 | 44.669 |
| n512_sketch_q32 | 0.048484 | 0.043883 | 1.4409 | 0.092689 | 1.4823e-07 | 0.00019031 | 44.735 |
| n512_sketch_q128 | 0.048638 | 0.043883 | 1.4386 | 0.092648 | 9.1666e-08 | 0.00018927 | 44.673 |

Time components in seconds. The plotted time-to-output is cumulative solver time plus that checkpoint's repair; diagnostic/serialization overhead is excluded from that axis and shown here. Total elapsed includes the instrumented run. Values are one observed run, not runtime estimates across repetitions.

| Run | Preparation/cache | Solver | Repairs, all checkpoints | Exact diagnostics | Serialization | Final float64 check | Instrumented total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n128_exact | 0.84018 | 0.35624 | 0.089774 | 7.3241 | 0.022467 | 0.0099507 | 8.6527 |
| n128_sketch_q8 | 0.84356 | 0.47841 | 0.089775 | 2.2463 | 0.022826 | 0.010179 | 3.7009 |
| n128_sketch_q32 | 0.75885 | 0.48882 | 0.094358 | 1.692 | 0.022505 | 0.0094464 | 3.0758 |
| n128_sketch_q128 | 0.82201 | 0.48916 | 0.09566 | 1.7142 | 0.023536 | 0.0098448 | 3.1643 |
| n512_exact | 0.89199 | 0.35754 | 0.092405 | 17.969 | 0.023116 | 0.23468 | 19.578 |
| n512_sketch_q8 | 0.90533 | 0.60036 | 0.097889 | 27.254 | 0.026944 | 0.30642 | 29.201 |
| n512_sketch_q32 | 0.81362 | 0.49218 | 0.093492 | 28.006 | 0.059661 | 0.33002 | 29.805 |
| n512_sketch_q128 | 0.70004 | 0.49318 | 0.095043 | 28.466 | 0.026605 | 0.32096 | 30.111 |

Peak memory includes diagnostics and terminal validation; it is not a solver-only memory measurement.

| Run | Peak process RSS MiB | Peak GPU allocated MiB | Peak GPU reserved MiB | Algorithm oracle calls | Exact-gradient calls | Sketch probes | Row LMO calls | Diagnostic gradients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n128_exact | 984.32 | 33.001 | 34 | 200 | 200 | 0 | 200 | 402 |
| n128_sketch_q8 | 985.82 | 33.001 | 34 | 200 | 0 | 1600 | 200 | 402 |
| n128_sketch_q32 | 984.81 | 33.001 | 34 | 200 | 0 | 6400 | 200 | 402 |
| n128_sketch_q128 | 983.56 | 33.002 | 34 | 200 | 0 | 25600 | 200 | 402 |
| n512_exact | 1018.6 | 48.004 | 70 | 200 | 200 | 0 | 200 | 402 |
| n512_sketch_q8 | 1025.6 | 48.004 | 70 | 200 | 0 | 1600 | 200 | 402 |
| n512_sketch_q32 | 1033.9 | 48.004 | 70 | 200 | 0 | 6400 | 200 | 402 |
| n512_sketch_q128 | 1035.1 | 48.004 | 70 | 200 | 0 | 25600 | 200 | 402 |

Fixed-plan oracle measurements compare batches at the same plan and true full gradient. Timing/error medians and IQR describe independent oracle evaluations; they are separate from optimizer-seed uncertainty. Anchor timings use the supplied Python-loop implementation. These measurements do not freeze an oracle or establish algorithmic speedups.

| n | Plan | Oracle | q | Evaluations | Mean absolute error | Mean relative error | Frobenius variance | Median row-LMO regret | Median oracle ms | Median full-gradient ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | product | anchor | 8 | 100 | 290.08 | 22.634 | 84163 | 0.0060475 | 1.0611 | 0.5365 |
| 128 | product | anchor | 32 | 100 | 145.16 | 11.327 | 21074 | 0.0053032 | 3.4722 | 0.5365 |
| 128 | product | anchor | 128 | 100 | 72.48 | 5.6554 | 5253.7 | 0.004197 | 13.098 | 0.5365 |
| 128 | product | trace_sketch | 8 | 100 | 569.07 | 44.403 | 3.4804e+05 | 0.0056678 | 1.1474 | 0.5365 |
| 128 | product | trace_sketch | 32 | 100 | 282.79 | 22.065 | 81637 | 0.0057437 | 1.1447 | 0.5365 |
| 128 | product | trace_sketch | 128 | 100 | 140.52 | 10.964 | 19900 | 0.0061395 | 1.1409 | 0.5365 |
| 128 | mixed_permutation | anchor | 8 | 100 | 289.95 | 22.623 | 84108 | 0.0066226 | 1.0608 | 0.53683 |
| 128 | mixed_permutation | anchor | 32 | 100 | 144.9 | 11.306 | 20998 | 0.0059443 | 3.469 | 0.53683 |
| 128 | mixed_permutation | anchor | 128 | 100 | 72.496 | 5.6563 | 5255.5 | 0.0049362 | 13.1 | 0.53683 |
| 128 | mixed_permutation | trace_sketch | 8 | 100 | 561.45 | 43.806 | 3.333e+05 | 0.0068653 | 1.1466 | 0.53683 |
| 128 | mixed_permutation | trace_sketch | 32 | 100 | 283.96 | 22.155 | 82374 | 0.0064804 | 1.1457 | 0.53683 |
| 128 | mixed_permutation | trace_sketch | 128 | 100 | 144.49 | 11.274 | 21069 | 0.0059914 | 1.1409 | 0.53683 |
| 128 | concentrated_permutation | anchor | 8 | 100 | 296.1 | 23.014 | 87975 | 0.026569 | 1.0603 | 0.53553 |
| 128 | concentrated_permutation | anchor | 32 | 100 | 148.03 | 11.505 | 21944 | 0.026569 | 3.472 | 0.53553 |
| 128 | concentrated_permutation | anchor | 128 | 100 | 73.617 | 5.722 | 5421 | 0.023861 | 13.106 | 0.53553 |
| 128 | concentrated_permutation | trace_sketch | 8 | 100 | 563.7 | 43.814 | 3.3537e+05 | 0.022957 | 1.1474 | 0.53553 |
| 128 | concentrated_permutation | trace_sketch | 32 | 100 | 287.87 | 22.375 | 85030 | 0.023131 | 1.1449 | 0.53553 |
| 128 | concentrated_permutation | trace_sketch | 128 | 100 | 141.62 | 11.007 | 20166 | 0.023549 | 1.1401 | 0.53553 |
| 512 | product | anchor | 8 | 100 | 4068.2 | 90.503 | 1.6552e+07 | 0.0045622 | 1.0935 | 0.54045 |
| 512 | product | anchor | 32 | 100 | 2032.5 | 45.215 | 4.131e+06 | 0.0043456 | 3.5854 | 0.54045 |
| 512 | product | anchor | 128 | 100 | 1016.9 | 22.622 | 1.034e+06 | 0.0039335 | 13.532 | 0.54045 |
| 512 | product | trace_sketch | 8 | 100 | 4530.2 | 100.78 | 2.2099e+07 | 0.0046489 | 1.1617 | 0.54045 |
| 512 | product | trace_sketch | 32 | 100 | 2283.3 | 50.795 | 5.3016e+06 | 0.004637 | 1.1657 | 0.54045 |
| 512 | product | trace_sketch | 128 | 100 | 1151.6 | 25.618 | 1.3341e+06 | 0.003957 | 1.1649 | 0.54045 |
| 512 | mixed_permutation | anchor | 8 | 100 | 4069.2 | 90.524 | 1.6559e+07 | 0.0046826 | 1.0933 | 0.53986 |
| 512 | mixed_permutation | anchor | 32 | 100 | 2033 | 45.227 | 4.1332e+06 | 0.0044695 | 3.5845 | 0.53986 |
| 512 | mixed_permutation | anchor | 128 | 100 | 1017.3 | 22.632 | 1.035e+06 | 0.004071 | 13.535 | 0.53986 |
| 512 | mixed_permutation | trace_sketch | 8 | 100 | 4465 | 99.33 | 2.153e+07 | 0.004715 | 1.1613 | 0.53986 |
| 512 | mixed_permutation | trace_sketch | 32 | 100 | 2216.4 | 49.306 | 4.9725e+06 | 0.0045538 | 1.1659 | 0.53986 |
| 512 | mixed_permutation | trace_sketch | 128 | 100 | 1147.5 | 25.528 | 1.3213e+06 | 0.0045776 | 1.1629 | 0.53986 |
| 512 | concentrated_permutation | anchor | 8 | 100 | 4089.1 | 90.869 | 1.6735e+07 | 0.012925 | 1.0917 | 0.53765 |
| 512 | concentrated_permutation | anchor | 32 | 100 | 2043.7 | 45.416 | 4.1779e+06 | 0.012925 | 3.5842 | 0.53765 |
| 512 | concentrated_permutation | anchor | 128 | 100 | 1020.5 | 22.677 | 1.0414e+06 | 0.012925 | 13.528 | 0.53765 |
| 512 | concentrated_permutation | trace_sketch | 8 | 100 | 4479.6 | 99.547 | 2.1228e+07 | 0.01218 | 1.1604 | 0.53765 |
| 512 | concentrated_permutation | trace_sketch | 32 | 100 | 2267.4 | 50.386 | 5.2094e+06 | 0.012141 | 1.1663 | 0.53765 |
| 512 | concentrated_permutation | trace_sketch | 128 | 100 | 1153.5 | 25.633 | 1.3354e+06 | 0.011947 | 1.1644 | 0.53765 |

Oracle measurement cost and memory (diagnostics included). Timed stochastic evaluations and timed exact gradients are counted separately from warmup and initial reference evaluations.

| n | Stochastic evaluations | Timed exact gradients | Preparation s | Warmup s | Sampled-oracle s | Exact-gradient s | Diagnostics s | Serialization s | Instrumented total s | RSS MiB | GPU allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1800 | 300 | 0.49037 | 0.201 | 6.3252 | 0.16035 | 0.89602 | 0.07921 | 18.195 | 988.54 | 34.439 |
| 512 | 1800 | 300 | 0.50134 | 0.15763 | 6.5175 | 0.16154 | 0.91862 | 1.1937 | 17 | 1053.8 | 68.008 |

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
- [oracle_time_vs_relative_error.png](oracle_time_vs_relative_error.png)
- [oracle_time_vs_relative_error.pdf](oracle_time_vs_relative_error.pdf)
- [oracle_time_vs_decision_regret.png](oracle_time_vs_decision_regret.png)
- [oracle_time_vs_decision_regret.pdf](oracle_time_vs_decision_regret.pdf)
