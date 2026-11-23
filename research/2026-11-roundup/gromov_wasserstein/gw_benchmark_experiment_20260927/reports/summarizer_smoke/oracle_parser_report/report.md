# Parser smoke using existing local outputs

Parser/plotting smoke test using existing recorded local outputs. Directory layout is only a test fixture; these are not A100 measurements or experiment results beyond the original CPU checks.

Panel status: **incomplete**. Completed smoothing runs: 0/4; completed oracle reports: 1/1.

| Expected run/report | Status | Missing evidence or issue |
|---|---|---|
| n128_exact | missing | Missing config.json, summary.json, trace.csv |
| n128_sketch_q8 | missing | Missing config.json, summary.json, trace.csv |
| n128_sketch_q32 | missing | Missing config.json, summary.json, trace.csv |
| n128_sketch_q128 | missing | Missing config.json, summary.json, trace.csv |
| oracle n=128 | completed | — |

Fixed-plan oracle measurements compare batches at the same plan and true full gradient. Timing/error medians and IQR describe independent oracle evaluations; they are separate from optimizer-seed uncertainty. Anchor timings use the supplied Python-loop implementation. These measurements do not freeze an oracle or establish algorithmic speedups.

| n | Plan | Oracle | q | Evaluations | Mean absolute error | Mean relative error | Frobenius variance | Median row-LMO regret | Median oracle ms | Median full-gradient ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | product | anchor | 8 | 100 | 289.53 | 22.591 | 83849 | 0.0060661 | 0.85772 | 0.51613 |
| 128 | product | anchor | 32 | 100 | 145.04 | 11.317 | 21033 | 0.0054482 | 2.8244 | 0.51613 |
| 128 | product | anchor | 128 | 100 | 72.532 | 5.6595 | 5260.6 | 0.0042608 | 10.646 | 0.51613 |
| 128 | product | trace_sketch | 8 | 100 | 550.25 | 42.934 | 3.2139e+05 | 0.0066871 | 0.41896 | 0.51613 |
| 128 | product | trace_sketch | 32 | 100 | 286.69 | 22.37 | 83700 | 0.0069295 | 0.72278 | 0.51613 |
| 128 | product | trace_sketch | 128 | 100 | 144.87 | 11.304 | 21184 | 0.0062514 | 1.6739 | 0.51613 |
| 128 | mixed_permutation | anchor | 8 | 100 | 289.84 | 22.614 | 84026 | 0.0065836 | 0.84126 | 0.50821 |
| 128 | mixed_permutation | anchor | 32 | 100 | 144.95 | 11.31 | 21013 | 0.0059561 | 2.8061 | 0.50821 |
| 128 | mixed_permutation | anchor | 128 | 100 | 72.531 | 5.6591 | 5261 | 0.0049076 | 10.611 | 0.50821 |
| 128 | mixed_permutation | trace_sketch | 8 | 100 | 543.35 | 42.394 | 3.1337e+05 | 0.0063154 | 0.41714 | 0.50821 |
| 128 | mixed_permutation | trace_sketch | 32 | 100 | 292.14 | 22.793 | 87170 | 0.0072656 | 0.71272 | 0.50821 |
| 128 | mixed_permutation | trace_sketch | 128 | 100 | 143.31 | 11.182 | 20705 | 0.0062771 | 1.7028 | 0.50821 |
| 128 | concentrated_permutation | anchor | 8 | 100 | 295.33 | 22.955 | 87536 | 0.026569 | 0.85245 | 0.51036 |
| 128 | concentrated_permutation | anchor | 32 | 100 | 147.76 | 11.485 | 21856 | 0.026569 | 2.8263 | 0.51036 |
| 128 | concentrated_permutation | anchor | 128 | 100 | 73.895 | 5.7436 | 5461.7 | 0.023454 | 10.621 | 0.51036 |
| 128 | concentrated_permutation | trace_sketch | 8 | 100 | 558.62 | 43.42 | 3.3162e+05 | 0.024046 | 0.41653 | 0.51036 |
| 128 | concentrated_permutation | trace_sketch | 32 | 100 | 279.76 | 21.745 | 79571 | 0.023018 | 0.71892 | 0.51036 |
| 128 | concentrated_permutation | trace_sketch | 128 | 100 | 142.34 | 11.064 | 20420 | 0.022737 | 1.669 | 0.51036 |

Figure interpretation: plotted objective values are current rounded outputs, never best-selected values. Time plots show only recorded checkpoints. True-gap means include every pre-update iterate and exclude the extra terminal point. With one optimization seed, these are realized trajectory statistics, not estimates of theorem expectations. The exact-gradient ablation retains the same momentum schedule. Signed numerical values are preserved; gap plots use a symmetric logarithmic scale near zero. No slopes are fitted, and no GPU/CPU algorithmic speedup conclusion is drawn.

Figures (PNG and PDF):

- [oracle_time_vs_relative_error.png](oracle_time_vs_relative_error.png)
- [oracle_time_vs_relative_error.pdf](oracle_time_vs_relative_error.pdf)
- [oracle_time_vs_decision_regret.png](oracle_time_vs_decision_regret.png)
- [oracle_time_vs_decision_regret.pdf](oracle_time_vs_decision_regret.pdf)
