# CPU preliminary oracle diagnostics

Instance: `metric512_s1000.npz`; n=512; float64; one CPU thread; 100 independent evaluations per configuration.

These fixed-plan diagnostics do not validate a complete A100 optimizer experiment, select a frozen default oracle, or implement the missing theorem-specific optimizer.

| Plan | Oracle | q | Mean relative error | Mean absolute error | Empirical mean bias | Frobenius variance | Median decision error | Median time (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| product | anchor | 8 | 90.52 | 4069 | 406.3 | 1.656e+07 | 0.00458 | 32.93 |
| product | anchor | 32 | 45.25 | 2034 | 203.5 | 4.138e+06 | 0.004355 | 129.3 |
| product | anchor | 128 | 22.63 | 1017 | 101.5 | 1.034e+06 | 0.003936 | 515.4 |
| product | trace_sketch | 8 | 103 | 4632 | 493.8 | 2.312e+07 | 0.004602 | 7.652 |
| product | trace_sketch | 32 | 50.84 | 2285 | 219.4 | 5.296e+06 | 0.004552 | 12.33 |
| product | trace_sketch | 128 | 25.62 | 1152 | 112.1 | 1.332e+06 | 0.004364 | 26.21 |
| mixed_permutation | anchor | 8 | 90.47 | 4067 | 406.6 | 1.654e+07 | 0.004675 | 32.98 |
| mixed_permutation | anchor | 32 | 45.27 | 2035 | 203.4 | 4.14e+06 | 0.004467 | 129.2 |
| mixed_permutation | anchor | 128 | 22.63 | 1017 | 101.5 | 1.035e+06 | 0.004057 | 514.6 |
| mixed_permutation | trace_sketch | 8 | 100.4 | 4514 | 466.9 | 2.173e+07 | 0.004908 | 7.441 |
| mixed_permutation | trace_sketch | 32 | 51 | 2293 | 227.9 | 5.345e+06 | 0.004462 | 12.32 |
| mixed_permutation | trace_sketch | 128 | 25.61 | 1151 | 110.4 | 1.333e+06 | 0.004841 | 26.3 |
| concentrated_permutation | anchor | 8 | 90.25 | 4061 | 406.7 | 1.65e+07 | 0.01293 | 32.78 |
| concentrated_permutation | anchor | 32 | 45.34 | 2040 | 203.9 | 4.164e+06 | 0.01293 | 129 |
| concentrated_permutation | anchor | 128 | 22.67 | 1020 | 102.1 | 1.04e+06 | 0.01293 | 513.5 |
| concentrated_permutation | trace_sketch | 8 | 100.5 | 4524 | 442 | 2.169e+07 | 0.01215 | 7.6 |
| concentrated_permutation | trace_sketch | 32 | 51.23 | 2305 | 229.2 | 5.389e+06 | 0.01238 | 12.46 |
| concentrated_permutation | trace_sketch | 128 | 25.6 | 1152 | 118.7 | 1.333e+06 | 0.01223 | 26.35 |

Exact-gradient median time (ms): product: 24.56; mixed_permutation: 25.12; concentrated_permutation: 25.1.

Empirical mean bias is the norm of a finite Monte Carlo mean error, not a demonstration of estimator bias. The JSON includes its sampling-noise scale, full mean/median/IQR distributions, plan seeds, source hashes and timing definitions. Anchor timings are for the supplied Python-loop correctness reference. Only saved A/B/a/b are loaded; the planted matching is not used.

Raw evaluations and reproducible evaluation seeds: `runs/oracle_cpu_n512_s1000/evaluations.csv`.
Structured report: `reports/oracle_cpu_n512_s1000.json`.
