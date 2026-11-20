# CPU preliminary oracle diagnostics

Instance: `metric128_s1000.npz`; n=128; float64; one CPU thread; 100 independent evaluations per configuration.

These fixed-plan diagnostics do not validate a complete A100 optimizer experiment, select a frozen default oracle, or implement the missing theorem-specific optimizer.

| Plan | Oracle | q | Mean relative error | Mean absolute error | Empirical mean bias | Frobenius variance | Median decision error | Median time (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| product | anchor | 8 | 22.59 | 289.5 | 28.88 | 8.385e+04 | 0.006066 | 0.8577 |
| product | anchor | 32 | 11.32 | 145 | 14.66 | 2.103e+04 | 0.005448 | 2.824 |
| product | anchor | 128 | 5.659 | 72.53 | 7.295 | 5261 | 0.004261 | 10.65 |
| product | trace_sketch | 8 | 42.93 | 550.2 | 52.27 | 3.214e+05 | 0.006687 | 0.419 |
| product | trace_sketch | 32 | 22.37 | 286.7 | 28.71 | 8.37e+04 | 0.00693 | 0.7228 |
| product | trace_sketch | 128 | 11.3 | 144.9 | 15.2 | 2.118e+04 | 0.006251 | 1.674 |
| mixed_permutation | anchor | 8 | 22.61 | 289.8 | 29 | 8.403e+04 | 0.006584 | 0.8413 |
| mixed_permutation | anchor | 32 | 11.31 | 145 | 14.48 | 2.101e+04 | 0.005956 | 2.806 |
| mixed_permutation | anchor | 128 | 5.659 | 72.53 | 7.255 | 5261 | 0.004908 | 10.61 |
| mixed_permutation | trace_sketch | 8 | 42.39 | 543.3 | 53.53 | 3.134e+05 | 0.006315 | 0.4171 |
| mixed_permutation | trace_sketch | 32 | 22.79 | 292.1 | 30.3 | 8.717e+04 | 0.007266 | 0.7127 |
| mixed_permutation | trace_sketch | 128 | 11.18 | 143.3 | 14.8 | 2.07e+04 | 0.006277 | 1.703 |
| concentrated_permutation | anchor | 8 | 22.95 | 295.3 | 29.64 | 8.754e+04 | 0.02657 | 0.8524 |
| concentrated_permutation | anchor | 32 | 11.49 | 147.8 | 14.6 | 2.186e+04 | 0.02657 | 2.826 |
| concentrated_permutation | anchor | 128 | 5.744 | 73.89 | 7.403 | 5462 | 0.02345 | 10.62 |
| concentrated_permutation | trace_sketch | 8 | 43.42 | 558.6 | 54.25 | 3.316e+05 | 0.02405 | 0.4165 |
| concentrated_permutation | trace_sketch | 32 | 21.74 | 279.8 | 28.14 | 7.957e+04 | 0.02302 | 0.7189 |
| concentrated_permutation | trace_sketch | 128 | 11.06 | 142.3 | 13.06 | 2.042e+04 | 0.02274 | 1.669 |

Exact-gradient median time (ms): product: 0.5161; mixed_permutation: 0.5082; concentrated_permutation: 0.5104.

Empirical mean bias is the norm of a finite Monte Carlo mean error, not a demonstration of estimator bias. The JSON includes its sampling-noise scale, full mean/median/IQR distributions, plan seeds, source hashes and timing definitions. Anchor timings are for the supplied Python-loop correctness reference. Only saved A/B/a/b are loaded; the planted matching is not used.

Raw evaluations and reproducible evaluation seeds: `runs/oracle_cpu_n128_s1000/evaluations.csv`.
Structured report: `reports/oracle_cpu_n128_s1000.json`.
