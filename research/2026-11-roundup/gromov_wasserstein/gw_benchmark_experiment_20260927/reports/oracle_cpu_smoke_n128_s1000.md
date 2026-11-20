# CPU preliminary oracle diagnostics

Instance: `metric128_s1000.npz`; n=128; float64; one CPU thread; 2 independent evaluations per configuration.

These fixed-plan diagnostics do not validate a complete A100 optimizer experiment, select a frozen default oracle, or implement the missing theorem-specific optimizer.

| Plan | Oracle | q | Mean relative error | Mean absolute error | Empirical mean bias | Frobenius variance | Median decision error | Median time (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| product | anchor | 8 | 22.6 | 289.7 | 204.8 | 8.392e+04 | 0.006177 | 0.8369 |
| product | anchor | 32 | 11.26 | 144.3 | 102 | 2.087e+04 | 0.005452 | 2.722 |
| product | anchor | 128 | 5.62 | 72.02 | 50.92 | 5189 | 0.004691 | 10.4 |
| product | trace_sketch | 8 | 55.59 | 712.4 | 497.6 | 6.146e+05 | 0.008531 | 0.3815 |
| product | trace_sketch | 32 | 23.91 | 306.4 | 202.5 | 1.063e+05 | 0.009024 | 0.6721 |
| product | trace_sketch | 128 | 11.88 | 152.3 | 104.6 | 2.453e+04 | 0.005172 | 1.703 |
| mixed_permutation | anchor | 8 | 22.49 | 288.2 | 203.7 | 8.313e+04 | 0.006921 | 0.8312 |
| mixed_permutation | anchor | 32 | 11.25 | 144.2 | 101.9 | 2.084e+04 | 0.005991 | 2.763 |
| mixed_permutation | anchor | 128 | 5.656 | 72.49 | 51.06 | 5296 | 0.004802 | 10.65 |
| mixed_permutation | trace_sketch | 8 | 44.58 | 571.4 | 396.5 | 3.455e+05 | 0.007268 | 0.3989 |
| mixed_permutation | trace_sketch | 32 | 21.52 | 275.8 | 194 | 8.23e+04 | 0.00558 | 0.6759 |
| mixed_permutation | trace_sketch | 128 | 10.7 | 137.1 | 89.86 | 2.158e+04 | 0.008616 | 1.675 |
| concentrated_permutation | anchor | 8 | 23.88 | 307.3 | 217.3 | 9.447e+04 | 0.02657 | 0.8195 |
| concentrated_permutation | anchor | 32 | 11.37 | 146.3 | 103 | 2.162e+04 | 0.02476 | 2.789 |
| concentrated_permutation | anchor | 128 | 5.646 | 72.64 | 51.14 | 5322 | 0.02533 | 10.41 |
| concentrated_permutation | trace_sketch | 8 | 41.3 | 531.4 | 412 | 2.253e+05 | 0.02597 | 0.4161 |
| concentrated_permutation | trace_sketch | 32 | 23.1 | 297.2 | 222.1 | 8.119e+04 | 0.02574 | 0.691 |
| concentrated_permutation | trace_sketch | 128 | 10.76 | 138.4 | 95.67 | 2.06e+04 | 0.02202 | 1.758 |

Exact-gradient median time (ms): product: 0.4888; mixed_permutation: 0.4947; concentrated_permutation: 0.4969.

Empirical mean bias is the norm of a finite Monte Carlo mean error, not a demonstration of estimator bias. The JSON includes its sampling-noise scale, full mean/median/IQR distributions, plan seeds, source hashes and timing definitions. Anchor timings are for the supplied Python-loop correctness reference. Only saved A/B/a/b are loaded; the planted matching is not used.

Raw evaluations and reproducible evaluation seeds: `runs/oracle_cpu_smoke_n128_s1000/evaluations.csv`.
Structured report: `reports/oracle_cpu_smoke_n128_s1000.json`.
