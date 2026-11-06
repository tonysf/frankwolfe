# CPU PARSER SMOKE — no A100 outcomes

Status: **incomplete**; 2/3 valid prescribed runs; 5 pre-update points per run.

The primary curve averages each run's full-prefix arithmetic mean of the true smoothed FW gap. All runs use the same frozen instance. The IQR shows stochastic run dispersion; it is not an expectation confidence interval or independent-dataset uncertainty. Terminal points are excluded from these averages.

No slope is fitted. Feasibility, estimator-error and current-output curves are descriptive; they do not establish a last-iterate guarantee.

| Seed | Status | Full-trajectory mean true gap | Terminal actual rounded loss (float64) | Terminal original FW gap (float64) | Raw col L1 | Raw col L2 | Rounded both-L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | completed | 71.7294 | 0.0500053 | 0.000738522 | 1.91253 | 0.46185 | 1.52049e-15 |
| 1 | completed | 71.7359 | 0.0500151 | 0.000506599 | 1.92187 | 0.464819 | 2.60209e-18 |
| 2 | missing | — | — | — | — | — | — |

At t=5, the mean across the 2 included stochastic runs is 71.7326; the run-level IQR is [71.731, 71.7343]. The prescribed replication set is incomplete.

| Seed | Solver s | Repair s | Diagnostics s | Serialization s | Float64 audit s | Total instrumented s | CPU peak MiB | GPU allocated peak MiB | GPU reserved peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0123952 | 0.00170146 | 0.257514 | 0.00232316 | 0.0110128 | 1.26074 | 346.363 | — | — |
| 1 | 0.0125441 | 0.00168117 | 0.258746 | 0.00226009 | 0.0111163 | 1.26907 | 346.25 | — | — |

Validation issues:

- seed2: Missing config.json, summary.json, trace.csv, final.npz, final_float64_evaluation.json

The JSON preserves per-seed prefix means, pointwise ensemble statistics, all source/input hashes, validation status, actual-output audits, timings and memory. PNG and PDF figures accompany this report. Missing or failed runs are retained in the status table.
