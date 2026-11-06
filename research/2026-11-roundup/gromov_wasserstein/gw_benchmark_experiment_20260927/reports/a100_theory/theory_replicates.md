# A100 fixed-instance stochastic replications

Status: **completed**; 10/10 valid prescribed runs; 1000 pre-update points per run.

The primary curve averages each run's full-prefix arithmetic mean of the true smoothed FW gap. All runs use the same frozen instance. The IQR shows stochastic run dispersion; it is not an expectation confidence interval or independent-dataset uncertainty. Terminal points are excluded from these averages.

No slope is fitted. Feasibility, estimator-error and current-output curves are descriptive; they do not establish a last-iterate guarantee.

| Seed | Status | Full-trajectory mean true gap | Terminal actual rounded loss (float64) | Terminal original FW gap (float64) | Raw col L1 | Raw col L2 | Rounded both-L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | completed | 4.62762 | 0.0500105 | 0.000600633 | 0.175463 | 0.0183493 | 7.31088e-08 |
| 1 | completed | 4.62089 | 0.050003 | 0.000608194 | 0.181731 | 0.0185286 | 6.98601e-08 |
| 2 | completed | 4.61634 | 0.0500095 | 0.000604448 | 0.186726 | 0.018817 | 5.99139e-08 |
| 3 | completed | 4.6503 | 0.0500156 | 0.000613748 | 0.175333 | 0.017984 | 5.71708e-08 |
| 4 | completed | 4.64741 | 0.0500093 | 0.000592247 | 0.178625 | 0.0181954 | 5.97611e-08 |
| 5 | completed | 4.64104 | 0.0500131 | 0.000610157 | 0.183559 | 0.0186221 | 6.40248e-08 |
| 6 | completed | 4.62831 | 0.0500055 | 0.000595138 | 0.172867 | 0.0177202 | 7.18337e-08 |
| 7 | completed | 4.52952 | 0.0500071 | 0.000605126 | 0.186538 | 0.0188153 | 6.39156e-08 |
| 8 | completed | 4.6428 | 0.0500114 | 0.000607814 | 0.185533 | 0.0189306 | 6.83904e-08 |
| 9 | completed | 4.55295 | 0.0500084 | 0.000591713 | 0.177551 | 0.0180653 | 6.48979e-08 |

At t=1000, the mean across the 10 included stochastic runs is 4.61572; the run-level IQR is [4.61748, 4.64236].

| Seed | Solver s | Repair s | Diagnostics s | Serialization s | Float64 audit s | Total instrumented s | CPU peak MiB | GPU allocated peak MiB | GPU reserved peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.32374 | 0.446974 | 12.9765 | 0.0848754 | 0.213149 | 17.1215 | 986.746 | 33.002 | 34 |
| 1 | 1.96048 | 0.431337 | 7.54672 | 0.0798847 | 0.00947554 | 10.804 | 981.59 | 33.002 | 34 |
| 2 | 1.90891 | 0.44067 | 7.35978 | 0.111948 | 0.00988475 | 10.6242 | 989.047 | 33.002 | 34 |
| 3 | 1.91516 | 0.441429 | 7.41353 | 0.10589 | 0.00980052 | 10.6081 | 1002.66 | 33.002 | 34 |
| 4 | 1.88024 | 0.426189 | 7.33437 | 0.117515 | 0.00929244 | 10.5189 | 988.391 | 33.002 | 34 |
| 5 | 2.00746 | 0.438666 | 7.41423 | 0.0914359 | 0.0100645 | 10.5935 | 997.98 | 33.002 | 34 |
| 6 | 1.97238 | 0.428794 | 7.29378 | 0.120793 | 0.00958282 | 10.6284 | 989.992 | 33.002 | 34 |
| 7 | 1.93344 | 0.428183 | 7.27458 | 0.0837664 | 0.00944723 | 10.397 | 987.82 | 33.002 | 34 |
| 8 | 1.88847 | 0.432285 | 7.31612 | 0.128527 | 0.00970141 | 10.4599 | 985.621 | 33.002 | 34 |
| 9 | 1.9029 | 0.434113 | 7.33381 | 0.0894899 | 0.010298 | 10.5772 | 990.309 | 33.002 | 34 |

The JSON preserves per-seed prefix means, pointwise ensemble statistics, all source/input hashes, validation status, actual-output audits, timings and memory. PNG and PDF figures accompany this report. Missing or failed runs are retained in the status table.
