# SHARE-seq 31,890-cell exploratory comparison

All six prescribed final runs passed the report audit: three paired stochastic seeds, product initialization, 300 solver seconds per method, and the same four A100 GPUs. Hyperparameters were transferred unchanged from the selected SNARE-10,000 configurations; there was no SHARE-seq tuning or selection.

- **Smoothed row FW + momentum (sketch gradient, row-sharded)**: gamma0 = 0.1, mu_scale = 0.1; fixed step exponent 3/4 and smoothing exponent 1/4.
- **Transport KL-proximal gradient + momentum (sketch gradient, row-sharded)**: eta = 0.1, Sinkhorn tolerance = 1e-4, maximum 1,000 inner iterations, checked every 10.
- Both methods use 128 Rademacher probes and momentum rho(k) = 0.3/(k+1)^(1/2); the first sketch gradient initializes momentum and is reused at update zero.

The table entries are median [25th percentile, 75th percentile], using linear quantiles over three seeds. Final losses are the full unregularized GW objective evaluated in float64 on the saved rounded output values, without re-rounding or choosing a best iterate.

| Method | Final rounded loss, float64 | Outer updates | Solver wall seconds | Solver GPU-hours (4× wall time) |
|---|---:|---:|---:|---:|
| Smoothed row FW + momentum (sketch gradient, row-sharded) | 0.116899656 [0.116877898, 0.116908477] | 3412 [3411, 3412.5] | 300.04 [300.03, 300.041] | 0.333377 [0.333367, 0.333379] |
| Transport KL-proximal gradient + momentum (sketch gradient, row-sharded) | 0.116552011 [0.116543937, 0.116553397] | 874 [869.5, 875] | 300.199 [300.103, 300.264] | 0.333555 [0.333448, 0.333627] |

FW has lower final loss in 0/3 paired seeds; KL has lower final loss in 3/3. Paired KL-minus-FW loss: -0.000344872701 [-0.000355080179, -0.000332575081]. Negative differences favor KL. Three seeds on one fixed dataset do not establish a general or asymptotic ranking.

| Method | Largest reserved memory on any GPU (GiB) | Inner stopping misses (total) | Runs with misses |
|---|---:|---:|---:|
| Smoothed row FW + momentum (sketch gradient, row-sharded) | 27.414 | 0 | 0/3 |
| Transport KL-proximal gradient + momentum (sketch gradient, row-sharded) | 27.414 | 0 | 0/3 |

Memory includes preparation, solver/diagnostics and final float64 validation. Per-rank, per-phase measurements and their three-run summaries are preserved in report.json. A Sinkhorn stopping miss means the requested inner tolerance was not reached within the cap; the corresponding completed run remains visible and its rounded final feasibility is checked.

| Method | Symmetric exact-cell top-1 accuracy | Normalized true-match rank | True-pair mass |
|---|---:|---:|---:|
| Smoothed row FW + momentum (sketch gradient, row-sharded) | 3.36759e-05 [2.69321e-05, 3.42677e-05] | 0.49657 [0.495957, 0.496642] | 3.2071e-05 [3.20501e-05, 3.21389e-05] |
| Transport KL-proximal gradient + momentum (sketch gradient, row-sharded) | 4.70367e-05 [3.13578e-05, 4.70367e-05] | 0.483487 [0.483281, 0.484714] | 3.32259e-05 [3.31046e-05, 3.32923e-05] |

Top-1 gives uniform credit to exact ties and averages both correspondence directions. Its uniform-plan baseline is 1/31,890 = 3.13577924e-05. Normalized true-match rank counts strictly larger weights plus half the tied wrong weights, divided by 31,889; its uniform baseline is 0.5 and lower is better. True-pair mass also has uniform baseline 3.13577924e-05. These metrics evaluate exact cell pairing, not cell-type matching. Objective ranking alone does not demonstrate biological alignment.

![Current rounded loss against solver plus current repair time](current_rounded_loss_vs_feasible_time.png)

![Current rounded loss against completed outer updates](current_rounded_loss_vs_outer_updates.png)

The plots use the recorded full-objective diagnostics of the current rounded outputs, calculated in the solver’s float32 arithmetic. Final table values use the separate float64 validation. Checkpoints occur at initialization and the first completed update reaching 10, 30, 100 and 300 solver seconds. The time axis includes solver work and repair of the particular output, excluding diagnostic overhead and repairs of earlier outputs. Communication needed by the algorithm is part of solver time. Lines connect observed summaries only; there is no interpolated target attainment, best-so-far curve or unobserved continuous trajectory.

Final solver workload totals 2.00071 GPU-hours. The recorded campaign workload elapsed time is 2490.99 seconds, or 2.76777 GPU-hours at four GPUs. That workload clock includes correctness, pilots and final execution after its start; it is not the full Slurm allocation accounting. Full allocation GPU-hours remain unreported until sacct is collected.

Slurm job: `1983902`; world size: 4; frozen source manifest SHA-256: `dc20a285e3d1f54bd26647789b94b69d5949cab5bc4fff99ecf410580a58a293`. report.json records exact data-array hashes, runtime hashes, per-run metadata hashes, allocation environment, trace observations, counters and final feasibility evidence. Saved coupling hashes are taken from runtime metadata; this lightweight report does not reread the approximately 48.8 GB of saved plans.

FW smoothed-gap diagnostics, when recorded, refer to the current checkpoint and its stated smoothing parameter. They are not a theorem-matched full-trajectory output statistic or an asymptotic claim.
