# Eight-qubit QPT at sigma=.05: verified 99% fidelity

The existing rank-one Haar-unitary instance reached **99.002653% factor fidelity at 148,000 total steps**, with Gaussian observation noise standard deviation **.05**. Independent NumPy coefficient-basis and matrix-basis calculations agree with JAX. The TP residual is **0.199007280** in Frobenius norm (about 1.244% of ||I||_F). Trace preservation remains approximate.

![Saved fidelity trajectories and close-up of the 99% crossing.](../coauthor_summary_20260915/figures/n8_fidelity_iterations.png)

[Vector PDF](../coauthor_summary_20260915/figures/n8_fidelity_iterations.pdf), [sampling-budget and TP plots](../coauthor_summary_20260915/figures/n8_sampling_and_tp.png), and [step-size, smoothing, momentum and batch/table parameters](../coauthor_summary_20260915/PARAMETERS.md). Figures use existing scalar traces only; the small-batch trace available locally starts at 30,000 steps.

| Batch | Total steps | Fidelity | TP violation | Sampled row draws |
|---:|---:|---:|---:|---:|
| 8,192 | 30,000 | 57.194503% | 1.356185 | 245,760,000 |
| 8,192 | 200,000 | 90.885265% | 0.483465 | 1,638,400,000 |
| 65,536 | 100,000 | 98.485651% | 0.244526 | 6,553,600,000 |
| 65,536 | 148,000 | 99.002653% | 0.199007 | 9,699,328,000 |

At the matched budget of 1,638,400,000 row draws, batch 65,536 at 25,000 steps reached 94.158%, versus 90.885% for batch 8,192 at 200,000 steps. This compares full configurations: schedules use iteration count, so their values differ at matched sample budgets. Sampling uses replacement; these counts are not unique observations.

The successful trajectory used rank 1, tau 20, the product-state measurement backend, float64/complex128, and initialization, sampling, channel and noise seeds all set to 0. Schedules were unchanged: rho = 2/(k+4)^0.6, beta = 1e7/(k+1)^0.25, gamma = 10/(k+10). The existing on-demand dataset and fixed row noise were reused. The restart at 100,000 steps retained the factor, momentum, RNG and absolute schedule position. No algorithm, objective or CLI default changed. Truth was used for synthetic targets, diagnostics and stopping; optimizer initialization and update directions used the usual procedure.

The two winning optimization segments took **21.66 optimizer minutes** and **78.32 runner minutes**, including CPU preparation and checkpointing in runner time. Their complete batch jobs lasted 58m44s and 21m20s, totaling 80m04s; the latter includes the short chunk comparison. These are separate timing definitions.

The continuation selected a chunk of 1 step after two 100-step comparisons against chunks of 10 steps gave bitwise-identical full factors, momentum, sampled metrics and sample hashes. Mean optimizer time plus blocking transfer time fell from 3.54375s to 2.50404s per 100 steps (29.34%). This percentage applies to the measured phases in this profile. CPU preparation still limits throughput.

All successful jobs completed with exit 0 on an A100 40GB at ruche-gpu15. Initial attempts 1889113_0/1 failed during CUDA initialization on gpu12 before optimization; their evidence remains preserved. Successful comparison jobs: 1889115_0/1. Successful target continuation: 1889210_0. The final queue check was empty. No further jobs or automations were started after reaching the target. No commit, push, switch or pull occurred.

This establishes 99% recovery for this one dataset and seed. Robustness across seeds and optimality of these settings remain untested. Fidelity is |c†u|²/(||c||²||u||²), evaluated using complete factors. Measurement loss uses a fixed independently sampled set of 512 rows and is not a convergence certificate.

[Independent verification](batch65536_200k/verification.json), [final scalar trace](batch65536_200k/summary.json), [source and parent provenance](batch65536_200k/provenance.json), [chunk comparison](batch65536_200k/report.json), [local archive checksums](verified_local_sha256.json), [campaign record](WORK_LOG.md).

The final factor archive, restart and reused dataset are also backed up locally beside this report. Remote originals and earlier results are preserved. The earlier n10 campaign remains stopped and its heartbeat paused.
