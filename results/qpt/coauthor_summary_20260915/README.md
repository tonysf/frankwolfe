# Structured stochastic-FRAMES QPT: summary for coauthors

Experiments completed through 13 September 2026; summary prepared 15 September 2026.

**Main result:** for the existing eight-qubit, rank-one Haar-unitary instance with additive Gaussian observation noise standard deviation **0.05**, we reached **99.00265% normalized factor fidelity at 148,000 iterations**. Independent NumPy calculations using the complete factors verified the result. Trace preservation remains approximate: the final TP residual is **0.1990** in Frobenius norm, or about **1.244% of the norm of the identity**. This is a result for one dataset and seed.

## Convergence across iterations

![Eight-qubit fidelity versus iterations, comparing batches 8,192 and 65,536, with a close-up of the verified 99% result.](figures/n8_fidelity_iterations.png)

[Vector PDF for slides or a paper](figures/n8_fidelity_iterations.pdf). Lines join saved scalar checkpoints without curve fitting. The small-batch trace available locally begins at 30,000 iterations; its earlier interval is not reconstructed. The large-batch trajectory joins the first 100,000-step run to its exact continuation and stops at the first saved checkpoint above 99%.

Both plotted runs use **step size γₖ = 10/(k+10)**, **smoothing βₖ = 10⁷/(k+1)^0.25**, and **new-gradient weight ρₖ = 2/(k+4)^0.6**, with zero-based update index k. The momentum estimate retains weight **1 − ρₖ** on the previous estimate. Rank is 1 and radius τ is 20. Schedules retain their absolute iteration index on restart.

The full eight-qubit table has **24⁸ = 110,075,314,176 possible rows**, represented on demand. Batch 8,192 uses **0.00000744218%** of that row count per iteration; batch 65,536 uses **0.0000595374%**. Sampling is with replacement and each row's observation noise remains fixed. [Complete parameter tables](PARAMETERS.md) cover step size, smoothing, momentum, batch/table ratios and total row draws for the selected four-, five-, eight- and ten-qubit results.

## What changed computationally

We retained the stochastic-FRAMES update and the measurement/TP formulation while implementing ways to avoid large intermediate arrays:

- An on-demand synthetic dataset stores the truth factor, local operators and a fixed noise recipe. Only sampled observations are generated. A row always receives the same noise when revisited; this is not fresh noise at every draw.
- An equivalent product-state measurement backend reduces batch workspace from O(B × 4^n) to O(B × 2^n). It preserves the mathematical measurements and gradients.
- CPU preparation of the next batch overlaps GPU execution. Restart files preserve the factor, momentum, sampling RNG and absolute schedule position.

This made ten-qubit computation feasible on an A100 40GB without storing the roughly 461 TiB full observation table. Recovery accuracy remained a separate challenge. The row-based noise generator defines a different realization from the earlier stored-data generator; existing datasets were preserved.

The validated implementation passed 484 tests plus 13 GPU checks. For the eight-qubit continuation, two 100-step comparisons of preparation chunks of 10 versus 1 step produced bitwise-identical factors, momentum, sampled metrics and sample hashes. Chunk 1 reduced the measured optimizer-plus-blocking-transfer time by 29.34% and was used for the continuation. CPU batch preparation still limited throughput.

## Recovery results

These are selected completed runs with different tuned configurations, not a controlled scaling comparison across qubit counts. Fidelity values below are percentages; TP residuals are unnormalized Frobenius norms.

| Qubits | Noise standard deviation | Iterations | Fidelity | TP residual |
|---:|---:|---:|---:|---:|
| 4 | 0.05 | 10,000 | 99.6561% | 0.00218 |
| 5 | 0.05 | 10,000 | 90.6799% | 0.07742 |
| 8 | 0.05 | 148,000 | **99.00265%** | **0.19901** |
| 10 | 0.05 | 100,000 | 0.004015% | 10.40702 |
| 10 | 0.0015625 | 57,000 | 99.00455% | 0.51053 |

At four and five qubits, increasing the smoothing scale substantially improved recovery. The implemented TP penalty is ||T(uu†) − I||² / (2 beta_k), with beta_k = beta_0/(k+1)^0.25. Increasing beta_0 weakens TP enforcement throughout the schedule. The selected scales were 100 for four qubits and 10,000 for five qubits. Penalized objectives and gaps should not be compared directly across these scales.

The ten-qubit reduced-noise result was independently verified, but its noise standard deviation was 32 times smaller than the original 0.05. At the original noise, the last inspected baseline-continuation checkpoint reached only 0.018935% fidelity at 148,000 steps. That measurement preceded cancellation and is not a terminal result. Larger batches and penalty/schedule pilots did not establish useful ten-qubit recovery within the tested budget. The ten-qubit campaign was stopped.

## Eight-qubit result in detail

With noise fixed at 0.05, the batch-8,192 run improved from 57.19% at 30,000 iterations to 90.89% at 200,000. A fresh batch-65,536 run from the same initial factor reached 98.48565% at 100,000 iterations. Continuing its saved state crossed 99% at the first saved qualifying checkpoint, iteration 148,000.

At a matched budget of 1,638,400,000 sampled row draws, batch 65,536 at 25,000 steps reached 94.16%, versus 90.89% for batch 8,192 at 200,000 steps. This favors the larger-batch configuration at that budget, but does not isolate a single mechanism: schedules depend on iteration count, so their values differ at matched draw counts. Sampling is with replacement; counts are not unique observations.

The successful configuration used rank 1, radius tau = 20, float64/complex128, the product-state backend, and channel, noise, initialization and sampling seeds all set to 0. Its schedules were rho_k = 2/(k+4)^0.6, beta_k = 10^7/(k+1)^0.25, and gamma_k = 10/(k+10). These schedules were unchanged across the eight-qubit batch comparison and continuation. The final trajectory consumed 9,699,328,000 row draws.

![Eight-qubit fidelity versus cumulative row draws and TP residual versus iterations.](figures/n8_sampling_and_tp.png)

[Vector PDF](figures/n8_sampling_and_tp.pdf). The successful run's draws amount to **8.8115% of the full table size**, including repeats; this is not unique-row coverage. The larger-batch run also has higher fidelity at the illustrated matched draw budget, although the iteration-dependent schedules then have different values. TP is evaluated from the complete factor and remains approximate at the final checkpoint.

On one A100, the two successful optimization segments took **21.66 minutes of optimizer time** and **78.32 minutes of runner time**, which includes CPU preparation and checkpointing. Their full batch jobs totaled **80 minutes 4 seconds**, including the short chunk comparison and other setup. These timings describe different scopes.

## Interpretation and evidence

Fidelity is |c†u|²/(||c||² ||u||²), evaluated from complete factors. TP is also evaluated from the complete factor. Measurement loss and gap use a fixed independently sampled set of 512 rows. A small sampled loss alone is not a convergence certificate, and 99% factor fidelity does not imply exact trace preservation. Truth is used for generating synthetic observations, diagnostics and stopping; it does not supply the optimizer's initialization or update direction.

The demonstrated conclusion is that this eight-qubit instance can reach 99% fidelity at the original noise level using larger batches and more iterations. Robustness across datasets/seeds, optimal hyperparameters and a maximum feasible qubit count remain unestablished. Fixed absolute noise also gives different signal-to-noise ratios across sizes. A natural follow-up, if pursued, is replication across seeds together with a separate TP-accuracy criterion.

Evidence in this package includes the [independent eight-qubit verification](n8/final_verification.json), [first 100,000-step trace](n8/first_100k_trace.json), [continuation trace](n8/final_segment_trace.json), [small-batch continuation](n8/small_batch_continuation.json), [run provenance](n8/provenance.json), [chunk comparison](n8/chunk_comparison.json), [ten-qubit outcomes](n10/outcomes.json), and [ten-qubit reduced-noise verification](n10/reduced_noise_verification.json). Earlier four/five-qubit values and original report locations are recorded in [earlier_sizes.json](earlier_sizes.json).

This is a lightweight evidence package. Raw datasets, factor archives and optimizer source code are preserved separately; archive and source checksums are included. No new experiments were run to prepare this summary. The implementation changes have not been committed or pushed.

The plotting update adds PNG/vector PDF figures, [all plot points and input checksums](plot_data.json), [the plotting script](make_plots.py), and [parameter/sampling-budget tables](PARAMETERS.md). Run `python make_plots.py` from this package with NumPy and Matplotlib installed to reproduce the figures. It reads only scalar JSON traces and does not import JAX or execute an optimizer.
