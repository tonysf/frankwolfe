# Run parameters and sampling budgets

These settings describe the selected runs in the coauthor summary. The two plotted eight-qubit runs use the same original-noise dataset and the same schedules. All plots are made from existing scalar checkpoints, without new optimization. The small-batch trace available locally begins at iteration 30,000; the first 30,000 iterations are not reconstructed.

## Schedules and their meaning

Let `k = 0, 1, ...` index optimizer updates. A checkpoint at iteration `t` is the state after `t` updates; its last applied update used `k = t − 1`. Native continuations preserve this absolute clock.

| Parameter | General schedule | Both plotted n = 8 runs |
|:--|:--|:--|
| Frank–Wolfe step size, γₖ | `min(1, step_scale / (k + step_offset)^step_exponent)` | `10 / (k + 10)` |
| TP smoothing, βₖ | `smoothing_scale / (k + smoothing_offset)^smoothing_exponent` | `10^7 / (k + 1)^0.25` |
| New-gradient weight, ρₖ | `min(1, rho_scale / (k + rho_offset)^rho_exponent)` | `2 / (k + 4)^0.6` |

The caps on step size and ρ do not alter these selected formulas for `k ≥ 0`. **There is no smoothing cap.** The objective uses TP penalty `||T(uu†) − I||_F² / (2βₖ)`. Thus larger β weakens TP enforcement; its slow decay strengthens the penalty over time. Changing the smoothing scale changes the entire schedule.

The measurement-gradient estimate follows `dₖ = (1 − ρₖ)dₖ₋₁ + ρₖ ĝₖ`, where `ĝₖ` is the gradient from the current batch. At `k = 0`, the implementation initializes `d₀ = ĝ₀` directly. **ρ is the weight on the new gradient; the retained momentum weight is 1 − ρ.** The exact TP gradient is added after this averaging, divided by β. If `sₖ` is the resulting linear-minimization atom, the factor update is `uₖ₊₁ = (1 − γₖ)uₖ + γₖsₖ`.

For reference, the actual schedule values for the eight-qubit runs are:

| Update | γₖ | βₖ | ρₖ, new-gradient weight |
|:--|--:|--:|--:|
| First, k = 0 | 1 | 10,000,000 | 0.870551* |
| Last before 100,000-step checkpoint, k = 99,999 | 0.0000999910 | 562,341.325 | 0.00199996 |
| Last before 148,000-step success, k = 147,999 | 0.0000675635 | 509,840.781 | 0.00158077 |
| Last before 200,000-step small-batch endpoint, k = 199,999 | 0.0000499978 | 472,870.805 | 0.00131950 |

*The first gradient is assigned directly, so its scheduled ρ value is not used for averaging. Saved checkpoint β may describe the next update at `k = t`; the table above explicitly reports the last applied value. The successful continuation's saved schedule arrays were [checked against these formulas](n8/schedule_verification.json).

## Selected configurations across sizes

All selected runs have rank 1. In this table, `βₖ = β₀/(k+1)^0.25` and `ρₖ = 2/(k+4)^0.6` for every row. These were tuned configurations, not a controlled comparison across qubit counts.

| Qubits | Noise σ | Batch B | Total iterations T | Step size γₖ | Smoothing scale β₀ | Radius τ | Final fidelity |
|--:|--:|--:|--:|:--|--:|--:|--:|
| 4 | 0.05 | 32 | 10,000 | `2/(k+2)` | 100 | 10 | 99.6561% |
| 5 | 0.05 | 32 | 10,000 | `2/(k+2)` | 10,000 | 10 | 90.6799% |
| 8 | 0.05 | 8,192 | 200,000 | `10/(k+10)` | 10,000,000 | 20 | 90.8853% |
| **8** | **0.05** | **65,536** | **148,000** | **`10/(k+10)`** | **10,000,000** | **20** | **99.00265%** |
| 10 | 0.05 | 65,536 | 100,000 | `10/(k+10)` | 1,000,000,000 | 40 | 0.004015% |
| 10 | 0.0015625 | 65,536 | 57,000 | `10/(k+10)` | 1,000,000,000 | 40 | 99.00455% |

The four/five-qubit matched smoothing sweep also tested β₀ = 10, 100, 1,000 and 10,000 with all other listed settings fixed. This table selects the settings emphasized in the summary. Ten-qubit continuation and penalty pilots are not plotted here; their campaign remains stopped. The ten-qubit reduced-noise run uses 32 times smaller noise standard deviation and is not a success at σ = 0.05.

## Batch size versus the full table

There are `M = 24^n` possible scalar measurement rows. Four/five-qubit compact datasets store the noisy observation table. Eight/ten-qubit synthetic datasets represent it virtually: the truth, local operators and a fixed row-noise recipe are stored, and sampled observations are generated on demand. **Stored observations = 0 does not mean that M is zero.** Repeated visits to the same row use the same noisy target.

Sampling is uniform with replacement. One iteration uses B draws; T iterations use `B × T` draws, including repeats. `B × T / M` is a draw budget relative to table size, not the fraction of distinct rows observed. It excludes diagnostic/metric draws and separate profiling or smoke runs.

| Qubits | Full table M | Batch B | B / M (%) | Total iterations | Total optimizer row draws | Draws / M (%) |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | 331,776 | 32 | 0.00964506 | 10,000 | 320,000 | 96.4506 |
| 5 | 7,962,624 | 32 | 0.000401878 | 10,000 | 320,000 | 4.01878 |
| 8 | 110,075,314,176 | 8,192 | 0.00000744218 | 200,000 | 1,638,400,000 | 1.48844 |
| **8** | **110,075,314,176** | **65,536** | **0.0000595374** | **148,000** | **9,699,328,000** | **8.81154** |
| 10 | 63,403,380,965,376 | 65,536 | 0.000000103364 | 100,000 | 6,553,600,000 | 0.0103364 |
| 10 | 63,403,380,965,376 | 65,536 | 0.000000103364 | 57,000 | 3,735,552,000 | 0.00589172 |

At an equal budget of 1,638,400,000 draws, the eight-qubit batch-65,536 run reaches **94.1584% at 25,000 iterations**, while batch 8,192 reaches **90.8853% at 200,000**. The larger batch therefore does better at this matched budget too. Because the schedules depend on iterations, their values differ between these two checkpoints; this does not isolate a pure batch-variance effect.

## Other settings and reproducibility

The eight-qubit runs use float64/complex128, the product-state backend and one A100 40GB. Channel, noise, initialization and sampling seeds are all 0. The original noise standard deviation is 0.05, with a deterministic fixed Gaussian realization indexed by row. The batch-65,536 run starts fresh from the same original initial factor as the small-batch lineage and resumes its complete state at 100,000 steps. The small-batch run resumes its earlier 30,000-step state. Factor, momentum, RNG and schedules are preserved on restart.

Measurement loss and gap use an independently sampled fixed set of 512 rows, metric seed 12345 and metric batch size 512. Metrics are saved every 1,000 iterations in these n = 8 runs. Fidelity and TP use complete factors; they are not computed on those 512 measurement rows. The 99% stop is checked at saved checkpoints. The batch-65,536 run was allowed to reach 200,000 total iterations but stopped at 148,000. Its TP residual is 0.199007 in Frobenius norm, so trace preservation remains approximate.

CPU prefetch is enabled. Preparation chunk size is 10 for the small-batch continuation and the first 100,000 large-batch steps, and 1 thereafter. The chunk choice was checked to preserve bitwise-identical factors, momentum and sampled metrics in the comparison runs. Chunk size controls preparation/execution grouping and is distinct from the stochastic batch size B. GPU autotuning was disabled with `XLA_FLAGS=--xla_gpu_autotune_level=0`.

Parameter evidence: [large-batch command/provenance](n8/provenance.json), [small-batch command/provenance](n8/small_batch_provenance.json), [n10 continuation settings](n10/continuation_parameters.json), and the recorded earlier experiment handoff for the n = 4/5 settings. The earlier handoff values have not been re-read from Ruche for this plotting update. [Plot data](plot_data.json) includes input checksums; [make_plots.py](make_plots.py) reproduces both PNG and vector PDF figures using NumPy and Matplotlib.
