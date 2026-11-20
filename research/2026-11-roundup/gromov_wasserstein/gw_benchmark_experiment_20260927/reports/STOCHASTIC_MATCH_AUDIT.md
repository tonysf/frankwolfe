# Audit of the primary stochastic comparison

**All 110 final instance/seed/start pairs match on the prescribed stochastic oracle, momentum schedule, initialization, precision policy, and solver-time budget.** The pair is **Smoothed row FW + momentum (sketch gradient)** (`smoothing-fw`) versus **Transport KL-proximal gradient + momentum (sketch gradient)** (`stochastic-kl-momentum`). This is a comparison between two stochastic momentum methods. Exact-gradient methods can remain contextual controls without being the primary stochastic comparator.

This is a read-only audit of existing records and frozen source, not a new experiment or performance ranking. The [machine-readable audit](STOCHASTIC_MATCH_AUDIT.json) preserves every pair, metadata/trace hashes, checks, counts, and environmental differences. No optimizer, objective evaluation, array reconstruction, or remote computation was run.

## Coverage and matching

There are 55 product-start pairs at 300 solver seconds and 55 randomized-start pairs at 30 solver seconds: five clean synthetic problems, five noisy synthetic problems, and one fixed BMMC dataset, each with five stochastic seeds. All 220 method-runs have valid saved final outputs. The audit checked 220 metadata hashes, 660 archived source-file hashes, and 1,316 observed trace rows, including 220 initial rows.

| Item | Verified matching condition |
|---|---|
| Objective and inputs | Same frozen instance and A/B/a/b array hashes within every pair. Reported losses use the full original GW objective. |
| Gradient estimator | The same `GWProblem.sketch_gradient` function and Rademacher probe generator; q=128 fresh probes per oracle call. No exact objective-gradient calls enter either optimizer. |
| Momentum | ρₖ=0.3/(k+1)^0.5, for k≥1, with dₖ=(1−ρₖ)dₖ₋₁+ρₖgₖ. |
| First gradient | Both set d₀=g₀(P₀) and reuse it on update zero; neither draws a second initial sample or applies ρ₀ to the initial estimate. |
| Sample accounting | At initialization: one oracle call and 128 probes. After N≥1 completed updates: N oracle calls and 128N probes, including initialization. No extra terminal oracle call. These equalities hold in all final counters and observed traces. |
| Stochastic seeds | 100–104, matched within each pair, using an explicit device Torch generator. |
| Product initialization | Same P₀=a bᵀ hash; recorded initialization seed 100 consumes no randomness. |
| Randomized initialization | Same P₀=0.9a bᵀ+0.1R/n hash and permutation within every pair; initialization seeds 900100–900104 use a separate NumPy generator. |
| Budgets and checkpoints | Identical requested budgets and target grids within every pair; every final run reaches its prescribed solver-time budget. Actual output times and update overshoots remain recorded. |
| Compute policy | CUDA float32, A100-SXM4-40GB, PyTorch 2.5.1+cu121/CUDA 12.1, deterministic settings, highest matmul precision, AMP/TF32 disabled; intra-op/BLAS/OpenMP threads one and PyTorch inter-op threads 32. |
| Source | Both methods ran the same frozen source revision `af7b7c549351b81de2aa886b987bc3e56ffeff9c`; archived runner and gradient/kernel files match that revision. |

The initialization and oracle RNGs are separate. The shared seed, device RNG, probe shape, and sampling routine imply the same probe-sequence prefix at the same oracle index. This is a source/configuration inference: individual probe matrices were not archived and were not reconstructed here. Gradient values differ once the iterate paths diverge. Randomized panels pair one initialization seed with one stochastic seed; they do not independently isolate initialization and sampling effects.

Of 1,100 prescribed individual checkpoint windows, 1,096 are observed and four are explicitly missing. Both methods have observations at 546/550 paired opportunities. Missing windows must stay visible; no interpolation or replacement with a later endpoint is warranted.

## What is deliberately different

The FW method uses the row constraint set, a current exact column-constraint penalty, row-wise linear minimization, and a scheduled FW step. The KL method targets both marginals through an iterative KL-proximal solve and has no column penalty. Only the objective-gradient estimate enters the EMA in both methods. This is not an ablation in which just one operator was changed.

The KL-momentum comparator inherits η=10, step exponent zero, inner tolerance 10⁻⁴ and cap 1,000 from the selected **plain stochastic KL** candidate `c09`. Its ρ₀=0.3 comes from selected smoothing candidate `c08`; it was **not independently tuned with momentum enabled**. The comparison therefore does not establish what a separately optimized stochastic momentum-KL method could achieve. All inheritance was frozen before final optimization.

The separately tuned **Smoothed row projection + momentum (sketch gradient)** control uses ρ₀=1 with the same exponent 0.5, rather than ρ₀=0.3. It is not matched on the primary pair's momentum constant. Plain stochastic KL has no EMA; native Sampled GW uses a different estimator and update implementation. Their roles should be distinguished from this primary pair.

## Timing and environment limits

Equal prescribed solver time is not equal oracle work or equal completed iterations. It also does not put the iteration-indexed momentum schedule at the same ρ value at a given wall-clock checkpoint. The recorded completed-update ranges illustrate the difference:

| Start / prescribed budget | Smoothed row FW, completed updates | Momentum KL, completed updates |
|---|---:|---:|
| Product / 300 s | 158,373–170,919 | 1,877–2,742 |
| Randomized / 30 s | 11,964–16,835 | 324–429 |

Every update consumes the same 128-probe batch, so these are also unequal total-sample budgets. Actual terminal solver times span 300.000005–300.001824 s and 300.014247–300.243237 s for the product FW/KL runs, respectively; randomized spans are 30.000072–30.001841 s and 30.008970–30.245489 s. Feasible-output plots add the repair time for that actual output. Diagnostics and other overhead remain separately reported; equal solver budgets do not mean equal total process time.

The GPU model, application libraries, CUDA runtime and precision policy match, but the complete machine environments do not. **25/110 pairs** (11 product, 14 randomized) differ in recorded OS/NVIDIA driver revision: 580.105.08 versus 580.167.08, with slightly different reported available GPU memory. Sixty-four pairs ran on different physical hostnames. The JSON lists every affected pair. No controlled same-host/same-driver timing claim is supported by this audit.

All outputs remain valid, but **103/110 momentum-KL runs report inner stopping failures**, totaling 57,996 such events. The 110 row-FW runs have no corresponding iterative inner solve. Inner stopping failures must remain visible alongside feasible quality and timing; they are not erased by a valid rounded output.

## Evidence binding

The authoritative input is `reports/stage_c_final/report.json`, SHA-256 `39730329a43e14f0ce3af063f8263bb5b6b45e6acc45696ccbff164813d9086c`. The audit also binds the four original product/randomized manifests, the frozen selection, all 220 actual metadata/trace files, and archived source. The source review covered `make_initial_plan`, `_kl_momentum_gradient`, initialization and update branches in the frozen `run_benchmark.py`; oracle initialization, sampling, EMA and row updates in `smoothing_fw.py`; and the shared sketch/probe functions in `gw_core.py`.

No scientific matching defect was found. The environmental differences and tuning/work-budget limitations above prevent describing the comparison as identical in every respect or as a fully isolated test of update geometry.
