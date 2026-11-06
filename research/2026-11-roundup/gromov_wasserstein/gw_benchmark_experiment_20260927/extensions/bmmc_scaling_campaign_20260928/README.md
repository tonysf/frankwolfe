# Independently tuned BMMC scaling study

The user authorized submission of the full study after the bounded scaling pilot. This prospectively freezes independent, equal-budget tuning followed by fresh five-seed comparisons. The pilot remains a separate feasibility check; its seed 100 outputs do not enter final statistics.

Both methods use q=128 Rademacher gradient sketches, the same rho(k)=0.3/(k+1)^0.5 momentum schedule, and the same first-gradient initialization/reuse convention. The optimizers are **Smoothed row FW + momentum (sketch gradient)** and **Transport KL-proximal gradient + momentum (sketch gradient)**. Only stochastic competitors enter this study.

## Frozen design

| Stage | Cells | Budget per run | Runs |
|---|---|---:|---:|
| Tuning | 2 sizes × 2 methods × 12 candidates × 3 seeds × 2 starts | 60 solver seconds | 288 |
| Finals | 2 sizes × 2 methods × 5 fresh seeds × 2 starts | 300 solver seconds | 40 |

Sizes are 3000 and 6224 cells per view. Development seeds are 0–2; final seeds are 200–204. Randomized starts use initialization seed 900000+stochastic_seed and the same 0.9 product + 0.1 permutation plan as before. Product initialization records seed 100 but consumes no random draw. Thus each pair shares its exact initial coupling. Both starts receive the same 300-second final budget, avoiding the previous duration/start confound.

FW candidates use gamma0∈{0.1,0.3,1} and m*mu0∈{0.1,1,10,100}, keeping theoretical exponents alpha=.75 and delta=.25. KL candidates use eta∈{0.1,1,3,10,30,100} and marginal tolerance∈{1e-4,1e-6}, with constant eta schedule and inner cap 1000. Both grids include the previous settings and have 12 candidates. The shared estimator/momentum parameters are fixed, not independently tuned.

Select separately for each (size, method) by the median final float64 rounded GW loss over its six development(start,seed) cells. Any invalid output disqualifies the candidate; valid outputs with inner stopping failures remain eligible and those failures remain reported. Exact ties use the candidate ID. Missing/infrastructure/provenance failures block selection; an entirely ineligible group receives no fallback. A single configuration per (size, method) applies to both final starts. No final run begins before all tuning attempts are terminal and a selection file is frozen.

This evaluates limited, equal-budget grids. It does not establish independently optimal settings. Sixty-second development ranking can differ from 300-second final ranking; endpoint choices are fixed in advance.

## Execution and provenance

The unchanged immutable solver runtime is `b5851980646be55a148b083a89cfbf6153dac77e`. Extension files and every job's configuration/manifest/source/input hashes are recorded. The existing runtime's momentum-KL metadata contains an obsolete Stage C statement about inherited plain-KL settings. **The extension manifest, controller receipt, selection and report explicitly supersede that field: momentum-KL is independently tuned here.** No optimizer equation changes are hidden in this reporting correction.

All downstream work waits for pilot 1968437 (a bounded startup retry of 1968346). The controller additionally verifies explicit completion of both sizes, successful canonical preparation reconstruction, validated paired outputs and frozen larger-input hashes. Slurm success alone is insufficient: a resource-gated pilot cannot start the full campaign. Scaling eligibility never depends on which method has lower loss.

Each tuning task contains 12 trials, alternating method order within matched(start,seed) cells. Each final task contains exactly one two-method pair, sequentially on the same A100 allocation, with alternating order across tasks. Every solver is a fresh process. Validation checks paired inputs, initial plans, GPU model, host, driver, CUDA runtime and application versions. This controls host/driver differences within a pair, but is not a dedicated repeated-timing or order-effects study.

Submission dependencies are pilot→CPU eligibility check→tuning array→CPU selection→final array→CPU report. GPU arrays have a maximum of three concurrent tasks, within the user's four-A100 site limit; unrelated jobs are left intact. Tuning requests one A100, two CPUs, 32 GiB and two hours per task; final pairs use the same resources with a two-hour ceiling. CPU selection/report use two CPUs, 4 GiB and a one-hour ceiling. Tuning allows at most 12 × 420 seconds of child-process time within its two-hour allocation; finals at most 2 × 1860 seconds within two hours, leaving setup margin. Unexpected infrastructure failures remain visible and halt dependent work.

Nominal solver budgets sum to 8.1333 GPU-hours (4.8 tuning +3.3333 finals), excluding the pilot, diagnostics, startup and final validation. Peak memory and all timing components remain recorded. Saved raw/rounded couplings total approximately 63 GB before logs/metadata; the inspected Ruche work quota had ample space. No old output is removed or overwritten.

## Measurements and interpretation

Tuning observes initial and 60-second outputs; finals observe initial and 30/100/300-second outputs at actual completed-update times. Missing crossed windows remain missing. The primary endpoint is full, unsampled GW loss at the current copy-rounded output; no hidden polishing or best-iterate selection. Solver-plus-current-repair time is distinct from diagnostics, validation and total process time.

Report full loss versus actual time, probe/update counts, raw marginal residuals, rounding displacement, KL inner counts/failures, underflow and memory. Equal solver time does not mean equal oracle work or the same iteration-indexed momentum coefficient at a given time. Sparse time observations do not establish performance at exactly matched probe counts.

Both sizes use nested subsets of the same real BMMC representations with separate per-size maximum-distance normalization. Absolute losses across sizes are different objectives; compare methods within size. Five optimizer seeds describe stochastic variation on this fixed dataset, not uncertainty over independent biological datasets. RNA/ATAC feature dimensions remain 50/38. No biological alignment claim or growing speed advantage is assumed.

Every GPU stage caps environment validation at 600 seconds with diagnostic stack dumps and persisted startup status. The original pilot startup failure is retained. CPU selection independently validates the saved plans; its one-hour limit allows reading the full coupling archive. No benchmark runs use the failed pilot attempt.

A CPU eligibility job checks the complete pilot before any tuning GPU task can be allocated. Resource-gated or failed pilot outputs therefore block the full array without repeating GPU startup checks.
