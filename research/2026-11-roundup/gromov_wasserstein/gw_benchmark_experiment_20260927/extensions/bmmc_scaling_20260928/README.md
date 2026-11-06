# BMMC scaling pilot

This prospective extension follows the user's request to investigate larger real-data problems after observing competitive FW solver-time performance on BMMC n=1000. It reopens the previously deferred larger-instance work for this bounded pilot. Original Stage A–C records, the original protocol and its historical gate remain unchanged.

The question is whether the low cost of smoothed FW updates produces better feasible loss within a fixed solver-time budget as the transport problem grows. A favorable outcome is not assumed. Both methods use the same stochastic objective-gradient oracle, so its increasing cost can also reduce the relative benefit of a cheap update.

## Frozen scope

- One paired product-start trial at n=3000, followed by one pair at n=6224 if resource/validity checks pass. These contain 9,000,000 and 38,738,176 coupling entries respectively.
- Smoothed row FW + momentum (sketch gradient), and Transport KL-proximal gradient + momentum (sketch gradient). No deterministic optimizer competitor.
- Stochastic seed 100, product-start seed 100 (unused); q=128, rho0=.3, beta=.5 and first-gradient reuse for both methods.
- FW retains gamma0=1, alpha=.75, m*mu0=10, delta=.25. KL retains eta10, step exponent0, marginal tolerance1e-4 and 1,000 inner iterations maximum.
- 300 solver seconds each; initial and 30/100/300-second observations, actual completion times, full unsampled GW evaluations, copy rounding, and final float64 validation. No oracle-budget matching or interpolation.
- Both methods run sequentially as fresh processes in the same one-A100 Slurm allocation. FW goes first at 3000; KL goes first at 6224. This is not a controlled order-effect study.

The pilot uses the already preserved runtime `b5851980646be55a148b083a89cfbf6153dac77e`, which contains the completed momentum diagnostics. It does not alter optimizer equations or the frozen Stage C releases. `pilot.json` and the transferred file checksums bind the extension scripts. All solver counters, residuals, momentum diagnostics, numerical failures, underflow, solver/repair/diagnostic/validation/process times and allocator peaks remain recorded. The reported peak includes instrumentation and is not total device memory.

## Data and preparation

Reuse the exact two public BMMC H5AD files already on Ruche, checking their original SHA256 values. Read only dense `GEX_X_pca` and `ATAC_lsi_red` arrays with an isolated h5py 3.15.1 wheel; no additional biological labels enter the computation. The validated PyTorch/NumPy/SciPy environment remains unchanged.

Before generating larger instances, reconstruct all six canonical n=1000 arrays and require bitwise equality to the existing archive. This checks that the direct HDF5 reader and installed numerical libraries reproduce the original preprocessing. The preparation code then uses the same row-normalized ATAC representation, independent permutation prefixes with seeds 2026/2027, ordinary Euclidean distances, per-matrix maximum scaling, float32 distance storage, and float64 uniform marginals. Nested cell sets do not imply identical distance normalization across sizes. Compare methods within each size; absolute losses across sizes are not a common objective.

“Larger dimension” here means more cells and coupling variables, with the original 50 RNA and 38 ATAC features retained. No biological alignment-quality claim follows from the GW loss alone.

## Resource and interpretation limits

Use Ruche account `fwllm`, partition `gpua100`, one A100, two CPUs, 32 GiB host RAM, and a two-hour walltime ceiling. Each child solver has a 30-minute process timeout, capped by the allocation deadline minus a five-minute bookkeeping reserve. The scheduler deadline is read before dependency installation. Attempt records are saved before launch and retain timeout/failure evidence. No GPU computation or preprocessing runs on the login node. Existing unrelated jobs remain untouched.

Advance to 6224 only after both 3000 outputs complete final validation and serialization, share the same initial coupling, and the measured GPU-reserved/host-peak memory scaled by (6224/3000)^2 stays below 80% of the allocations. Require at least 4,200 seconds remaining for the two child limits, preparation and final bookkeeping. This memory projection is a screening heuristic, not a fit guarantee. Any failure or resource gate stops the extension and retains its evidence; no favorable-loss gate selects what runs next.

One seed per size establishes feasibility and preliminary scaling evidence, not a replicated performance result. The constants transfer from the smaller study; momentum-KL was not independently tuned. A later replicated/tuned study needs its own prospective specification after this pilot is inspected. No such sweep is launched automatically.

GPU allocator peaks cover the solver plus expensive exact diagnostics and float64 postprocessing. Dense storage scales quadratically; the common exact diagnostics include cubic matrix multiplications. They are timed separately rather than hidden in a solver speedup claim. Gap diagnostics requiring a full assignment LMO stay restricted to n ≤ 512, as in the existing runner.

Infrastructure retry: job1968346 stopped during environment validation at its180-second timeout, before data preparation or either solver. A new immutable launcher allows600seconds for that check and records diagnostic stack dumps every120seconds; all scientific settings and stage gates are unchanged. The failed attempt is preserved separately.
