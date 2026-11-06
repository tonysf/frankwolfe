# On-demand noisy structured QPT

The explicit `synthetic-noisy` mode stores a Haar-unitary truth factor, local
operators, and a random-access noise recipe. It never enumerates or allocates
the `24**n` observations. Existing stored datasets and CLI defaults retain
their original behavior.

For each sampled legacy row `s`, the target is
`real(vdot(D_s, c c^H)) + sigma * z_s`. The GPU evaluates the truth response
inside the same bounded scan as the optimizer. The CPU generates only the
small array of fixed row noise for the next chunk; GPU target generation is
included in optimizer timing. No asynchronous CPU prefetch is enabled.

## Fixed virtual noise

`splitmix64_box_muller_row_v1` maps the unsigned 64-bit noise seed and legacy
row index to two domain-separated SplitMix64 outputs, then applies the
Box-Muller transform in NumPy float64. A row always has the same noise,
independent of sampling order, repeats, chunk size, and metric cadence.
There is no cache whose memory grows with rows visited. The row index uses
all its bits; supported virtual datasets fit signed int64 (up to 13 qubits).

This is a new pseudorandom realization, different from the sequential PCG64
noise in the earlier stored archives. It preserves the fixed noisy empirical
problem interpretation. Generating fresh independent noise at every revisit
would instead define an online-noise experiment and is not this mode.
Cross-platform elementary functions need not produce bitwise-identical noise.
Archives record the algorithm version, seeds, standard deviation, truth hash,
and source hashes. New code using another recipe must use another version.

## Allocated usage

Run both truth construction and GPU work inside scheduler allocations. These
commands are the payload inside an already prepared allocation, not a Slurm
submission recipe. Every output path must be new.

```bash
python -m paper.experiments.qpt_generate_data \
  --n-qubits 10 --observation-mode synthetic-noisy \
  --noise-std 0.05 --channel-seed 0 --noise-seed 0 \
  --save /new/output/n10.npz
```

Use the existing structured benchmark with `--data /new/output/n10.npz` and
the explicit opt-in `--allow-synthetic-noisy`. The benchmark otherwise rejects
non-stored observations as before. It reports zero stored observation bytes
and `observations_preserved=false`; source/truth provenance is still checked.
The direct structured runner also accepts the new archive mode.

At rank one, choose `tau >= sqrt(2**n)` to contain the TP truth and the current
initializer. `tau=10` only suffices through six qubits. Record all schedules
and the radius explicitly; increasing the qubit count is not a reason to
silently change defaults or normalize the mathematical objective.

## What to measure

`scripts/profile_qpt_components.py` separately times truth target evaluation,
measurement gradients with ready targets, generated targets plus measurement
gradients, and exact TP gradients. It also times fixed-noise preparation for a
100-step host chunk. Isolated kernel timings include dispatch/synchronization;
compilation and one warmup call are excluded. They are not additive timings
for the fused optimizer scan and do not establish its precise time breakdown.

The tensor backend and existing rank-one backend evaluate the same operators;
their floating-point trajectories can diverge. Run their profiles separately.
Use the main benchmark's fresh worker repetitions and separate memory run for
whole-optimizer timing, peak process RSS, JAX allocator memory, and sampled
driver VRAM. Those memory definitions remain distinct.

The data-size wall is removed, but one rank-one factor still has `4**n`
complex entries. A tensor measurement batch costs `O(B*n*4**n)` arithmetic;
the rank-one backend costs `O(B*4**n)`. Exact TP costs
`O(n*4**n + 8**n)` and is evaluated at every optimizer step.

Computational feasibility is separate from recovery. A general unitary has
approximately `4**n` real degrees of freedom, and a batch-32, 10,000-step run
visits only 320,000 measurement rows. With Haar truth, mean outcome probability
is `1/2**n`, so fixed absolute noise `0.05` changes relative noise with size.
A control using `sigma_n = 0.05 * 2**(5-n)` preserves n=5's ratio of noise
standard deviation to mean probability; label it separately from fixed-noise
runs. Neither sample count nor noise scaling guarantees optimizer convergence.

## Validation

`tests/test_qpt_on_demand.py` checks an independent scalar noise reference,
high row-index bits, repeated/permuted/batched access, distribution sanity,
archive round trips, and generation guards. It compares complete small
optimizer trajectories and metrics against explicitly materialized noisy
observations for both tensor and rank-one backends, then changes chunk and
metric boundaries. `QPT_TEST_DEVICE=gpu` runs those trajectory checks on the
allocated GPU. An integration test covers the explicit benchmark opt-in.
