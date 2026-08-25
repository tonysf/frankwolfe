# Paper Figure Scripts

This directory contains the scripts to generate the figures and their
local experiment helpers. They use algorithms from `frank_wolfe/`.

- `generate_main_figures.py` generates the main matrix-factorization and splitting
  figures.
- `generate_nonintersecting_linf_figures.py` generates the inconsistent
  nonintersecting L-infinity example.
- `generate_trend_filtering_trajectory_figure.py` generates the SCAD/MCP
  trend-filtering trajectory comparison.
- `experiments/quantum_process_tomography.py` consumes an HDF5 file from
  [QPT_BFW](https://github.com/LeNavil/QPT_BFW) and runs stochastic FRAMES
  with measurement minibatches and configurable momentum, smoothing, and
  step-size schedules. The adapter was written against upstream commit
  `48e9aa80250e8da065734de593afe549f05912ce`.
- `experiments/quantum_process_tomography_jax.py` runs the same nonconvex
  stochastic-FRAMES formulation with its optimization loop and dense tensor
  contractions JIT-compiled on a selected JAX CPU, GPU, or TPU device.
- `experiments/qpt_bfw_factor_baseline.py` keeps QPT_BFW's original
  Burer--Monteiro rank, operator-norm radius, and fixed TP penalty as a
  formulation-compatible momentum stochastic-FW baseline.

Run scripts from the repository root, for example:

```bash
python -m paper.generate_main_figures
python -m paper.generate_nonintersecting_linf_figures
python -m paper.generate_trend_filtering_trajectory_figure
```

The quantum-process-tomography adapter can be run without JAX or Qiskit once
the source experiment has generated its HDF5 data:

```bash
python -m pip install -e '.[qpt]'
python -m paper.experiments.quantum_process_tomography \
  --h5 jax_arr_gt_xi_0.05.h5 \
  --steps 10000 \
  --rank 1 --tau 10 \
  --batch-size 32 \
  --rho-scale 4 --rho-offset 8 --rho-exponent 0.6666666667 \
  --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
  --step-scale 1 --step-offset 1 --step-exponent 0.5 \
  --metrics-every 100 \
  --save qpt_stochastic_frames.npz \
  --plot qpt_stochastic_frames.png
```

The adapter keeps QPT_BFW's nonconvex Burer--Monteiro factor and operator-norm
ball. At the user's request, it treats the nonlinear trace-preserving map
`A(U) = T(U U^H)` like FRAMES's linear composite map, using the current-point
Jacobian adjoint in place of a fixed linear adjoint. Consequently `beta_k`
actively weights `||A(U)-I||_F^2 / (2 beta_k)`, while only the sampled
measurement-loss gradient enters the momentum estimator. This is an explicit
experimental heuristic, not a claim that `A` is linear.

Full loss, fidelity, and exact gaps are evaluated only at
`--metrics-every` checkpoints, so those passes do not erase the
measurement-minibatch savings inside the optimizer timing. They can still
dominate total post-processing time; use `--metrics-every 0` to record only the
endpoints. At the upstream three-qubit setting, the dense complex128 sensing
tensor alone uses roughly 0.9 GB of RAM. Use `--quiet` for timing runs. Saved
`process_fidelity_proxy` values are the NumPy overlap proxy used by QPT_BFW's
fast path, not Qiskit's exact `process_fidelity` metric.

For schedules that are not power laws, use the Python API. Each callable is
given the zero-based iteration and returns the actual parameter value:

```python
from paper.experiments.quantum_process_tomography import (
    QPTData,
    run_qpt_stochastic_frames,
)

data = QPTData.from_hdf5("jax_arr_gt_xi_0.05.h5")
result = run_qpt_stochastic_frames(
    data,
    n_steps=10_000,
    rank=1,
    tau=10.0,
    batch_size=32,
    rho_schedule=lambda k: min(1.0, 2.0 / (k + 4) ** 0.6),
    smoothing_schedule=lambda k: 10.0 / (k + 1) ** 0.25,
    step_size_schedule=lambda k: 1.0 / (k + 1) ** 0.5,
)
```

## JAX/GPU stochastic FRAMES QPT

The boosted factorized experiment in QPT_BFW's timing notebook used JAX/JIT
and explicitly selected the GPU backend. The JAX adapter restores that device
path for stochastic FRAMES while retaining the same nonconvex factor `U`, the
sampled-gradient momentum estimator initialized by the complete first sample
`d_0`, and the active scheduled trace-preserving smoothing heuristic described
above.

First install the optional experiment dependencies:

```bash
python -m pip install -e '.[qpt-jax]'
```

This installs generic JAX support and is enough for `--device cpu`. For GPU or
another accelerator, follow the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
and install the wheel matching the host CUDA/ROCm/TPU runtime. In particular,
do not select or pin a `jaxlib` build independently of that guidance.

For an A100 checkout, the packaged path is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
./scripts/setup_qpt_jax_gpu.sh --h5 /data/jax_arr_gt_xi_0.05.h5
./scripts/run_qpt_jax_gpu.sh /data/jax_arr_gt_xi_0.05.h5 \
  --steps 10000 --rank 1 --tau 10 --batch-size 32 \
  --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
  --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
  --step-scale 2 --step-offset 2 --step-exponent 1 \
  --metrics-every 100 --save qpt_stochastic_frames_jax.npz --quiet
```

The setup script selects an official CUDA 12/13 JAX wheel from the detected
driver, runs a strict GPU preflight, compiles the actual stochastic-FRAMES scan
kernel, and optionally validates the HDF5 schema. The launcher always enforces
GPU, 64-bit precision, and scan mode while forwarding the experiment and
schedule flags. See `../docs/qpt_a100.md` for fresh-clone and cluster details.

This example requests a GPU and supplies all three power schedules explicitly:

```bash
JAX_ENABLE_X64=1 python -m paper.experiments.quantum_process_tomography_jax \
  --h5 jax_arr_gt_xi_0.05.h5 \
  --steps 10000 \
  --rank 1 --tau 10 \
  --batch-size 32 \
  --device gpu --device-index 0 \
  --precision 64 --execution-mode scan \
  --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
  --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
  --step-scale 2 --step-offset 2 --step-exponent 1 \
  --metrics-every 100 \
  --save qpt_stochastic_frames_jax.npz \
  --plot qpt_stochastic_frames_jax.png \
  --quiet
```

Each CLI schedule is `scale / (iteration + offset)^exponent`; rho and step
size are capped at one. `--device gpu` is strict and fails with a diagnostic if
JAX cannot see a GPU, while `--device auto` prefers GPU, then TPU, then CPU.
The default precision is 64 bits. The runner configures it before creating its
own arrays, and `JAX_ENABLE_X64=1` is recommended so it is also enabled before
any earlier JAX import in the process. Use `--precision 32` only when the
memory/performance tradeoff is worth changing the experiment's numerics.

`--execution-mode scan` fuses the complete optimizer into one `lax.scan`,
synchronizes once, and is the appropriate mode for GPU throughput. Its elapsed
optimizer time is intrinsically available only at the endpoints.
`--execution-mode step` invokes a JIT-compiled update and synchronizes after
every iteration, providing per-step wall-clock timings comparable to the
upstream timing notebook at the cost of Python dispatch and synchronization.
Compilation is warmed up and reported separately by default; pass
`--no-warmup` to include first-call compilation in optimizer timing. A scan
warmup executes and discards one complete compiled scan, so benchmark timing
performs one additional optimizer pass; a step-mode warmup executes and
discards one update.

The Python entry point accepts scalars or arbitrary callables for the same
three schedules. Each callable is evaluated exactly once per zero-based
iteration on the host before JIT tracing:

```python
from paper.experiments.quantum_process_tomography import QPTData
from paper.experiments.quantum_process_tomography_jax import (
    run_qpt_stochastic_frames_jax,
)

data = QPTData.from_hdf5("jax_arr_gt_xi_0.05.h5")
result = run_qpt_stochastic_frames_jax(
    data,
    n_steps=10_000,
    rank=1,
    tau=10.0,
    batch_size=32,
    rho_schedule=lambda k: min(1.0, 2.0 / (k + 4) ** 0.6),
    smoothing_schedule=lambda k: 10.0 / (k + 1) ** 0.25,
    step_size_schedule=lambda k: min(1.0, 2.0 / (k + 2)),
    device="gpu",
    precision="64",
    execution_mode="scan",
)
```

To compare against the source experiment's fixed smooth TP penalty, run:

```bash
python -m paper.experiments.qpt_bfw_factor_baseline \
  --h5 jax_arr_gt_xi_0.05.h5 \
  --steps 10000 \
  --rank 1 --tau 10 --lam 0.05 \
  --batch-size 32 \
  --rho-scale 4 --rho-offset 8 --rho-exponent 0.6666666667 \
  --step-scale 1 --step-offset 1 --step-exponent 0.5 \
  --metrics-every 100 \
  --save qpt_factor_momentum_sfw.npz
```

This comparison runner retains `lambda * ||A(U)-I||_F^2` as part of the smooth
objective. Its accepted smoothing schedule is therefore inert by construction;
the primary runner above is the nonconvex factor-space experiment with active
smoothing.

For numerical source parity, note that QPT_BFW uses `2 / (k + 2)` rather than
the FRAMES step schedule; select it with
`--step-scale 2 --step-offset 2 --step-exponent 1`. The comparison runner
uses the exact SVD polar LMO for rank greater than one, while upstream uses a
polynomial approximation. It reports both true packed-real FW gaps and
`qpt_bfw_*` gap arrays in upstream's half-scaled Wirtinger convention. Batches
are sampled with replacement, so `--batch-size m` is stochastic rather than an
exact full gradient.
