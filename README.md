# Frank-Wolfe Algorithms

This is a simple Python repository for Frank-Wolfe or conditional gradient
methods. A small `paper/` section contains scripts that use the FRAMES pieces to
generate figures for the accompanying paper studying the FRAMES algorithm.

The package currently includes:

- vanilla Frank-Wolfe;
- away-step Frank-Wolfe;
- boosted Frank-Wolfe;
- mismatch Frank-Wolfe;
- FRAMES (Frank-Wolfe with Moreau envelope smoothing);
- adaptive FRAMES with gap-triggered smoothing reductions;
- stochastic FRAMES with momentum-averaged stochastic gradients;
- conditional gradient sliding;
- linear minimization oracles and small utility projections.

## Installation

Create an environment with Python 3.9 or newer, then install the package in
editable mode:

```bash
python -m pip install -e .
```

The core dependencies are NumPy, SciPy, Matplotlib, and tqdm.

## Quick Start

```python
import numpy as np

from frank_wolfe import FrankWolfe, ObjectiveFunction, create_lmo


class QuadraticObjective(ObjectiveFunction):
    def evaluate(self, x):
        return 0.5 * np.dot(x, x)

    def gradient(self, x):
        return x

    def linear_operator(self, x):
        return x

    def linear_operator_adjoint(self, y):
        return y


objective = QuadraticObjective()
lmo = create_lmo(radius=1.0, constraint_set="l2_ball")
algorithm = FrankWolfe(objective, lmo)
algorithm.run(np.array([1.0, 0.0]), n_steps=100)
```

## Adaptive FRAMES

`AdaptiveFrames` uses the usual FRAMES open-loop step size while adapting the
Moreau smoothing parameter to the observed smoothed Frank-Wolfe gap:

```python
from frank_wolfe import AdaptiveFrames


prox_zero_penalty = lambda y, beta: y
algorithm = AdaptiveFrames(
    objective, lmo, prox_zero_penalty, objective_type="indicator"
)
algorithm.run(
    np.array([1.0, 0.0]),
    beta0=1.0,
    n_steps=100,
    step_size_schedule=lambda k: 2.0 / (k + 2),
)
```

At iteration `k`, the method uses the current `beta_k`. If the resulting
Frank-Wolfe gap of the Moreau-smoothed objective is strictly below `beta_k`,
then `beta_{k+1} = beta_k / 2`; otherwise it remains constant. Equality does
not trigger a reduction. The step size defaults to
`gamma_k = 1 / sqrt(k + 1)`; `step_size_schedule` accepts either a scalar or a
callable of the zero-based iteration returning a value in `[0, 1]`. The
realized values are available in `smoothing_parameters` and `step_sizes`, and
`next_smoothing_parameter` records the beta value that would be used after the
final completed iteration.

## Stochastic FRAMES

`StochasticFrames` replaces the exact gradient of the smooth term with a
one-sample momentum estimate. The objective can own its random generator and
implement `stochastic_gradient(x)`:

```python
from frank_wolfe import StochasticFrames


class StochasticQuadraticObjective(QuadraticObjective):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def stochastic_gradient(self, x):
        return x + self.rng.normal(scale=0.1, size=x.shape)

    def linear_operator(self, x):
        return x

    def linear_operator_adjoint(self, y):
        return y


objective = StochasticQuadraticObjective(seed=7)
prox_zero_penalty = lambda y, beta: y
algorithm = StochasticFrames(
    objective, lmo, prox_zero_penalty, objective_type="indicator"
)
algorithm.run(
    np.array([1.0, 0.0]),
    n_steps=100,
    rho_schedule=lambda k: 1.0 / (k + 1) ** 0.4,
    smoothing_schedule=lambda k: 0.5 / (k + 1) ** 0.2,
    step_size_schedule=lambda k: 2.0 / (k + 2),
)
```

The initial point is used directly and must belong to the Frank-Wolfe feasible
set. Each schedule accepts either a scalar constant or a callable of the
zero-based iteration returning the actual parameter value. With no overrides,
the defaults are `rho_k = 4 / (k + 8)^(2/3)`,
`beta_k = beta0 / (k + 1)^(1/4)`, and
`gamma_k = 1 / (k + 1)^(1/2)`. A custom `smoothing_schedule` replaces the
`beta0` law instead of being multiplied by it. The realized values are stored
in `momentum_weights`, `smoothing_parameters`, and `step_sizes`.
`estimated_gaps` records gaps formed from the momentum gradient estimate, and
`num_stochastic_oracles` records cumulative stochastic-gradient calls.
The first sample initializes `d_0` exactly; `rho_schedule(0)` is validated and
recorded, but momentum averaging begins with `rho_schedule(1)`.

An adapter for the QPT_BFW quantum-process-tomography experiment is available
in `paper/experiments/quantum_process_tomography.py`. It loads the source HDF5
format without JAX or Qiskit, keeps the nonconvex factor `chi = U U^H`,
subsamples measurements for the momentum estimator, and exposes all three
schedules through both Python and CLI APIs. To make smoothing active in factor
space, the nonlinear trace-preserving map is deliberately supplied through the
point-dependent adjoint hook as though it were FRAMES's linear composite map.
`paper/experiments/qpt_bfw_factor_baseline.py` additionally preserves the
source's fixed-`lambda` smooth penalty for comparison.

An optional JAX implementation of the same stochastic-FRAMES experiment lives
in `paper/experiments/quantum_process_tomography_jax.py`. The source QPT_BFW
timing notebook used JAX/JIT and explicitly selected a GPU; this runner keeps
the nonconvex factorization and scheduled momentum, smoothing, and step-size
updates on a selected JAX device. Install its Python dependencies with:

```bash
python -m pip install -e '.[qpt-jax]'
```

That command is sufficient for a CPU run. For an accelerator, install the JAX
wheel appropriate for the machine's CUDA/ROCm/TPU runtime by following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html);
the project deliberately does not pin `jaxlib` to a particular accelerator
build. Use `--device gpu` for a strict GPU request: it raises an error rather
than silently falling back when no GPU backend is available. The default
`--precision 64` matches the source data; setting `JAX_ENABLE_X64=1` before
launching ensures 64-bit mode is enabled before any JAX import.

Use `--execution-mode scan` for a fused `lax.scan` throughput run, or
`--execution-mode step` for a synchronized, per-iteration timing convention
like the upstream timing notebook. See `paper/README.md` for a complete GPU
command with user-selected power schedules and for the Python API.

### A100 quick start

The `adaptive-frames` branch includes a strict GPU installer, an end-to-end
preflight, and a launcher. The repository has a large historical Git object
store, so use a shallow branch clone on the experiment machine:

```bash
git clone --depth 1 --branch adaptive-frames \
  https://github.com/tonysf/frankwolfe.git
cd frankwolfe
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

The setup script uses the NVIDIA driver version to select the official CUDA
12 or CUDA 13 JAX extra, then compiles and synchronizes both a complex128
kernel and a tiny stochastic-FRAMES scan on the GPU. Set `PYTHON_BIN` to
choose another Python interpreter or `JAX_CUDA_VARIANT=cuda12|cuda13` to make
the CUDA choice explicit. Current GPU setup requires Python 3.12 or newer.

The generated QPT_BFW HDF5 data is not distributed by either repository; copy
it to the machine separately. The preflight's `--h5` option validates its
schema before a long run. See [docs/qpt_a100.md](docs/qpt_a100.md) for the
complete deployment and troubleshooting guide.

## FRAMES Paper Figures

The paper figure generation entrypoints live in `paper/`.

```bash
python -m paper.generate_main_figures
```

The scripts write generated figures back to `paper/`. Some experiments are
long-running and may require a TeX installation if Matplotlib is configured to
render labels with TeX.
