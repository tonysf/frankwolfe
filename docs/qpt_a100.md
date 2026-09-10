# A100 deployment: stochastic FRAMES for QPT

For the compact tensor-product runner, streaming HDF5 conversion, and dense
parity benchmarks, see [Structured QPT](qpt_structured.md).

This guide deploys the nonconvex factor-space stochastic-FRAMES QPT experiment
on one visible NVIDIA A100. The optimizer stays in the Burer--Monteiro factor
`U`; only the sampled measurement-loss gradient enters the momentum estimator,
and the first sample initializes `d_0` exactly.

## What you need

- A Linux A100 host with a working NVIDIA driver.
- Python 3.12 or newer for the current JAX GPU wheels.
- Network access to PyPI during setup, or an equivalent pre-populated package
  cache.
- A QPT_BFW-compatible HDF5 file. Neither this repository nor upstream checks
  generated data into Git. The required datasets are `f_jax_vector`,
  `D_jax_tensors`, and `A_jax_basis`; `Chi_star_tensor` and `B_jax_tensors` are
  optional.

The upstream data-generation notebook is
[`qutomo_gt_gen.ipynb`](https://github.com/LeNavil/QPT_BFW/blob/48e9aa80250e8da065734de593afe549f05912ce/qutomo_gt_gen.ipynb).
Generate the file there or copy an existing file to the A100 machine.

## Fresh clone and GPU setup

The repository has a large historical object store. A shallow clone avoids
downloading unrelated history:

```bash
git clone --depth 1 --branch stochastic-frames \
  https://github.com/tonysf/frankwolfe.git
cd frankwolfe

python3.12 -m venv .venv
source .venv/bin/activate
./scripts/setup_qpt_jax_gpu.sh --h5 /data/jax_arr_gt_xi_0.05.h5
```

`setup_qpt_jax_gpu.sh` installs the editable `qpt-jax` project extra and the
official JAX accelerator extra. With `JAX_CUDA_VARIANT=auto`, it chooses:

- `jax[cuda13]` for NVIDIA driver 580 or newer;
- `jax[cuda12]` for NVIDIA driver 525 through 579;
- no installation for an older or unparsable driver.

Override the choices when necessary:

```bash
PYTHON_BIN=/opt/python3.12/bin/python3.12 \
JAX_CUDA_VARIANT=cuda12 \
./scripts/setup_qpt_jax_gpu.sh --h5 /data/jax_arr_gt_xi_0.05.h5
```

The setup is successful only after the selected GPU executes and synchronizes
a complex128 JIT kernel and a two-step stochastic-FRAMES `lax.scan`. Supplying
`--h5` additionally loads the real dataset and reports its tensor shapes,
dtypes, and host-memory footprint.

You can rerun the preflight without reinstalling anything:

```bash
JAX_ENABLE_X64=1 python \
  scripts/check_qpt_jax_environment.py \
  --device gpu --device-index 0 \
  --h5 /data/jax_arr_gt_xi_0.05.h5
```

## Run the experiment

The launcher forces the GPU, complex128/float64 numerics, and fused scan mode.
Every remaining CLI argument is forwarded to the experiment:

```bash
./scripts/run_qpt_jax_gpu.sh /data/jax_arr_gt_xi_0.05.h5 \
  --steps 10000 \
  --rank 1 --tau 10 \
  --batch-size 32 \
  --initialization-seed 0 --sampling-seed 0 \
  --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
  --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
  --step-scale 2 --step-offset 2 --step-exponent 1 \
  --metrics-every 100 \
  --save qpt_stochastic_frames_jax.npz \
  --plot qpt_stochastic_frames_jax.png \
  --quiet
```

Each CLI schedule has the form
`scale / (iteration + offset)^exponent`; rho and step size are capped at one.
Use the Python API for arbitrary callable schedules.

The default warmup compiles, executes, and discards one complete scan before
the timed scan. This produces clean synchronized timings but doubles the
optimizer work for that run. Add `--no-warmup` for a single execution whose
reported optimizer time includes first-call compilation.

The result archive records the realized schedules, sampled measurement
indices, initialization and sampling settings, JAX/jaxlib versions, device
kind, precision, compilation time, and synchronized optimizer time.

## Cluster notes

- `--device-index 0` means the first device visible to the process. A scheduler
  may remap a physical A100 through `CUDA_VISIBLE_DEVICES`.
- The runner currently uses one JAX device. Request one GPU per process.
- The three-qubit sensing tensor is large; keep the HDF5 file on fast local or
  scratch storage and allow several GiB of host RAM during loading and transfer.
- JAX normally preallocates GPU memory. If the node is shared, follow the
  cluster's policy before changing JAX memory-allocation environment variables.

## Fast failure checks

JAX must report a GPU, not merely import successfully:

```bash
JAX_ENABLE_X64=1 python -c \
  'import jax; print(jax.__version__); print(jax.devices("gpu"))'
```

If the setup reports only CPU devices, confirm that `nvidia-smi` works inside
the job/container and that installation and execution use the same Python
interpreter. If HDF5 validation fails, regenerate or transfer the file rather
than starting a long optimization with a partial dataset.
