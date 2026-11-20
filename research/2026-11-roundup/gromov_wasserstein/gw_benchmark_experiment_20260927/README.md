> Completed experiment: [stochastic comparison](reports/stochastic_primary/report.md), [full experiment account](reports/EXPERIMENT_REPORT.md), [method definitions](reports/METHOD_NAMES.md), [reproduction instructions](EXPERIMENT.md). The original starter documentation follows below.

# GW optimization benchmark starter

This package accompanies `PROTOCOL.md`. It supplies reference problem kernels,
instance preparation, tests, and a small baseline driver. It does **not** implement
the user's unpublished method or choose theorem-valid schedules for it.

## Contents

- `PROTOCOL.md`: experiment instructions, equations, sources and reporting rules.
- `gw_core.py`: full off-feasible objective and gradient; two fixed-distribution
  stochastic-gradient oracles; row LMO; marginal rounding; Euclidean projections;
  log-domain KL-prox step; exact assignment LMO and gap for square uniform cases.
- `test_gw_core.py`: 12 deterministic float64 correctness tests.
- `make_instances.py`: synthetic metrics or fixed BMMC representations to NPZ.
- `run_reference.py`: small FW, KL-prox, stochastic KL-prox and Euclidean PG runs.
- `requirements.txt`: bootstrapping dependencies, not a publication lockfile.

## Start

Install an appropriate PyTorch build for your CPU or CUDA host, then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python make_instances.py synthetic --n 128 --seed 2000 --noise 0 --out data/toy128.npz
python run_reference.py --instance data/toy128.npz --method fw --steps 100 --threads 8 --out runs/fw128
python run_reference.py --instance data/toy128.npz --method kl --eta 10 --steps 100 --threads 8 --out runs/kl128
python run_reference.py --instance data/toy128.npz --method stochastic-kl --eta 1 --batch 32 --steps 100 --threads 8 --out runs/skl128
```

For GPU pilot runs, add `--device cuda --dtype float32 --inner-tol 1e-6` after
checking your CUDA environment. For small Euclidean PG, use `--method pg --eta
0.01`; here `--inner-tol` means a **projection primal-dual gap**, whereas for KL
methods it means a **sum of marginal L1 residuals**. They are not interchangeable.

Use `--init mixed-permutation` for a positive, randomized feasible initialization.
`--seed` controls both its independent initialization seed and the optimization
random stream. A publication harness should store them as separate fields.

## Scope and testing

The 12 tests and five-step, 16-by-16 smoke runs for all four driver methods passed
in the delivery environment: Python 3, NumPy 2.3.5, SciPy 1.17.0 and CPU-only
PyTorch 2.10.0. These are correctness checks, **not** optimization performance
results. GPU execution and real-data downloads/preprocessing have not been tested
here. POT native solver calls are instructions in the protocol, not a tested
adapter in this package. The recent Liang--Yang iPG and BAPG are not implemented.

The driver deliberately computes exact diagnostics often. It records solver,
repair and diagnostic times separately. For publishable runtime comparisons,
replace this with common elapsed-time checkpoints and an independently timed
benchmark harness as described in the protocol. It saves raw and rounded outputs;
rounding never feeds back into the trajectory, except as the feasible primal
candidate used inside the explicitly separate Dykstra projection baseline.

`sketch_gradient` is a newly derived reference oracle in this protocol, not the
published Sampled GW estimator. Check its compatibility with your full theorem.
The scalar sampled losses can be negative; their expectation is the original GW
objective. Do not confuse this with replacing the objective by a fixed sketch.

The code stores `A`, `B`, `A**2`, and `B**2`. Account for all four, as well as
couplings, momentum, log-state and matrix-product scratch space, in memory reports.
No licenses or upstream source code are bundled; the reference kernels were
written for this response. Cite the algorithm/data sources in the protocol when
using those methods or data.
