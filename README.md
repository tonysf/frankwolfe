# Frank-Wolfe Algorithms

This is a personal Python repository for Frank-Wolfe and conditional gradient
methods. A small `paper/` section contains scripts that use the FRAMES pieces to
generate figures for the accompanying paper.

The package currently includes:

- vanilla Frank-Wolfe;
- away-step Frank-Wolfe;
- boosted Frank-Wolfe;
- mismatch Frank-Wolfe;
- FRAMES (Frank-Wolfe with Moreau envelope smoothing);
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


objective = QuadraticObjective()
lmo = create_lmo(radius=1.0, constraint_set="l2_ball")
algorithm = FrankWolfe(objective, lmo)
algorithm.run(np.array([1.0, 0.0]), n_steps=100)
```

## Paper Figures

The paper figure generation entrypoints live in `paper/`.

```bash
python -m paper.generate_main_figures
```

The scripts write generated figures back to `paper/`. Some experiments are
long-running and may require a TeX installation if Matplotlib is configured to
render labels with TeX.

## Tests

Run the smoke tests with:

```bash
python -m pytest
```

## Repository Hygiene

Generated files such as `__pycache__`, `.pyc`, `.DS_Store`, local Matplotlib
caches, build artifacts, and scratch experiment outputs are ignored. The
`examples/codex/` directory is treated as local scratch space and is not part of
the public repository.

## Citation

If you use this code, please cite the accompanying paper. Formal citation
metadata will be added when the paper record is available.
