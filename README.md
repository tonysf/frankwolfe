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

## FRAMES Paper Figures

The paper figure generation entrypoints live in `paper/`.

```bash
python -m paper.generate_main_figures
```

The scripts write generated figures back to `paper/`. Some experiments are
long-running and may require a TeX installation if Matplotlib is configured to
render labels with TeX.
