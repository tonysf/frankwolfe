"""Small FP64 matrix-completion experiment; numerical routines only."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Problem:
    Y: np.ndarray
    signal: np.ndarray
    mask: np.ndarray
    test_mask: np.ndarray
    tau: float
    lower: float = -1.0
    upper: float = 1.0

    def __post_init__(self):
        self.indices = np.flatnonzero(self.mask.ravel())
        self.count = len(self.indices)
        assert self.lower == -1 and self.upper == 1
        assert self.Y.dtype == np.float64 and self.tau > 0 and self.count > 0

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            return cls(**{k: d[k].copy() if d[k].ndim else float(d[k])
                          for k in ['Y', 'signal', 'mask', 'test_mask', 'tau', 'lower', 'upper']})

    def loss(self, x):
        r = (x-self.Y)[self.mask]
        return float(r @ r / (2*self.count))

    def gradient(self, x):
        return np.where(self.mask, x-self.Y, 0.) / self.count

    def sample_gradient(self, x, draws):
        """draws are IID uniform indices in the list of observed entries."""
        flat = self.indices[draws]
        g = np.zeros(x.size, dtype=np.float64)
        np.add.at(g, flat, (x.ravel()[flat]-self.Y.ravel()[flat])/len(draws))
        return g.reshape(x.shape)

    def penalty_gradient(self, x, beta):
        return (x-np.clip(x, self.lower, self.upper))/beta

    def lmo(self, direction):
        if not np.any(direction):
            return np.zeros_like(direction)
        u, _, vh = np.linalg.svd(direction, full_matrices=False)
        return -self.tau * np.outer(u[:, 0], vh[0])

    def true_gap(self, x, beta):
        h = self.gradient(x)+self.penalty_gradient(x, beta)
        return float(np.sum(h*x)+self.tau*np.linalg.svd(h, compute_uv=False)[0])

    def feasible_repair(self, x):
        # Radial contraction preserves the nuclear ball and enforces [-1,1].
        return x/max(1., float(np.max(np.abs(x))))

    def metrics(self, x, beta):
        repaired = self.feasible_repair(x)
        violation = x-np.clip(x, self.lower, self.upper)
        return dict(raw_loss=self.loss(x), feasible_loss=self.loss(repaired),
                    box_violation_fro=float(np.linalg.norm(violation)),
                    box_violation_max=float(np.max(np.abs(violation))),
                    nuclear_norm=float(np.linalg.svd(x, compute_uv=False).sum()),
                    feasible_nuclear_norm=float(np.linalg.svd(repaired, compute_uv=False).sum()),
                    feasible_max_abs=float(np.max(np.abs(repaired))),
                    repair_fro=float(np.linalg.norm(x-repaired)),
                    true_smoothed_gap=self.true_gap(x, beta),
                    clean_test_rmse=float(np.sqrt(np.mean((repaired[self.test_mask]-self.signal[self.test_mask])**2))))


def schedule(method, iteration, beta0, gamma0=1., rho0=1.):
    if method == 'sframes':
        k = iteration+1
        return gamma0/k**.75, rho0/k**.5, beta0/k**.25
    if method == 'shcgm':
        # Algorithm 1 uses k=1 for the first update.
        k = iteration+1
        return 9/(k+8), 4/(k+7)**(2/3), beta0/(k+8)**.5
    raise ValueError(method)


def update(problem, x, momentum, sampled_gradient, *, method, iteration, beta0, gamma0=1., rho0=1.):
    gamma, rho, beta = schedule(method, iteration, beta0, gamma0, rho0)
    if momentum is None:
        momentum = sampled_gradient.copy() if method == 'sframes' else rho*sampled_gradient
    else:
        momentum = (1-rho)*momentum+rho*sampled_gradient
    direction = momentum+problem.penalty_gradient(x, beta)
    s = problem.lmo(direction)
    return (1-gamma)*x+gamma*s, momentum
