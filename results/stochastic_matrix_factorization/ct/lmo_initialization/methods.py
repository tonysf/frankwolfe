"""Four optimization methods for a shared nonnegative, Frobenius-capped model.

Each call to problem.grad(U, V, views, block=...) uses a forward and an adjoint
restricted to the supplied view IDs. It returns both partial gradients if
block is None, or one partial gradient for block 0 or 1. Counters include all
refreshes and both terms of SARAH differences.
"""

from copy import deepcopy

import numpy as np


def project_factor(factor, cap):
    """Euclidean projection onto the intersection of the orthant and ball."""
    projected = np.maximum(factor, 0)
    norm = np.linalg.norm(projected)
    if norm > cap:
        projected *= cap / norm
    return projected


def ball_lmo(gradient, cap):
    norm = np.linalg.norm(gradient)
    return gradient * (-cap / norm) if norm else np.zeros_like(gradient)


def _positive(config, name):
    value = float(config[name])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _integer(config, name, lower, upper=None):
    value = config[name]
    if int(value) != value or value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} must be an integer in the supported range")
    return int(value)


class Solver:
    state_fields = ()

    def __init__(self, problem, U, V, caps, *, seed, config):
        self.problem = problem
        self.train_ids = np.asarray(problem.train_ids, dtype=np.intp)
        if self.train_ids.ndim != 1 or not len(self.train_ids):
            raise ValueError("expected a nonempty vector of training view IDs")
        self.config = dict(config)
        self.ru, self.rv = map(float, caps)
        self.U = np.array(U, dtype=np.float64, copy=True)
        self.V = np.array(V, dtype=np.float64, copy=True)
        for factor, cap in ((self.U, self.ru), (self.V, self.rv)):
            if (
                not np.isfinite(cap) or cap <= 0 or not np.all(np.isfinite(factor))
                or np.any(factor < 0) or np.linalg.norm(factor) > cap * (1 + 1e-12)
            ):
                raise ValueError("initial factors must satisfy nonnegativity and their caps")
        self.rng = np.random.default_rng(seed)
        self.k = 0
        self.forward_views = 0
        self.backward_views = 0
        self.gradient_calls = 0

    def sample(self, size):
        if size == len(self.train_ids):
            return self.train_ids
        return self.rng.choice(self.train_ids, size, replace=False)

    def grad(self, U, V, ids, *, block=None):
        gradient = self.problem.grad(U, V, ids, block=block)
        self.forward_views += len(ids)
        self.backward_views += len(ids)
        self.gradient_calls += 1
        return gradient

    def step(self):
        self.forward_views = self.backward_views = self.gradient_calls = 0
        details = self._update()
        if not np.all(np.isfinite(self.U)) or not np.all(np.isfinite(self.V)):
            raise FloatingPointError("nonfinite factor after optimizer update")
        self.k += 1
        return {
            "forward_views": self.forward_views,
            "backward_views": self.backward_views,
            "component_gradient_calls": self.forward_views,
            "gradient_calls": self.gradient_calls,
            **details,
        }

    def state_dict(self):
        names = ("U", "V", "k", *self.state_fields)
        return deepcopy({
            "version": 1,
            "solver": type(self).__name__,
            "config": self.config,
            "caps": (self.ru, self.rv),
            "train_ids": self.train_ids,
            "rng": self.rng.bit_generator.state,
            **{name: getattr(self, name) for name in names},
        })

    def load_state_dict(self, state):
        if (
            state["version"] != 1 or state["solver"] != type(self).__name__
            or state["config"] != self.config or tuple(state["caps"]) != (self.ru, self.rv)
            or not np.array_equal(state["train_ids"], self.train_ids)
            or state["U"].shape != self.U.shape or state["V"].shape != self.V.shape
        ):
            raise ValueError("checkpoint does not match optimizer configuration")
        state = deepcopy(state)
        for name in ("U", "V", "k", *self.state_fields):
            setattr(self, name, state[name])
        self.rng.bit_generator.state = state["rng"]


class SFrames(Solver):
    state_fields = ("mU", "mV")

    def __init__(self, problem, U, V, caps, *, seed, config):
        super().__init__(problem, U, V, caps, seed=seed, config=config)
        self.batch = _integer(config, "batch", 1, len(self.train_ids))
        self.gamma0 = _positive(config, "gamma0")
        self.rho0 = _positive(config, "rho0")
        self.mu0 = _positive(config, "mu0")
        if self.gamma0 > 1 or self.rho0 > 1:
            raise ValueError("SFRAMES gamma0 and rho0 must be at most one")
        self.mU = self.mV = None

    def _update(self):
        grad_u, grad_v = self.grad(self.U, self.V, self.sample(self.batch))
        t = self.k + 1
        gamma = self.gamma0 / t**0.75
        rho = self.rho0 / t**0.5
        mu = self.mu0 / t**0.25
        if self.mU is None:
            self.mU, self.mV = grad_u, grad_v
        else:
            self.mU = (1 - rho) * self.mU + rho * grad_u
            self.mV = (1 - rho) * self.mV + rho * grad_v
        # The orthant Moreau gradient is added after smoothing the stochastic gradient.
        atom_u = ball_lmo(self.mU + np.minimum(self.U, 0) / mu, self.ru)
        atom_v = ball_lmo(self.mV + np.minimum(self.V, 0) / mu, self.rv)
        self.U = self.U + gamma * (atom_u - self.U)
        self.V = self.V + gamma * (atom_v - self.V)
        return {"gamma": gamma, "rho": rho, "mu": mu}


class SpringSARAH(Solver):
    """Alternating SPRING with practical steps and loopless SARAH refreshes.

    Each block has an independent Bernoulli full-gradient refresh with
    probability 1 / refresh_period. Both blocks fully initialize at step one.
    Configured step sizes and refresh period are experimental settings.
    """

    state_fields = ("est_u", "est_v", "previous_u", "previous_v")

    def __init__(self, problem, U, V, caps, *, seed, config):
        super().__init__(problem, U, V, caps, seed=seed, config=config)
        self.batch = _integer(config, "batch", 1, len(self.train_ids))
        self.eta_u = _positive(config, "eta_u")
        self.eta_v = _positive(config, "eta_v")
        period = _positive(config, "refresh_period")
        if period < 1:
            raise ValueError("SPRING refresh_period must be at least one")
        self.refresh_probability = 1 / period
        self.est_u = self.est_v = self.previous_u = self.previous_v = None

    def _estimate(self, current, previous, estimate, block):
        refresh = previous is None or self.rng.random() < self.refresh_probability
        if refresh:
            return self.grad(*current, self.train_ids, block=block), True
        ids = self.sample(self.batch)
        newest = self.grad(*current, ids, block=block)
        prior = self.grad(*previous, ids, block=block)
        return estimate + (newest - prior), False

    def _update(self):
        query_u = self.U, self.V
        est_u, refresh_u = self._estimate(query_u, self.previous_u, self.est_u, 0)
        new_u = project_factor(self.U - self.eta_u * est_u, self.ru)
        query_v = new_u, self.V
        est_v, refresh_v = self._estimate(query_v, self.previous_v, self.est_v, 1)
        new_v = project_factor(self.V - self.eta_v * est_v, self.rv)
        self.U, self.V = new_u, new_v
        self.previous_u, self.previous_v = query_u, query_v
        self.est_u, self.est_v = est_u, est_v
        return {"refresh_u": refresh_u, "refresh_v": refresh_v, "projection_calls": 2}


class ProxSARAH(Solver):
    """Algorithm 1, finite-sum estimator (29), Pham et al., JMLR 2020."""

    state_fields = ("est_u", "est_v", "previous")

    def __init__(self, problem, U, V, caps, *, seed, config):
        super().__init__(problem, U, V, caps, seed=seed, config=config)
        self.batch = _integer(config, "batch", 1, len(self.train_ids))
        self.snapshot_batch = _integer(config, "snapshot_batch", 1, len(self.train_ids))
        self.inner_steps = _integer(config, "inner_steps", 1)
        self.eta = _positive(config, "eta")
        self.relaxation = _positive(config, "relaxation")
        if self.relaxation > 1:
            raise ValueError("ProxSARAH relaxation must be at most one")
        self.est_u = self.est_v = self.previous = None

    def _update(self):
        query = self.U, self.V
        refresh = self.k % (self.inner_steps + 1) == 0
        if refresh:
            est_u, est_v = self.grad(*query, self.sample(self.snapshot_batch))
        else:
            ids = self.sample(self.batch)
            gU, gV = self.grad(*query, ids)
            old_u, old_v = self.grad(*self.previous, ids)
            est_u = self.est_u + (gU - old_u)
            est_v = self.est_v + (gV - old_v)
        prox_u = project_factor(self.U - self.eta * est_u, self.ru)
        prox_v = project_factor(self.V - self.eta * est_v, self.rv)
        self.U = (1 - self.relaxation) * self.U + self.relaxation * prox_u
        self.V = (1 - self.relaxation) * self.V + self.relaxation * prox_v
        self.previous = query
        self.est_u, self.est_v = est_u, est_v
        return {"refresh": refresh, "projection_calls": 2}


class PALM(Solver):
    def __init__(self, problem, U, V, caps, *, seed, config):
        super().__init__(problem, U, V, caps, seed=seed, config=config)
        self.eta_u = _positive(config, "eta_u")
        self.eta_v = _positive(config, "eta_v")

    def _update(self):
        gU = self.grad(self.U, self.V, self.train_ids, block=0)
        new_u = project_factor(self.U - self.eta_u * gU, self.ru)
        gV = self.grad(new_u, self.V, self.train_ids, block=1)
        self.U, self.V = new_u, project_factor(self.V - self.eta_v * gV, self.rv)
        return {"projection_calls": 2}


METHODS = {"sframes": SFrames, "spring_sarah": SpringSARAH, "prox_sarah": ProxSARAH, "palm": PALM}


def make_solver(method, problem, U, V, caps, *, seed, config):
    return METHODS[method](problem, U, V, caps, seed=seed, config=config)
