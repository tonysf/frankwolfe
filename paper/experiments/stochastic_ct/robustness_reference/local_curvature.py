"""SPRING--SARAH with the paper's practical, current-block step-size rule.

Section 5.1 uses eta=1/(2*Lhat). The authors' power_method.m starts
from a fresh normalized Gaussian direction, performs five normalized normal
operator applications, then a sixth application for its norm estimate.
Here the normal operator is the CT minibatch block Hessian, with the same
mean-loss normalization. Recursive steps reuse the gradient-difference sample;
full-gradient refresh steps draw a separate curvature minibatch, as Section
5.1 prescribes minibatch curvature for the stochastic methods at each iteration.

The formal loopless SARAH recursion and independent block refreshes are
retained. This is the published practical step rule adapted to CT, not a
literal port of the authors' different, epoch-based SNMF experiment driver.
Curvature work is included in total forward/adjoint views and oracle calls;
gradient_calls continues to count only actual gradient evaluations.
"""

from copy import deepcopy

import numpy as np

from .methods import Solver, _integer, _positive


class SpringSARAHLocalCurvature(Solver):
    state_fields = ("est_u", "est_v", "previous_u", "previous_v")

    def __init__(self, problem, U, V, caps, *, seed, config):
        super().__init__(problem, U, V, caps, seed=seed, config=config)
        self.batch = _integer(config, "batch", 1, len(self.train_ids))
        self.power_iterations = _integer(config, "power_iterations", 1)
        self.step_coefficient = _positive(config, "step_coefficient")
        if self.power_iterations != 5 or self.step_coefficient != 0.5:
            raise ValueError("This baseline implements the published five-iteration, 0.5/L rule")
        period = _positive(config, "refresh_period")
        if period < 1:
            raise ValueError("SPRING refresh_period must be at least one")
        self.refresh_probability = 1 / period
        self.est_u = self.est_v = self.previous_u = self.previous_v = None
        # Independent stream avoids changing gradient sampling solely because
        # curvature estimation now draws random starting directions.
        self.curvature_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53505249]))
        self.curvature_operator_calls = 0
        self.curvature_forward_views = self.curvature_backward_views = 0

    def _estimate(self, current, previous, estimate, block):
        refresh = previous is None or self.rng.random() < self.refresh_probability
        if refresh:
            ids = self.train_ids
            return self.grad(*current, ids, block=block), True, ids
        ids = self.sample(self.batch)
        newest = self.grad(*current, ids, block=block)
        prior = self.grad(*previous, ids, block=block)
        return estimate + (newest - prior), False, ids

    def hessian_action(self, U, V, direction, ids, block):
        """Apply the Hessian of f_ids in the selected factor, charging all work."""
        if block == 0:
            image_direction = direction @ V.T
        elif block == 1:
            image_direction = U @ direction.T
        else:
            raise ValueError("Expected factor block 0 or 1")
        projected = self.problem.forward(image_direction, ids)
        adjoint = self.problem.backward(projected, ids).astype(np.float64) / projected.size
        result = adjoint @ V if block == 0 else adjoint.T @ U
        self.forward_views += len(ids)
        self.backward_views += len(ids)
        self.curvature_forward_views += len(ids)
        self.curvature_backward_views += len(ids)
        self.curvature_operator_calls += 1
        return result

    def estimate_curvature(self, U, V, ids, block):
        shape = U.shape if block == 0 else V.shape
        direction = self.curvature_rng.standard_normal(shape)
        direction /= np.linalg.norm(direction)
        for _ in range(self.power_iterations):
            action = self.hessian_action(U, V, direction, ids, block)
            norm = float(np.linalg.norm(action))
            if not np.isfinite(norm) or norm <= 0:
                raise FloatingPointError("SPRING power iteration has zero or nonfinite block curvature")
            direction = action / norm
        estimate = float(np.linalg.norm(self.hessian_action(U, V, direction, ids, block)))
        if not np.isfinite(estimate) or estimate <= 0:
            raise FloatingPointError("SPRING has zero or nonfinite local curvature")
        return estimate

    def curvature_sample(self, gradient_ids, refresh):
        if not refresh or self.batch == len(self.train_ids):
            return gradient_ids
        return self.curvature_rng.choice(self.train_ids, self.batch, replace=False)

    def _update(self):
        self.curvature_operator_calls = 0
        self.curvature_forward_views = self.curvature_backward_views = 0
        query_u = self.U, self.V
        est_u, refresh_u, ids_u = self._estimate(query_u, self.previous_u, self.est_u, 0)
        curvature_ids_u = self.curvature_sample(ids_u, refresh_u)
        curvature_u = self.estimate_curvature(*query_u, curvature_ids_u, 0)
        eta_u = self.step_coefficient / curvature_u
        new_u = self.project(self.U - eta_u * est_u, self.ru)
        query_v = new_u, self.V
        est_v, refresh_v, ids_v = self._estimate(query_v, self.previous_v, self.est_v, 1)
        curvature_ids_v = self.curvature_sample(ids_v, refresh_v)
        curvature_v = self.estimate_curvature(*query_v, curvature_ids_v, 1)
        eta_v = self.step_coefficient / curvature_v
        new_v = self.project(self.V - eta_v * est_v, self.rv)
        self.U, self.V = new_u, new_v
        self.previous_u, self.previous_v = query_u, query_v
        self.est_u, self.est_v = est_u, est_v
        return {
            "refresh_u": refresh_u, "refresh_v": refresh_v,
            "curvature_u": curvature_u, "curvature_v": curvature_v,
            "eta_u": eta_u, "eta_v": eta_v,
            "curvature_sample_views_u": len(curvature_ids_u),
            "curvature_sample_views_v": len(curvature_ids_v),
            "curvature_operator_calls": self.curvature_operator_calls,
            "curvature_forward_views": self.curvature_forward_views,
            "curvature_backward_views": self.curvature_backward_views,
            "component_gradient_calls": self.forward_views - self.curvature_forward_views,
            "oracle_calls": self.gradient_calls + self.curvature_operator_calls,
        }

    def state_dict(self):
        state = super().state_dict()
        state["curvature_rng"] = deepcopy(self.curvature_rng.bit_generator.state)
        return state

    def load_state_dict(self, state):
        if "curvature_rng" not in state:
            raise ValueError("Local-curvature checkpoint must include its power-iteration RNG")
        super().load_state_dict(state)
        self.curvature_rng.bit_generator.state = deepcopy(state["curvature_rng"])


def make_solver(problem, U, V, caps, *, seed, config):
    return SpringSARAHLocalCurvature(problem, U, V, caps, seed=seed, config=config)
