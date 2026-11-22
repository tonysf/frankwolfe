"""PALM with the full-data, current-curvature rule used in SPRING Section 5.1.

Each block uses five normalized power iterations followed by a sixth operator
application for the norm estimate, then eta=1/Lhat. The second block uses the
updated first factor. The validated CT Hessian implementation is copied without
modification in curvature_reference.py. Curvature work is charged in full.
"""
from copy import deepcopy

import numpy as np

from .curvature_reference import SpringSARAHLocalCurvature
from .methods import Solver


def parameters(n_train, config=None):
    if not isinstance(n_train, (int, np.integer)) or n_train < 1:
        raise ValueError('PALM requires a positive training-view count')
    expected = dict(power_iterations=5, step_coefficient=1.0, curvature_batch_size=int(n_train))
    if config is not None and any(k not in expected or value != expected[k] for k,value in config.items()):
        raise ValueError('PALM uses five full-data power iterations and the published 1/Lhat step')
    return expected


class PALMPaper(Solver):
    hessian_action = SpringSARAHLocalCurvature.hessian_action

    def __init__(self, problem, U, V, caps, *, seed, config):
        params = parameters(len(problem.train_ids), config)
        super().__init__(problem, U, V, caps, seed=seed, config=params)
        self.power_iterations = params['power_iterations']
        self.curvature_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x50414C4D]))
        self.curvature_operator_calls = 0
        self.curvature_forward_views = self.curvature_backward_views = 0

    def estimate_curvature(self, U, V, ids, block):
        try:
            return SpringSARAHLocalCurvature.estimate_curvature(self, U, V, ids, block)
        except FloatingPointError as error:
            raise FloatingPointError('PALM current block curvature is zero or nonfinite') from error

    def _update(self):
        self.curvature_operator_calls = 0
        self.curvature_forward_views = self.curvature_backward_views = 0
        ids = self.train_ids
        grad_u = self.grad(self.U, self.V, ids, block=0)
        curvature_u = self.estimate_curvature(self.U, self.V, ids, 0)
        eta_u = 1 / curvature_u
        new_u = self.project(self.U - eta_u * grad_u, self.ru)
        grad_v = self.grad(new_u, self.V, ids, block=1)
        curvature_v = self.estimate_curvature(new_u, self.V, ids, 1)
        eta_v = 1 / curvature_v
        new_v = self.project(self.V - eta_v * grad_v, self.rv)
        self.U, self.V = new_u, new_v
        return dict(curvature_u=curvature_u, curvature_v=curvature_v,
            eta_u=eta_u, eta_v=eta_v,
            curvature_sample_views_u=len(ids), curvature_sample_views_v=len(ids),
            curvature_operator_calls=self.curvature_operator_calls,
            curvature_forward_views=self.curvature_forward_views,
            curvature_backward_views=self.curvature_backward_views,
            component_gradient_calls=self.forward_views-self.curvature_forward_views,
            oracle_calls=self.gradient_calls+self.curvature_operator_calls)

    def state_dict(self):
        state = super().state_dict()
        state['curvature_rng'] = deepcopy(self.curvature_rng.bit_generator.state)
        return state

    def load_state_dict(self, state):
        if 'curvature_rng' not in state:
            raise ValueError('PALM checkpoint must contain its curvature RNG')
        super().load_state_dict(state)
        self.curvature_rng.bit_generator.state = deepcopy(state['curvature_rng'])


def make_solver(problem, U, V, caps, *, seed, config):
    return PALMPaper(problem, U, V, caps, seed=seed, config=config)
