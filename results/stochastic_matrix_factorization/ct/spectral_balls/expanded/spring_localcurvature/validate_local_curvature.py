"""Bounded algebra, work-accounting and checkpoint checks; no campaign run."""
import hashlib
import json
from pathlib import Path

import numpy as np

from local_curvature import SpringSARAHLocalCurvature


class DenseProblem:
    def __init__(self):
        rng = np.random.default_rng(913)
        self.train_ids = np.arange(7)
        self.operator = rng.normal(size=(7, 3, 12))
        self.y = rng.normal(size=(7, 3))

    def forward(self, image, ids):
        return self.operator[ids] @ image.ravel()

    def backward(self, observations, ids):
        return np.einsum("vdi,vd->i", self.operator[ids], observations).reshape(4, 3)

    def grad(self, U, V, ids, *, block):
        residual = self.forward(U @ V.T, ids) - self.y[ids]
        adjoint = self.backward(residual, ids) / residual.size
        return adjoint @ V if block == 0 else adjoint.T @ U


class Observed(SpringSARAHLocalCurvature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries = []

    def hessian_action(self, U, V, direction, ids, block):
        self.queries.append((block, U.copy(), V.copy(), ids.copy()))
        return super().hessian_action(U, V, direction, ids, block)


def main():
    problem = DenseProblem()
    rng = np.random.default_rng(177)
    U, V = rng.uniform(.1, .6, (4, 2)), rng.uniform(.1, .6, (3, 2))
    config = dict(batch=3, refresh_period=16, power_iterations=5, step_coefficient=.5)
    def solver(seed=101):
        return Observed(problem, U, V, (5., 5.), seed=seed, config=config)
    s = solver()
    ids = np.array([1, 3, 5])
    errors = []
    for block, variable in enumerate((U, V)):
        # Form the exact block design directly from P and the OTHER factor.
        design_columns = []
        for i in range(variable.size):
            basis = np.zeros_like(variable)
            basis.flat[i] = 1
            image = basis @ V.T if block == 0 else U @ basis.T
            design_columns.append(problem.forward(image, ids).ravel())
        design = np.stack(design_columns, axis=1)
        hessian = design.T @ design / (len(ids) * 3)
        direction = rng.normal(size=variable.shape)
        actual = s.hessian_action(U, V, direction, ids, block)
        expected = (hessian @ direction.ravel()).reshape(variable.shape)
        error = float(np.max(np.abs(actual - expected)))
        errors.append(error)
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
        epsilon = 1e-5
        plus, minus = [U.copy(), V.copy()], [U.copy(), V.copy()]
        plus[block] += epsilon * direction
        minus[block] -= epsilon * direction
        difference = (problem.grad(*plus, ids, block=block) - problem.grad(*minus, ids, block=block)) / (2*epsilon)
        np.testing.assert_allclose(actual, difference, atol=1e-10, rtol=1e-9)
        clone_rng = np.random.default_rng()
        clone_rng.bit_generator.state = s.curvature_rng.bit_generator.state
        vector = clone_rng.standard_normal(variable.shape).ravel()
        vector /= np.linalg.norm(vector)
        for _ in range(5):
            vector = hessian @ vector
            vector /= np.linalg.norm(vector)
        expected_estimate = np.linalg.norm(hessian @ vector)
        before = s.curvature_operator_calls
        estimate = s.estimate_curvature(U, V, ids, block)
        assert s.curvature_operator_calls - before == 6
        np.testing.assert_allclose(estimate, expected_estimate, atol=1e-13, rtol=1e-13)
    s = solver()
    totals = dict(gradient_calls=0, curvature_operator_calls=0, oracle_calls=0, forward_views=0)
    for _ in range(4):
        initial_u, initial_v = s.U.copy(), s.V.copy()
        s.queries.clear()
        row = s.step()
        assert len(s.queries) == row['curvature_operator_calls'] == 12
        for block, u, v, sample in s.queries:
            assert len(sample) == 3
            np.testing.assert_array_equal(v, initial_v)
            np.testing.assert_array_equal(u, initial_u if block == 0 else s.U)
        refreshes = int(row['refresh_u']) + int(row['refresh_v'])
        assert row['gradient_calls'] == 4 - refreshes
        expected_views = refreshes*7 + (2-refreshes)*6 + 36
        assert row['forward_views'] == row['backward_views'] == expected_views
        assert row['component_gradient_calls'] == expected_views - 36
        assert row['curvature_forward_views'] == row['curvature_backward_views'] == 36
        assert row['oracle_calls'] == row['gradient_calls'] + 12
        assert row['projection_calls'] == 2
        for block in ('u','v'):
            np.testing.assert_allclose(row['eta_'+block] * row['curvature_'+block], .5, atol=1e-15, rtol=0)
        for name in totals:
            totals[name] += row[name]
    restored = solver()
    restored.load_state_dict(s.state_dict())
    for _ in range(3):
        assert s.step() == restored.step()
        np.testing.assert_array_equal(s.U, restored.U)
        np.testing.assert_array_equal(s.V, restored.V)
    try:
        s.estimate_curvature(U, np.zeros_like(V), ids, 0)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Zero curvature must be rejected explicitly')
    root = Path(__file__).resolve().parent
    result = dict(status='passed', checks=['explicit dense block Hessians', 'central gradient differences',
        'published five power iterations plus sixth norm', 'alternating block points',
        'minibatch curvature on full refresh', 'inclusive views and separate oracle counts',
        'bitwise checkpoint continuation of both RNG streams', 'zero curvature fails explicitly'],
        max_dense_hessian_error=max(errors), checked_step_totals=totals,
        source_sha256=hashlib.sha256((root/'local_curvature.py').read_bytes()).hexdigest())
    (root/'algebra_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
