import numpy as np

from frank_wolfe import (
    BoostedFrankWolfe,
    Frames,
    FrankWolfe,
    ObjectiveFunction,
    create_lmo,
)


class QuadraticObjective(ObjectiveFunction):
    def evaluate(self, x):
        return 0.5 * np.dot(x, x)

    def gradient(self, x):
        return x

    def linear_operator(self, x):
        return x

    def linear_operator_adjoint(self, x):
        return x


def test_l2_lmo_handles_zero_gradient():
    lmo = create_lmo(radius=1.0, constraint_set="l2_ball")

    np.testing.assert_allclose(lmo(np.zeros(3)), np.zeros(3))


def test_nuclear_norm_lmo_preserves_matrix_shape():
    gradient = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    lmo = create_lmo(radius=2.0, constraint_set="nuclear_norm_ball")

    assert lmo(gradient).shape == gradient.shape


def test_basic_frank_wolfe_run_records_history():
    objective = QuadraticObjective()
    lmo = create_lmo(radius=1.0, constraint_set="l2_ball")
    algorithm = FrankWolfe(objective, lmo)

    algorithm.run(np.array([1.0, 0.0]), n_steps=3)

    assert algorithm.x.shape == (2,)
    assert algorithm.func_vals.shape == (3,)
    assert algorithm.gaps.shape == (3,)


def test_boosted_frank_wolfe_runs_with_short_step():
    objective = QuadraticObjective()
    objective.lipschitz = 1.0
    lmo = create_lmo(radius=1.0, constraint_set="l2_ball")
    algorithm = BoostedFrankWolfe(objective, lmo, diam=2.0)

    algorithm.run(np.array([1.0, 0.0]), n_steps=2, K=2, step="short")

    assert algorithm.x.shape == (2,)
    assert algorithm.oracle_calls.shape == (2,)


def test_frames_run_records_history():
    objective = QuadraticObjective()
    lmo = create_lmo(radius=1.0, constraint_set="l2_ball")
    prox = lambda y, beta: np.zeros_like(y)
    algorithm = Frames(objective, lmo, prox, "indicator")

    algorithm.run(np.array([1.0, 0.0]), n_steps=2, show_progress=False)

    assert algorithm.x.shape == (2,)
    assert algorithm.func_vals.shape == (2,)
    assert algorithm.ns_gaps.shape == (2,)
