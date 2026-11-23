import numpy as np
import pytest

from frank_wolfe import AdaptiveFrames, AdaptiveFramesFrankWolfe
from frank_wolfe.algorithms import (
    AdaptiveFrames as AlgorithmsAdaptiveFrames,
)
from frank_wolfe.algorithms import (
    AdaptiveFramesFrankWolfe as AlgorithmsAdaptiveFramesFrankWolfe,
)
from frank_wolfe.core.objective import ObjectiveFunction


class ScriptedGapObjective(ObjectiveFunction):
    def __init__(self, target_gaps, directions):
        self.target_gaps = list(target_gaps)
        self.directions = [np.asarray(value, dtype=float) for value in directions]
        self.gradient_calls = 0

    def evaluate(self, x):
        return 0.5 * np.vdot(x, x).real

    def gradient(self, x):
        if self.gradient_calls == 0:
            gradient = np.zeros_like(x, dtype=float)
        else:
            iteration = self.gradient_calls - 1
            displacement = x - self.directions[iteration]
            gradient = np.asarray(
                [self.target_gaps[iteration] / displacement.item()]
            )
        self.gradient_calls += 1
        return gradient

    def linear_operator(self, x):
        return x

    def linear_operator_adjoint(self, x):
        return x


class ScriptedLMO:
    def __init__(self, initial_point, directions):
        self.outputs = iter(
            [np.asarray(initial_point, dtype=float)]
            + [np.asarray(value, dtype=float) for value in directions]
        )

    def __call__(self, gradient):
        return next(self.outputs).copy()


def identity_prox(y, beta):
    return y


class ZeroObjective(ObjectiveFunction):
    def evaluate(self, x):
        return 0.0

    def gradient(self, x):
        return np.zeros_like(x)

    def linear_operator(self, x):
        return x

    def linear_operator_adjoint(self, x):
        return x


def test_adaptive_frames_exports_and_gap_triggered_beta_plateaus():
    assert AdaptiveFrames is AdaptiveFramesFrankWolfe
    assert AlgorithmsAdaptiveFrames is AdaptiveFrames
    assert AlgorithmsAdaptiveFramesFrankWolfe is AdaptiveFramesFrankWolfe

    target_gaps = [1.25, 1.0, 0.75, 0.6, 0.49, 0.3]
    directions = [[0.0], [-1.0], [1.0], [-1.0], [1.0], [-1.0]]
    objective = ScriptedGapObjective(target_gaps, directions)
    algorithm = AdaptiveFrames(
        objective,
        ScriptedLMO([1.0], directions),
        identity_prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]),
        beta0=1.0,
        n_steps=len(target_gaps),
        show_progress=False,
    )

    np.testing.assert_allclose(algorithm.gaps, target_gaps)
    np.testing.assert_allclose(
        algorithm.smoothing_parameters,
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.25],
    )
    expected_steps = 1.0 / np.sqrt(np.arange(len(target_gaps)) + 1)
    np.testing.assert_allclose(algorithm.step_sizes, expected_steps)
    expected_x = np.array([1.0])
    for step_size, direction in zip(expected_steps, directions):
        expected_x = (
            (1.0 - step_size) * expected_x
            + step_size * np.asarray(direction)
        )
    np.testing.assert_allclose(algorithm.x, expected_x)
    assert algorithm.next_smoothing_parameter == 0.25


def test_moreau_smoothed_gap_controls_beta_and_final_trigger_is_recorded():
    seen_betas = []

    def project_to_zero(y, beta):
        seen_betas.append(beta)
        return np.zeros_like(y)

    directions = [[0.0], [-1.0]]
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        ScriptedLMO([1.0], directions),
        project_to_zero,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]), beta0=1.0, n_steps=2, show_progress=False
    )

    # At the first iteration, the smooth gap is zero but the Moreau-smoothed
    # gap equals beta. The strict test therefore keeps beta unchanged.
    np.testing.assert_allclose(algorithm.gaps, [1.0, 0.0])
    np.testing.assert_allclose(algorithm.smoothing_parameters, [1.0, 1.0])
    np.testing.assert_allclose(seen_betas, algorithm.smoothing_parameters)
    assert algorithm.next_smoothing_parameter == 0.5
    np.testing.assert_allclose(algorithm.x, [-1.0 / np.sqrt(2.0)])


def test_adaptive_beta_state_resets_between_runs():
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        lambda gradient: np.zeros_like(gradient),
        identity_prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]), beta0=1.0, n_steps=2, show_progress=False
    )
    np.testing.assert_allclose(algorithm.smoothing_parameters, [1.0, 0.5])

    algorithm.run(
        np.array([0.0]), beta0=2.0, n_steps=2, show_progress=False
    )
    np.testing.assert_allclose(algorithm.smoothing_parameters, [2.0, 1.0])


def test_scalar_step_size_schedule_is_applied_and_recorded():
    directions = [[0.0], [-1.0]]
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        ScriptedLMO([1.0], directions),
        identity_prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]),
        n_steps=2,
        step_size_schedule=0.25,
        show_progress=False,
    )

    np.testing.assert_allclose(algorithm.step_sizes, [0.25, 0.25])
    np.testing.assert_allclose(algorithm.x, [0.3125])


def test_callable_step_size_schedule_is_zero_based_and_evaluated_once():
    calls = []
    values = [0.0, 0.25, 0.5]

    def schedule(iteration):
        calls.append(iteration)
        return values[iteration]

    directions = [[2.0], [4.0], [6.0]]
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        ScriptedLMO([1.0], directions),
        identity_prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]),
        n_steps=3,
        step_size_schedule=schedule,
        show_progress=False,
    )

    assert calls == [0, 1, 2]
    np.testing.assert_allclose(algorithm.step_sizes, values)
    np.testing.assert_allclose(algorithm.x, [3.875])


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_step_size_value_is_rejected(value):
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        lambda gradient: np.zeros_like(gradient),
        identity_prox,
        "indicator",
    )

    with pytest.raises(ValueError, match=r"finite value in \[0, 1\]"):
        algorithm.run(
            np.array([0.0]),
            n_steps=1,
            step_size_schedule=value,
            show_progress=False,
        )


@pytest.mark.parametrize(
    "value", [np.array([0.5]), 0.5 + 0.1j, "0.5"]
)
def test_step_size_schedule_must_return_a_real_scalar(value):
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        lambda gradient: np.zeros_like(gradient),
        identity_prox,
        "indicator",
    )

    with pytest.raises(TypeError, match="step_size_schedule.*iteration 0"):
        algorithm.run(
            np.array([0.0]),
            n_steps=1,
            step_size_schedule=lambda iteration: value,
            show_progress=False,
        )


def test_step_size_schedule_argument_must_be_scalar_callable_or_none():
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        lambda gradient: np.zeros_like(gradient),
        identity_prox,
        "indicator",
    )

    with pytest.raises(TypeError, match="step_size_schedule"):
        algorithm.run(
            np.array([0.0]),
            n_steps=1,
            step_size_schedule=[0.5],
            show_progress=False,
        )


@pytest.mark.parametrize("beta0", [0.0, -0.1, np.nan, np.inf])
def test_adaptive_frames_rejects_invalid_beta0_before_oracles(beta0):
    oracle_calls = 0

    class CountingObjective(ObjectiveFunction):
        def gradient(self, x):
            nonlocal oracle_calls
            oracle_calls += 1
            return np.zeros_like(x)

    algorithm = AdaptiveFrames(
        CountingObjective(),
        lambda gradient: np.zeros_like(gradient),
        identity_prox,
        "indicator",
    )

    with pytest.raises(ValueError, match="beta0"):
        algorithm.run(
            np.array([0.0]), beta0=beta0, n_steps=1, show_progress=False
        )

    assert oracle_calls == 0


def test_adaptive_frames_reports_beta_underflow():
    algorithm = AdaptiveFrames(
        ZeroObjective(),
        lambda gradient: np.zeros_like(gradient),
        identity_prox,
        "indicator",
    )

    with pytest.raises(FloatingPointError, match="underflow"):
        algorithm.run(
            np.array([0.0]),
            beta0=np.nextafter(0.0, 1.0),
            n_steps=1,
            show_progress=False,
        )


def test_adaptive_frames_rejects_a_nonfinite_smoothed_gap():
    class HugeGradientObjective(ZeroObjective):
        def gradient(self, x):
            return np.full_like(x, 1e308)

    algorithm = AdaptiveFrames(
        HugeGradientObjective(),
        ScriptedLMO([2.0], [[-2.0]]),
        identity_prox,
        "indicator",
    )

    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="gap must be finite"):
            algorithm.run(
                np.array([0.0]), n_steps=1, show_progress=False
            )
