import numpy as np
import pytest

from frank_wolfe import (
    ObjectiveFunction,
    StochasticFrames,
    StochasticFramesFrankWolfe,
)
from frank_wolfe.algorithms import (
    StochasticFrames as AlgorithmsStochasticFrames,
)
from frank_wolfe.algorithms import (
    StochasticFramesFrankWolfe as AlgorithmsStochasticFramesFrankWolfe,
)


class ScriptedStochasticObjective(ObjectiveFunction):
    def __init__(self, gradients, minimal_norm_value=0.0):
        self.set_gradients(gradients)
        self.minimal_norm_value = minimal_norm_value
        self.gradient_calls = 0
        self.stochastic_gradient_calls = 0

    def set_gradients(self, gradients):
        self.gradients = [
            np.asarray(gradient, dtype=float) for gradient in gradients
        ]

    def evaluate(self, x):
        return 0.5 * np.sum(np.asarray(x) ** 2)

    def gradient(self, x):
        self.gradient_calls += 1
        raise AssertionError("stochastic FRAMES must not request the exact gradient")

    def stochastic_gradient(self, x):
        self.stochastic_gradient_calls += 1
        if not self.gradients:
            raise AssertionError("the scripted stochastic gradients were exhausted")
        return self.gradients.pop(0).copy()

    def linear_operator(self, x):
        return np.asarray(x)

    def linear_operator_adjoint(self, x):
        return np.asarray(x)

    def minimal_norm_selection(self, x):
        return np.full_like(x, self.minimal_norm_value, dtype=float)


class RecordingLMO:
    def __init__(self):
        self.gradients = []

    def __call__(self, gradient):
        gradient = np.asarray(gradient)
        self.gradients.append(gradient.copy())
        return np.zeros_like(gradient)


class RecordingProx:
    def __init__(self):
        self.parameters = []

    def __call__(self, y, beta):
        self.parameters.append(beta)
        return np.asarray(y)


def identity_prox(y, beta):
    return np.asarray(y)


def test_stochastic_frames_exports_and_counts_one_gradient_oracle_per_step():
    assert StochasticFrames is StochasticFramesFrankWolfe
    assert AlgorithmsStochasticFrames is StochasticFrames
    assert AlgorithmsStochasticFramesFrankWolfe is StochasticFramesFrankWolfe

    objective = ScriptedStochasticObjective([[1.0], [2.0], [3.0]])
    lmo = RecordingLMO()
    algorithm = StochasticFrames(objective, lmo, identity_prox, "indicator")

    algorithm.run(
        np.array([1.0]),
        n_steps=3,
        rho_schedule=1.0,
        show_progress=False,
    )

    assert objective.gradient_calls == 0
    assert objective.stochastic_gradient_calls == 3
    assert algorithm.estimated_gaps is algorithm.gaps
    np.testing.assert_array_equal(
        algorithm.num_stochastic_oracles, [1, 2, 3]
    )
    np.testing.assert_array_equal(algorithm.num_gradient_oracles, [1, 2, 3])
    np.testing.assert_array_equal(algorithm.num_oracles, [1, 2, 3])
    assert len(lmo.gradients) == 3
    assert algorithm.x.shape == (1,)
    assert algorithm.func_vals.shape == (3,)
    assert algorithm.gaps.shape == (3,)
    assert algorithm.ns_gaps.shape == (3,)


def test_first_sample_initializes_estimator_independently_of_rho_zero():
    objective = ScriptedStochasticObjective([[2.0], [-2.0]])
    lmo = RecordingLMO()
    algorithm = StochasticFrames(objective, lmo, identity_prox, "indicator")

    algorithm.run(
        np.array([0.0]),
        n_steps=2,
        rho_schedule=0.25,
        show_progress=False,
    )

    # d_0 = 2, independently of rho_0 = .25
    # d_1 = .75 * 2 + .25 * -2 = 1
    np.testing.assert_allclose(lmo.gradients, [[2.0], [1.0]])
    np.testing.assert_allclose(algorithm.gradient_estimate, [1.0])
    np.testing.assert_allclose(algorithm.momentum_weights, [0.25, 0.25])


def test_momentum_estimates_only_the_smooth_gradient():
    objective = ScriptedStochasticObjective([[2.0], [-2.0]])
    lmo = RecordingLMO()

    def constant_moreau_prox(y, beta):
        return np.asarray(y) - 10.0 * beta * np.ones_like(y)

    algorithm = StochasticFrames(
        objective,
        lmo,
        constant_moreau_prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]),
        n_steps=2,
        rho_schedule=0.25,
        show_progress=False,
    )

    # The exact Moreau gradient, 10, is added after the smooth estimates
    # 2 and 1 are formed. It must not itself enter the momentum state.
    np.testing.assert_allclose(lmo.gradients, [[12.0], [11.0]])
    np.testing.assert_allclose(algorithm.gradient_estimate, [1.0])


def test_lipschitz_gap_reuses_the_same_momentum_estimate():
    objective = ScriptedStochasticObjective(
        [[2.0], [-2.0]],
        minimal_norm_value=3.0,
    )
    lmo = RecordingLMO()
    algorithm = StochasticFrames(objective, lmo, identity_prox, "lipschitz")

    algorithm.run(
        np.array([0.0]),
        n_steps=2,
        rho_schedule=0.25,
        show_progress=False,
    )

    # Each iteration makes one algorithmic LMO call with d_k and one
    # diagnostic LMO call with d_k plus the minimal-norm selection.
    np.testing.assert_allclose(
        lmo.gradients,
        [[2.0], [5.0], [1.0], [4.0]],
    )
    assert objective.gradient_calls == 0
    assert objective.stochastic_gradient_calls == 2
    np.testing.assert_array_equal(algorithm.num_gradient_oracles, [1, 2])
    np.testing.assert_array_equal(algorithm.num_oracles, [1, 2])


def test_momentum_estimate_is_reset_between_runs():
    objective = ScriptedStochasticObjective([[4.0]])
    lmo = RecordingLMO()
    algorithm = StochasticFrames(objective, lmo, identity_prox, "indicator")

    algorithm.run(
        np.array([0.0]),
        n_steps=1,
        rho_schedule=0.25,
        show_progress=False,
    )
    np.testing.assert_allclose(algorithm.gradient_estimate, [4.0])

    objective.set_gradients([[0.0]])
    lmo.gradients.clear()
    algorithm.run(
        np.array([0.0]),
        n_steps=1,
        rho_schedule=0.25,
        show_progress=False,
    )

    np.testing.assert_allclose(lmo.gradients, [[0.0]])
    np.testing.assert_allclose(algorithm.gradient_estimate, [0.0])
    np.testing.assert_array_equal(algorithm.num_gradient_oracles, [1])
    np.testing.assert_array_equal(algorithm.num_oracles, [1])


def test_default_momentum_schedule_starts_with_a_full_sample():
    objective = ScriptedStochasticObjective([[1.0], [2.0], [3.0]])
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        identity_prox,
        "indicator",
    )

    algorithm.run(np.array([0.0]), n_steps=3, show_progress=False)

    expected = 4.0 / (np.arange(3) + 8) ** (2.0 / 3.0)
    np.testing.assert_allclose(algorithm.momentum_weights, expected)
    assert algorithm.momentum_weights[0] == 1.0


def test_default_smoothing_and_step_schedules_match_frames_formulas():
    objective = ScriptedStochasticObjective([[0.0], [0.0], [0.0]])
    prox = RecordingProx()
    directions = [np.array([1.0]), np.array([0.0]), np.array([1.0])]
    direction_iterator = iter(directions)
    algorithm = StochasticFrames(
        objective,
        lambda gradient: next(direction_iterator).copy(),
        prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]),
        beta0=2.0,
        n_steps=3,
        show_progress=False,
    )

    iterations = np.arange(3)
    expected_rho = 4.0 / (iterations + 8) ** (2.0 / 3.0)
    expected_beta = 2.0 / (iterations + 1) ** 0.25
    expected_step_size = 1.0 / (iterations + 1) ** 0.5
    np.testing.assert_allclose(algorithm.momentum_weights, expected_rho)
    np.testing.assert_allclose(algorithm.smoothing_parameters, expected_beta)
    np.testing.assert_allclose(prox.parameters, expected_beta)
    np.testing.assert_allclose(algorithm.step_sizes, expected_step_size)

    expected_x = np.array([0.0])
    for step_size, direction in zip(expected_step_size, directions):
        expected_x = (1.0 - step_size) * expected_x + step_size * direction
    np.testing.assert_allclose(algorithm.x, expected_x)


def test_scalar_schedules_are_constant_actual_values():
    objective = ScriptedStochasticObjective([[1.0], [1.0]])
    prox = RecordingProx()
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        prox,
        "indicator",
    )

    algorithm.run(
        np.array([4.0]),
        beta0=123.0,
        n_steps=2,
        rho_schedule=0.5,
        smoothing_schedule=0.25,
        step_size_schedule=0.25,
        show_progress=False,
    )

    np.testing.assert_allclose(algorithm.momentum_weights, [0.5, 0.5])
    np.testing.assert_allclose(algorithm.smoothing_parameters, [0.25, 0.25])
    np.testing.assert_allclose(prox.parameters, [0.25, 0.25])
    np.testing.assert_allclose(algorithm.step_sizes, [0.25, 0.25])
    np.testing.assert_allclose(algorithm.x, [2.25])


def test_callable_schedules_are_zero_based_and_evaluated_once():
    objective = ScriptedStochasticObjective([[2.0], [0.0], [5.0]])
    prox = RecordingProx()
    lmo = RecordingLMO()
    schedule_calls = {"rho": [], "smoothing": [], "step_size": []}

    def record_schedule(name, values):
        def schedule(iteration):
            schedule_calls[name].append(iteration)
            return values[iteration]

        return schedule

    rho_values = [1.0, 0.5, 0.25]
    smoothing_values = [0.8, 0.4, 0.2]
    step_size_values = [0.0, 0.25, 0.5]
    algorithm = StochasticFrames(objective, lmo, prox, "indicator")

    algorithm.run(
        np.array([8.0]),
        beta0=123.0,
        n_steps=3,
        rho_schedule=record_schedule("rho", rho_values),
        smoothing_schedule=record_schedule("smoothing", smoothing_values),
        step_size_schedule=record_schedule("step_size", step_size_values),
        show_progress=False,
    )

    assert schedule_calls == {
        "rho": [0, 1, 2],
        "smoothing": [0, 1, 2],
        "step_size": [0, 1, 2],
    }
    np.testing.assert_allclose(algorithm.momentum_weights, rho_values)
    np.testing.assert_allclose(
        algorithm.smoothing_parameters, smoothing_values
    )
    np.testing.assert_allclose(prox.parameters, smoothing_values)
    np.testing.assert_allclose(algorithm.step_sizes, step_size_values)
    np.testing.assert_allclose(lmo.gradients, [[2.0], [1.0], [2.0]])
    np.testing.assert_allclose(algorithm.x, [3.0])


def test_lmo_output_shape_is_validated():
    objective = ScriptedStochasticObjective([[1.0, 1.0]])
    algorithm = StochasticFrames(
        objective,
        lambda gradient: np.zeros((2, 1)),
        identity_prox,
        "indicator",
    )

    with pytest.raises(ValueError, match="LMO direction"):
        algorithm.run(
            np.zeros(2),
            n_steps=1,
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("schedule_name", "value", "message"),
    [
        ("rho_schedule", 0.0, r"finite value in \(0, 1\]"),
        ("rho_schedule", -0.1, r"finite value in \(0, 1\]"),
        ("rho_schedule", 1.1, r"finite value in \(0, 1\]"),
        ("rho_schedule", np.nan, r"finite value in \(0, 1\]"),
        ("rho_schedule", np.inf, r"finite value in \(0, 1\]"),
        ("smoothing_schedule", 0.0, "positive finite value"),
        ("smoothing_schedule", -0.1, "positive finite value"),
        ("smoothing_schedule", np.nan, "positive finite value"),
        ("smoothing_schedule", np.inf, "positive finite value"),
        ("step_size_schedule", -0.1, r"finite value in \[0, 1\]"),
        ("step_size_schedule", 1.1, r"finite value in \[0, 1\]"),
        ("step_size_schedule", np.nan, r"finite value in \[0, 1\]"),
        ("step_size_schedule", np.inf, r"finite value in \[0, 1\]"),
    ],
)
def test_invalid_scalar_schedule_value_is_rejected_before_oracles(
    schedule_name,
    value,
    message,
):
    objective = ScriptedStochasticObjective([[1.0]])
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        identity_prox,
        "indicator",
    )
    schedules = {
        "rho_schedule": 1.0,
        "smoothing_schedule": 1.0,
        "step_size_schedule": 1.0,
    }
    schedules[schedule_name] = value

    with pytest.raises(ValueError, match=message) as error:
        algorithm.run(
            np.array([0.0]),
            n_steps=1,
            show_progress=False,
            **schedules,
        )

    assert schedule_name in str(error.value)
    assert "iteration 0" in str(error.value)
    assert objective.stochastic_gradient_calls == 0


@pytest.mark.parametrize(
    ("schedule_name", "valid_value", "invalid_value"),
    [
        ("rho_schedule", 1.0, 0.0),
        ("smoothing_schedule", 1.0, 0.0),
        ("step_size_schedule", 0.5, 1.1),
    ],
)
def test_invalid_callable_schedule_stops_before_failing_iteration_oracles(
    schedule_name,
    valid_value,
    invalid_value,
):
    objective = ScriptedStochasticObjective([[1.0], [1.0]])
    prox = RecordingProx()
    lmo = RecordingLMO()
    algorithm = StochasticFrames(
        objective,
        lmo,
        prox,
        "indicator",
    )
    schedules = {
        "rho_schedule": 1.0,
        "smoothing_schedule": 1.0,
        "step_size_schedule": 1.0,
    }
    schedules[schedule_name] = (
        lambda iteration: valid_value if iteration == 0 else invalid_value
    )

    with pytest.raises(ValueError, match="iteration 1") as error:
        algorithm.run(
            np.array([0.0]),
            n_steps=2,
            show_progress=False,
            **schedules,
        )

    assert schedule_name in str(error.value)
    assert objective.stochastic_gradient_calls == 1
    assert len(prox.parameters) == 1
    assert len(lmo.gradients) == 1


@pytest.mark.parametrize(
    "schedule_name",
    ["rho_schedule", "smoothing_schedule", "step_size_schedule"],
)
def test_schedule_argument_must_be_scalar_callable_or_none(schedule_name):
    objective = ScriptedStochasticObjective([[1.0]])
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        identity_prox,
        "indicator",
    )

    with pytest.raises(TypeError, match=schedule_name):
        algorithm.run(
            np.array([0.0]),
            n_steps=1,
            show_progress=False,
            **{schedule_name: [0.5]},
        )

    assert objective.stochastic_gradient_calls == 0


@pytest.mark.parametrize(
    "schedule_name",
    ["rho_schedule", "smoothing_schedule", "step_size_schedule"],
)
@pytest.mark.parametrize(
    "return_value",
    [np.array([0.5]), 0.5 + 0.1j, np.complex128(0.5 + 0.1j), "0.5"],
)
def test_callable_schedule_must_return_a_real_scalar(
    schedule_name,
    return_value,
):
    objective = ScriptedStochasticObjective([[1.0]])
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        identity_prox,
        "indicator",
    )

    with pytest.raises(TypeError, match=rf"{schedule_name}.*iteration 0"):
        algorithm.run(
            np.array([0.0]),
            n_steps=1,
            show_progress=False,
            **{schedule_name: lambda iteration: return_value},
        )

    assert objective.stochastic_gradient_calls == 0


@pytest.mark.parametrize(
    "schedule_value", [0.5 + 0.1j, np.complex128(0.5 + 0.1j), "0.5"]
)
@pytest.mark.parametrize(
    "schedule_name",
    ["rho_schedule", "smoothing_schedule", "step_size_schedule"],
)
def test_scalar_schedule_argument_must_be_real_numeric(
    schedule_name,
    schedule_value,
):
    objective = ScriptedStochasticObjective([[1.0]])
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        identity_prox,
        "indicator",
    )

    with pytest.raises(TypeError, match=rf"{schedule_name}.*iteration 0"):
        algorithm.run(
            np.array([0.0]),
            n_steps=1,
            show_progress=False,
            **{schedule_name: schedule_value},
        )

    assert objective.stochastic_gradient_calls == 0


@pytest.mark.parametrize("beta0", [0.0, -0.1, np.nan, np.inf])
def test_beta0_is_validated_even_with_a_custom_smoothing_schedule(beta0):
    objective = ScriptedStochasticObjective([[1.0]])
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        identity_prox,
        "indicator",
    )

    with pytest.raises(ValueError, match="beta0"):
        algorithm.run(
            np.array([0.0]),
            beta0=beta0,
            n_steps=1,
            smoothing_schedule=0.5,
            show_progress=False,
        )

    assert objective.stochastic_gradient_calls == 0


def test_objective_evaluation_can_be_disabled_for_expensive_stochastic_runs():
    objective = ScriptedStochasticObjective([[1.0], [1.0]])

    def fail_if_evaluated(x):
        raise AssertionError("the full objective should not be evaluated")

    objective.evaluate = fail_if_evaluated
    algorithm = StochasticFrames(
        objective,
        RecordingLMO(),
        identity_prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]),
        n_steps=2,
        evaluate_objective=False,
        show_progress=False,
    )

    assert np.isnan(algorithm.func_vals).all()


def test_iterate_callback_receives_initial_and_post_update_copies():
    objective = ScriptedStochasticObjective([[0.0], [0.0]])
    directions = iter([np.array([2.0]), np.array([4.0])])
    seen = []

    def callback(completed_steps, x):
        seen.append((completed_steps, x.copy()))
        x[...] = -999.0

    algorithm = StochasticFrames(
        objective,
        lambda gradient: next(directions).copy(),
        identity_prox,
        "indicator",
    )
    algorithm.run(
        np.array([0.0]),
        n_steps=2,
        step_size_schedule=0.5,
        iterate_callback=callback,
        show_progress=False,
    )

    assert [step for step, _ in seen] == [0, 1, 2]
    np.testing.assert_allclose([x for _, x in seen], [[0.0], [1.0], [2.5]])
    np.testing.assert_allclose(algorithm.x, [2.5])


def test_iterate_callback_frequency_includes_a_nonmultiple_final_step():
    objective = ScriptedStochasticObjective([[0.0]] * 5)
    seen_steps = []
    algorithm = StochasticFrames(
        objective,
        lambda gradient: np.zeros_like(gradient),
        identity_prox,
        "indicator",
    )

    algorithm.run(
        np.array([0.0]),
        n_steps=5,
        iterate_callback=lambda step, x: seen_steps.append(step),
        iterate_callback_frequency=2,
        show_progress=False,
    )

    assert seen_steps == [0, 2, 4, 5]


def test_complex_gap_uses_the_real_frobenius_inner_product():
    class ComplexObjective(ObjectiveFunction):
        def evaluate(self, x):
            return float(np.vdot(x, x).real)

        def stochastic_gradient(self, x):
            return np.array([1.0 + 2.0j])

        def linear_operator(self, x):
            return x

        def linear_operator_adjoint(self, x):
            return x

    algorithm = StochasticFrames(
        ComplexObjective(),
        lambda gradient: np.array([-1.0j]),
        identity_prox,
        "indicator",
    )
    algorithm.run(
        np.array([1.0j]),
        n_steps=1,
        rho_schedule=1.0,
        step_size_schedule=0.0,
        show_progress=False,
    )

    expected = np.vdot(1.0 + 2.0j, 2.0j).real
    np.testing.assert_allclose(algorithm.gaps, [expected])
