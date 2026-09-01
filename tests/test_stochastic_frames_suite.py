import importlib

import numpy as np
import pytest

from paper.experiments.stochastic_frames_suite.components import (
    BoxConstraint,
    DenseLinearMap,
    FiniteDifferenceMap,
    FrobeniusBallConstraint,
    IdentityMap,
    L1Penalty,
    L2BallConstraint,
    MaxPenalty,
    NonnegativeIndicator,
    NuclearNormBallConstraint,
    QuadraticLiftMap,
    SimplexConstraint,
    UnitDiagonalIndicator,
    ZeroPenalty,
)
from paper.experiments.stochastic_frames_suite.config import ExperimentConfig
from paper.experiments.stochastic_frames_suite.io import load_result, save_result
from paper.experiments.stochastic_frames_suite.problem import SamplePlan
from paper.experiments.stochastic_frames_suite.registry import (
    create_problem,
    problem_names,
)
from paper.experiments.stochastic_frames_suite.results import aggregate_results
from paper.experiments.stochastic_frames_suite.runner import (
    registry_configs,
    run_experiment,
    run_many,
)


EXPECTED_PROBLEMS = (
    "simplex_regression",
    "sparse_logistic",
    "tv_denoising",
    "robust_portfolio",
    "matrix_completion",
    "phase_retrieval",
    "factorized_correlation",
)


@pytest.fixture(scope="module")
def tiny_registry_results():
    configs = registry_configs(
        problems=("all",),
        methods=("momentum",),
        seeds=(0,),
        profile="tiny",
        steps=2,
        batch_size=2,
        checkpoint_steps=(0, 2),
    )
    return run_many(configs)


@pytest.fixture(scope="module")
def simplex_seed_results():
    configs = registry_configs(
        problems=("simplex_regression",),
        methods=("momentum",),
        seeds=(0, 1),
        profile="tiny",
        steps=2,
        batch_size=2,
        checkpoint_steps=(0, 2),
    )
    return run_many(configs)


@pytest.mark.parametrize(
    ("linear_map", "x", "y"),
    (
        (
            IdentityMap(),
            np.array([0.2, -1.0, 3.0]),
            np.array([-2.0, 0.5, 1.5]),
        ),
        (
            DenseLinearMap(
                np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
            ),
            np.array([0.3, -0.7, 1.2]),
            np.array([1.5, -0.4]),
        ),
        (
            FiniteDifferenceMap(size=5),
            np.array([0.1, -0.3, 0.8, 1.1, -0.2]),
            np.array([0.4, -1.0, 0.2, 2.0]),
        ),
    ),
    ids=("identity", "dense", "finite_difference"),
)
def test_linear_map_adjoint_identity(linear_map, x, y):
    lhs = np.vdot(linear_map.forward(x), y)
    rhs = np.vdot(x, linear_map.jacobian_adjoint(x, y))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-13)


def test_quadratic_lift_derivative_and_adjoint_identity():
    rng = np.random.default_rng(12)
    lift = QuadraticLiftMap()
    x = rng.normal(size=(5, 2))
    direction = rng.normal(size=x.shape)
    dual = rng.normal(size=(5, 5))
    epsilon = 1e-6

    finite_difference = (
        lift.forward(x + epsilon * direction)
        - lift.forward(x - epsilon * direction)
    ) / (2.0 * epsilon)
    derivative = lift.directional_derivative(x, direction)
    np.testing.assert_allclose(
        finite_difference, derivative, rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(
        np.vdot(derivative, dual),
        np.vdot(direction, lift.jacobian_adjoint(x, dual)),
        rtol=1e-13,
        atol=1e-13,
    )


def test_penalty_prox_formulas():
    value = np.array([-2.0, -0.25, 0.5, 3.0])
    np.testing.assert_array_equal(ZeroPenalty().prox(value, 0.7), value)
    np.testing.assert_allclose(
        L1Penalty(weight=2.0).prox(value, 0.25),
        [-1.5, 0.0, 0.0, 2.5],
    )
    np.testing.assert_allclose(
        MaxPenalty(weight=2.0).prox(np.array([3.0, 1.0, -1.0]), 0.5),
        [2.0, 1.0, -1.0],
    )
    np.testing.assert_allclose(
        NonnegativeIndicator().prox(value, 1.0), [0.0, 0.0, 0.5, 3.0]
    )

    matrix = np.array([[2.0, -3.0], [4.0, 5.0]])
    np.testing.assert_allclose(
        UnitDiagonalIndicator().prox(matrix, 0.1),
        [[1.0, -3.0], [4.0, 1.0]],
    )


@pytest.mark.parametrize(
    ("constraint", "gradient", "optimal_value"),
    (
        (
            SimplexConstraint(radius=2.0, dimension=3),
            np.array([2.0, -1.0, 0.5]),
            -2.0,
        ),
        (
            L2BallConstraint(radius=1.5),
            np.array([3.0, 4.0]),
            -7.5,
        ),
        (
            FrobeniusBallConstraint(radius=2.0),
            np.array([[3.0, 0.0], [0.0, 4.0]]),
            -10.0,
        ),
        (
            BoxConstraint(lower=-1.0, upper=2.0),
            np.array([3.0, -4.0, 0.0]),
            -11.0,
        ),
        (
            NuclearNormBallConstraint(radius=1.25),
            np.array([[3.0, 0.0], [0.0, -2.0]]),
            -3.75,
        ),
    ),
    ids=("simplex", "l2", "frobenius", "box", "nuclear"),
)
def test_lmo_is_feasible_and_optimal(constraint, gradient, optimal_value):
    atom = constraint.lmo(gradient)
    assert constraint.contains(atom)
    np.testing.assert_allclose(
        np.vdot(gradient, atom).real,
        optimal_value,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("constraint", "shape", "expected"),
    (
        (
            SimplexConstraint(radius=1.0, dimension=4),
            (4,),
            np.full(4, 0.25),
        ),
        (L2BallConstraint(radius=2.0), (3,), np.zeros(3)),
        (
            FrobeniusBallConstraint(radius=2.0),
            (2, 3),
            np.zeros((2, 3)),
        ),
        (
            BoxConstraint(lower=-1.0, upper=2.0),
            (3,),
            np.zeros(3),
        ),
        (
            NuclearNormBallConstraint(radius=2.0),
            (2, 3),
            np.zeros((2, 3)),
        ),
    ),
    ids=("simplex", "l2", "frobenius", "box", "nuclear"),
)
def test_lmo_zero_gradient_behavior(constraint, shape, expected):
    atom = constraint.lmo(np.zeros(shape))
    assert constraint.contains(atom)
    np.testing.assert_array_equal(atom, expected)


@pytest.mark.parametrize("problem_name", EXPECTED_PROBLEMS)
def test_registry_problem_full_batch_gradient_is_exact_unbiased_mean(
    problem_name,
):
    problem = create_problem(
        problem_name, profile="tiny", problem_seed=3, initialization_seed=4
    )
    x = problem.x0
    population = np.arange(problem.population_size, dtype=np.int64)
    exact = problem.smooth.gradient(x)
    full_batch = problem.smooth.gradient_for_batch(x, population)
    singleton_mean = np.mean(
        np.stack(
            [
                problem.smooth.gradient_for_batch(
                    x, np.array([index], dtype=np.int64)
                )
                for index in population
            ]
        ),
        axis=0,
    )

    assert exact.shape == x.shape
    assert np.all(np.isfinite(exact))
    np.testing.assert_allclose(full_batch, exact, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(singleton_mean, exact, rtol=2e-13, atol=2e-13)


def test_sample_plan_same_seed_repeats_and_different_seed_changes():
    first = SamplePlan.generate(23, steps=8, batch_size=4, seed=91)
    repeated = SamplePlan.generate(23, steps=8, batch_size=4, seed=91)
    changed = SamplePlan.generate(23, steps=8, batch_size=4, seed=92)

    np.testing.assert_array_equal(first.indices, repeated.indices)
    assert not np.array_equal(first.indices, changed.indices)
    returned_batch = first.batch(0)
    returned_batch[0] = (returned_batch[0] + 1) % first.population_size
    assert not np.array_equal(returned_batch, first.batch(0))
    with pytest.raises(ValueError):
        first.indices[0, 0] = 0


def test_same_seed_run_is_reproducible_except_wall_clock():
    config = ExperimentConfig(
        problem="simplex_regression",
        profile="tiny",
        steps=3,
        batch_size=2,
        problem_seed=5,
        initialization_seed=6,
        sampling_seed=7,
        checkpoint_steps=(0, 3),
    )
    first = run_experiment(config)
    repeated = run_experiment(config)

    assert first.metadata == repeated.metadata
    for name in (
        "checkpoint_steps",
        "checkpoint_iterates",
        "momentum_weights",
        "smoothing_parameters",
        "step_sizes",
        "estimated_gaps",
        "task_loss",
        "composite_objective",
        "smoothed_objective",
        "exact_smoothed_gap",
        "reference_objective",
        "reference_gap",
        "estimator_error",
        "feasibility_or_regularizer",
        "final_iterate",
        "final_gradient_estimate",
    ):
        np.testing.assert_allclose(
            getattr(first, name), getattr(repeated, name), equal_nan=True
        )
    assert first.oracle_counts == repeated.oracle_counts
    assert first.problem_metrics.keys() == repeated.problem_metrics.keys()
    for name in first.problem_metrics:
        np.testing.assert_allclose(
            first.problem_metrics[name], repeated.problem_metrics[name]
        )


def test_run_many_reuses_sampling_and_initialization_fairly_across_methods():
    methods = ("momentum", "no-momentum", "deterministic", "fixed-smoothing")
    configs = [
        ExperimentConfig(
            problem="simplex_regression",
            method=method,
            profile="tiny",
            steps=3,
            batch_size=2,
            problem_seed=11,
            initialization_seed=12,
            sampling_seed=13,
            beta0=0.6,
            checkpoint_steps=(0, 3),
        )
        for method in methods
    ]
    results = run_many(configs)

    digests = {result.metadata.extra["sample_plan_sha256"] for result in results}
    assert len(digests) == 1
    for result in results[1:]:
        np.testing.assert_array_equal(
            result.checkpoint_iterates[0], results[0].checkpoint_iterates[0]
        )
    by_method = {result.method: result for result in results}
    np.testing.assert_array_equal(
        by_method["no-momentum"].momentum_weights, np.ones(3)
    )
    np.testing.assert_array_equal(
        by_method["deterministic"].momentum_weights, np.ones(3)
    )
    np.testing.assert_allclose(
        by_method["fixed-smoothing"].smoothing_parameters, 0.6
    )
    np.testing.assert_array_equal(
        by_method["momentum"].oracle_counts.sampled_observations,
        [2, 6],
    )
    population_size = create_problem("simplex_regression").population_size
    np.testing.assert_array_equal(
        by_method["deterministic"].oracle_counts.sampled_observations,
        population_size * np.asarray([1, 3]),
    )


def test_explicit_sample_plan_seed_is_the_recorded_sampling_seed():
    problem = create_problem("simplex_regression")
    plan = SamplePlan.generate(
        problem.population_size, steps=2, batch_size=2, seed=19
    )
    result = run_experiment(
        ExperimentConfig(
            problem="simplex_regression",
            steps=2,
            batch_size=2,
            sampling_seed=7,
            checkpoint_steps=(0, 2),
        ),
        problem=problem,
        sample_plan=plan,
    )

    assert result.sampling_seed == 19
    assert result.metadata.extra["configured_sampling_seed"] == 7


def test_preupdate_oracle_counts_align_estimator_checkpoint_work():
    result = run_experiment(
        ExperimentConfig(
            problem="simplex_regression",
            steps=2,
            batch_size=2,
            checkpoint_steps=(0, 2),
        )
    )
    aligned = result.oracle_counts_at_checkpoints()

    np.testing.assert_array_equal(aligned.stochastic_gradients, [1, 2])
    np.testing.assert_array_equal(aligned.sampled_observations, [2, 4])


def test_registry_tiny_runs_are_finite_feasible_and_counted(
    tiny_registry_results,
):
    assert problem_names() == EXPECTED_PROBLEMS
    assert tuple(result.problem_name for result in tiny_registry_results) == (
        EXPECTED_PROBLEMS
    )

    for result in tiny_registry_results:
        problem = create_problem(result.problem_name, profile="tiny")
        assert problem.constraint.contains(result.final_iterate)
        assert all(
            problem.constraint.contains(iterate)
            for iterate in result.checkpoint_iterates
        )
        np.testing.assert_array_equal(
            result.checkpoint_iterates[0], problem.x0
        )
        for name in (
            "momentum_weights",
            "smoothing_parameters",
            "step_sizes",
            "estimated_gaps",
            "task_loss",
            "smoothed_objective",
            "exact_smoothed_gap",
            "reference_objective",
            "reference_gap",
            "estimator_error",
            "feasibility_or_regularizer",
            "optimizer_time",
            "final_iterate",
            "final_gradient_estimate",
        ):
            assert np.all(np.isfinite(getattr(result, name))), (
                result.problem_name,
                name,
            )
        assert not np.any(np.isnan(result.composite_objective))
        for values in result.problem_metrics.values():
            assert np.all(np.isfinite(values))

        np.testing.assert_array_equal(
            result.oracle_counts.stochastic_gradients, [1, 2]
        )
        np.testing.assert_array_equal(
            result.oracle_counts.sampled_observations, [2, 4]
        )
        for values in result.oracle_counts.to_dict().values():
            assert values.dtype.kind in "iu"
            assert np.all(values >= 0)
            assert np.all(np.diff(values) >= 0)
        calls_per_step = 1 if problem.penalty.kind == "indicator" else 2
        map_calls_per_step = 2 if problem.penalty.kind == "indicator" else 3
        assert result.oracle_counts.lmo_calls[-1] == 2 * calls_per_step
        assert result.oracle_counts.prox_calls[-1] == 2
        assert result.oracle_counts.map_calls[-1] == 2 * map_calls_per_step


def test_result_round_trip_is_pickle_free(tmp_path, tiny_registry_results):
    original = tiny_registry_results[0]
    path = save_result(tmp_path / "result.npz", original)

    with np.load(path, allow_pickle=False) as archive:
        assert archive.files
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    loaded = load_result(
        path,
        expected_metadata={
            "problem_name": original.problem_name,
            "sample_plan_sha256": original.metadata.extra[
                "sample_plan_sha256"
            ],
        },
    )
    assert loaded == original


def test_seed_aggregation_reports_pointwise_median_and_quantiles(
    simplex_seed_results,
):
    aggregate = aggregate_results(
        simplex_seed_results,
        quantiles=(0.25, 0.75),
        metric_names=("task_loss", "test_mse"),
    )

    assert aggregate.n_runs == 2
    np.testing.assert_array_equal(aggregate.checkpoint_steps, [0, 2])
    np.testing.assert_allclose(aggregate.quantile_levels, [0.25, 0.75])
    task_values = np.stack(
        [result.task_loss for result in simplex_seed_results], axis=0
    )
    np.testing.assert_allclose(
        aggregate.metric("task_loss").median,
        np.median(task_values, axis=0),
    )
    np.testing.assert_allclose(
        aggregate.metric("task_loss").quantile_values,
        np.quantile(task_values, [0.25, 0.75], axis=0),
    )


def test_seed_aggregation_rejects_different_realized_schedules():
    base = dict(
        problem="simplex_regression",
        steps=2,
        batch_size=2,
        checkpoint_steps=(0, 2),
    )
    first = run_experiment(ExperimentConfig(**base, rho_scale=1.0))
    second = run_experiment(ExperimentConfig(**base, rho_scale=0.5))

    with pytest.raises(ValueError, match="momentum_weights"):
        aggregate_results([first, second])


def test_fixed_smoothing_rejects_a_varying_schedule_override():
    with pytest.raises(ValueError, match="fixed-smoothing"):
        ExperimentConfig(
            problem="simplex_regression",
            method="fixed-smoothing",
            smoothing_schedule=lambda iteration: 1.0 / (iteration + 1),
        )


def test_plotting_smoke(tmp_path, tiny_registry_results):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plotting = importlib.import_module(
        "paper.experiments.stochastic_frames_suite.plotting"
    )
    path = plotting.save_common_plot(
        tmp_path / "common.png",
        [tiny_registry_results[0]],
        metric="exact_smoothed_gap",
        dpi=50,
    )
    assert path.is_file()
    assert path.stat().st_size > 0
    per_step_path = plotting.save_common_plot(
        tmp_path / "estimated.png",
        [tiny_registry_results[0]],
        metric="estimated_gaps",
        dpi=50,
    )
    assert per_step_path.is_file()
    assert per_step_path.stat().st_size > 0


def test_plotting_disambiguates_equal_step_and_checkpoint_lengths():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plotting = importlib.import_module(
        "paper.experiments.stochastic_frames_suite.plotting"
    )
    result = run_experiment(
        ExperimentConfig(
            problem="simplex_regression",
            steps=4,
            batch_size=2,
            checkpoint_steps=(0, 2, 3, 4),
        )
    )

    checkpoint_axis = plotting.plot_metric(
        result, "exact_smoothed_gap", x_axis="optimizer_time"
    )
    np.testing.assert_allclose(
        checkpoint_axis.lines[0].get_ydata(), result.exact_smoothed_gap
    )
    per_step_axis = plotting.plot_metric(
        result, "estimated_gaps", x_axis="sampled_observations"
    )
    np.testing.assert_allclose(
        per_step_axis.lines[0].get_ydata(),
        result.estimated_gaps[[0, 2, 3, 3]],
    )
    import matplotlib.pyplot as plt

    plt.close(checkpoint_axis.figure)
    plt.close(per_step_axis.figure)


def test_cli_smoke_writes_loadable_archive(tmp_path):
    pytest.importorskip("matplotlib")
    cli = importlib.import_module("paper.experiments.stochastic_frames_suite.cli")
    output = tmp_path / "cli"
    status = cli.main(
        [
            "--problem",
            "simplex_regression",
            "--methods",
            "momentum",
            "--seeds",
            "0",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--output-dir",
            str(output),
            "--no-plots",
        ]
    )

    assert status == 0
    archives = list(output.glob("*.npz"))
    assert len(archives) == 1
    assert load_result(archives[0]).problem_name == "simplex_regression"
    assert not list(output.glob("*.png"))
