"""Execution engine for reproducible stochastic-FRAMES comparisons."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter
from typing import Iterable, List, Optional

import numpy as np

from frank_wolfe import StochasticFrames

from .config import ExperimentConfig
from .problem import ObjectiveAdapter, SamplePlan, moreau_value
from .registry import canonical_problem_name, create_problem, problem_names
from .results import OracleCounts, RunMetadata, RunResult


def _sample_plan_digest(plan):
    digest = sha256()
    digest.update(np.asarray(plan.indices, dtype=np.int64).tobytes())
    digest.update(str(plan.population_size).encode("ascii"))
    digest.update(str(plan.batch_size).encode("ascii"))
    return digest.hexdigest()


def _moreau_gradient(problem, x, beta):
    mapped = np.asarray(problem.composite_map.forward(x))
    prox = np.asarray(problem.penalty.prox(mapped, beta))
    residual = mapped - prox
    gradient = np.asarray(
        problem.composite_map.jacobian_adjoint(x, residual)
    ) / beta
    return mapped, gradient


def _gap(constraint, x, gradient):
    atom = np.asarray(constraint.lmo(gradient))
    value = float(np.vdot(gradient, np.asarray(x) - atom).real)
    # Tiny negative values can result from SVD/inner-product roundoff.
    return max(0.0, value)


def _checkpoint_beta(step, n_steps, realized, reference_beta):
    if n_steps == 0:
        return float(reference_beta)
    return float(realized[min(int(step), n_steps - 1)])


def _reconstruct_estimates(samples, momentum_weights, shape):
    if not samples:
        return np.empty((0,) + tuple(shape), dtype=float)
    estimates = []
    current = np.asarray(samples[0]).copy()
    estimates.append(current.copy())
    for index in range(1, len(samples)):
        rho = momentum_weights[index]
        current = (1.0 - rho) * current + rho * np.asarray(samples[index])
        estimates.append(current.copy())
    return np.stack(estimates, axis=0)


def run_experiment(
    config: ExperimentConfig,
    *,
    problem=None,
    sample_plan: Optional[SamplePlan] = None,
) -> RunResult:
    """Run one configured benchmark and return a validated result.

    Passing an explicit ``sample_plan`` is useful for Python-side ablations.
    :func:`run_many` supplies the same plan object to matching method runs.
    """

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    canonical = canonical_problem_name(config.problem)
    if problem is None:
        problem = create_problem(
            canonical,
            profile=config.profile,
            problem_seed=config.problem_seed,
            initialization_seed=config.initialization_seed,
        )
    if problem.name != canonical:
        raise ValueError(
            f"Problem object is named {problem.name!r}, expected {canonical!r}."
        )
    if not np.all(np.isfinite(problem.x0)):
        raise ValueError("The initial iterate must contain finite values.")
    if not problem.constraint.contains(problem.x0):
        raise ValueError("The initial iterate must be constraint-feasible.")

    if sample_plan is None:
        sample_plan = SamplePlan.generate(
            problem.population_size,
            config.steps,
            config.batch_size,
            config.sampling_seed,
        )
    if sample_plan.steps != config.steps:
        raise ValueError("SamplePlan step count does not match the config.")
    if sample_plan.batch_size != config.batch_size:
        raise ValueError("SamplePlan batch size does not match the config.")

    deterministic = config.method == "deterministic"
    adapter = ObjectiveAdapter(
        problem, sample_plan, deterministic=deterministic
    )
    algorithm = StochasticFrames(
        adapter,
        adapter.lmo,
        adapter.prox,
        problem.penalty.kind,
    )
    rho, smoothing, step_size = config.resolved_schedules()
    checkpoints = config.resolved_checkpoint_steps()
    time_by_step = {0: 0.0}
    counts_by_step = {0: adapter.counts()}
    start_time = None

    def recorder(completed_steps, _iterate):
        counts_by_step[completed_steps] = adapter.counts()
        if completed_steps == 0:
            time_by_step[0] = 0.0
        else:
            time_by_step[completed_steps] = perf_counter() - start_time

    start_time = perf_counter()
    algorithm.run(
        problem.x0,
        beta0=config.beta0,
        n_steps=config.steps,
        show_progress=config.show_progress,
        rho_schedule=rho,
        smoothing_schedule=smoothing,
        step_size_schedule=step_size,
        evaluate_objective=False,
        iterate_callback=recorder,
        iterate_callback_frequency=1,
    )
    optimizer_total_time = perf_counter() - start_time
    time_by_step[config.steps] = optimizer_total_time
    counts_by_step[config.steps] = adapter.counts()

    if config.steps:
        pre_update_points = adapter.stochastic_points
        checkpoint_iterates = np.stack(
            [
                pre_update_points[int(checkpoint)]
                if checkpoint < config.steps
                else np.asarray(algorithm.x).copy()
                for checkpoint in checkpoints
            ],
            axis=0,
        )
    else:
        checkpoint_iterates = np.expand_dims(
            np.asarray(algorithm.x).copy(), axis=0
        )
    if not all(
        problem.constraint.contains(iterate) for iterate in checkpoint_iterates
    ):
        raise RuntimeError("The optimizer produced an infeasible checkpoint.")

    estimates = _reconstruct_estimates(
        adapter.stochastic_gradients,
        algorithm.momentum_weights,
        problem.x0.shape,
    )
    if config.steps:
        checkpoint_estimates = [
            estimates[min(int(checkpoint), config.steps - 1)]
            for checkpoint in checkpoints
        ]
    else:
        checkpoint_estimates = [np.zeros_like(problem.x0, dtype=float)]

    task_loss = []
    composite_objective = []
    smoothed_objective = []
    exact_smoothed_gap = []
    reference_objective = []
    reference_gap = []
    estimator_error = []
    feasibility_or_regularizer = []
    metric_rows = []

    for checkpoint, x, estimate in zip(
        checkpoints, checkpoint_iterates, checkpoint_estimates
    ):
        beta = _checkpoint_beta(
            checkpoint,
            config.steps,
            algorithm.smoothing_parameters,
            config.reference_beta,
        )
        smooth_value = float(problem.smooth.value(x))
        exact_gradient = np.asarray(problem.smooth.gradient(x))
        mapped, moreau_gradient = _moreau_gradient(problem, x, beta)
        original_penalty = float(problem.penalty.value(mapped))
        active_moreau = moreau_value(problem.penalty, mapped, beta)
        reference_moreau = moreau_value(
            problem.penalty, mapped, config.reference_beta
        )
        _, reference_moreau_gradient = _moreau_gradient(
            problem, x, config.reference_beta
        )

        task_loss.append(float(problem.task_loss(x)))
        composite_objective.append(smooth_value + original_penalty)
        smoothed_objective.append(smooth_value + active_moreau)
        exact_smoothed_gap.append(
            _gap(problem.constraint, x, exact_gradient + moreau_gradient)
        )
        reference_objective.append(smooth_value + reference_moreau)
        reference_gap.append(
            _gap(
                problem.constraint,
                x,
                exact_gradient + reference_moreau_gradient,
            )
        )
        error = np.asarray(estimate) - exact_gradient
        estimator_error.append(float(np.vdot(error, error).real))
        feasibility_or_regularizer.append(
            float(problem.feasibility_or_regularizer(x))
        )
        metric_rows.append(dict(problem.metrics(x)))

    metric_keys = set(metric_rows[0]) if metric_rows else set()
    if any(set(row) != metric_keys for row in metric_rows):
        raise ValueError("Problem metric names changed between checkpoints.")
    problem_metrics = {
        key: np.asarray([float(row[key]) for row in metric_rows], dtype=float)
        for key in sorted(metric_keys)
    }

    # Checkpoint k < T stores the pre-update triple (x_k, d_k, beta_k).
    # Forming that direction consumes iteration k's work.  The explicit
    # terminal checkpoint carries the last estimator and beta without another
    # stochastic-oracle call.
    work_steps = np.asarray(
        [
            min(int(checkpoint) + 1, config.steps)
            if checkpoint < config.steps
            else config.steps
            for checkpoint in checkpoints
        ],
        dtype=np.int64,
    )
    count_rows = [counts_by_step[int(step)] for step in work_steps]
    metric_counts = np.arange(1, len(checkpoints) + 1, dtype=np.int64)
    oracle_counts = OracleCounts(
        stochastic_gradients=np.asarray(
            [row["stochastic_gradient"] for row in count_rows], dtype=np.int64
        ),
        sampled_observations=np.asarray(
            [row["sampled_observation"] for row in count_rows], dtype=np.int64
        ),
        lmo_calls=np.asarray(
            [row["lmo"] for row in count_rows], dtype=np.int64
        ),
        prox_calls=np.asarray(
            [row["prox"] for row in count_rows], dtype=np.int64
        ),
        map_calls=np.asarray(
            [row["map_forward"] + row["map_adjoint"] for row in count_rows],
            dtype=np.int64,
        ),
        metric_calls=metric_counts,
    )
    optimizer_time = np.asarray(
        [time_by_step[int(step)] for step in work_steps],
        dtype=float,
    )

    constraint_metadata = getattr(problem.constraint, "metadata", {})
    metadata = RunMetadata(
        problem_name=problem.name,
        n_steps=config.steps,
        problem_seed=config.problem_seed,
        initialization_seed=config.initialization_seed,
        sampling_seed=sample_plan.seed,
        batch_size=config.batch_size,
        method=config.method,
        reference_beta=config.reference_beta,
        extra={
            "profile": config.profile,
            "problem_schema_version": problem.schema_version,
            "problem": problem.metadata,
            "constraint": constraint_metadata,
            "sample_plan_sha256": _sample_plan_digest(sample_plan),
            "configured_sampling_seed": config.sampling_seed,
            "actual_batch_size": (
                problem.population_size if deterministic else config.batch_size
            ),
            "oracle_count_axis": "checkpoint_pre_update",
        },
    )

    return RunResult(
        metadata=metadata,
        checkpoint_steps=checkpoints,
        checkpoint_iterates=checkpoint_iterates,
        momentum_weights=algorithm.momentum_weights,
        smoothing_parameters=algorithm.smoothing_parameters,
        step_sizes=algorithm.step_sizes,
        estimated_gaps=algorithm.estimated_gaps,
        task_loss=np.asarray(task_loss),
        composite_objective=np.asarray(composite_objective),
        smoothed_objective=np.asarray(smoothed_objective),
        exact_smoothed_gap=np.asarray(exact_smoothed_gap),
        reference_objective=np.asarray(reference_objective),
        reference_gap=np.asarray(reference_gap),
        estimator_error=np.asarray(estimator_error),
        feasibility_or_regularizer=np.asarray(feasibility_or_regularizer),
        oracle_counts=oracle_counts,
        optimizer_time=optimizer_time,
        problem_metrics=problem_metrics,
        final_iterate=np.asarray(algorithm.x).copy(),
        final_gradient_estimate=np.asarray(algorithm.gradient_estimate).copy(),
    )


def run_many(configs: Iterable[ExperimentConfig]) -> List[RunResult]:
    """Run configs while reusing identical sample plans across methods."""

    configs = list(configs)
    plan_cache = {}
    results = []
    for config in configs:
        problem = create_problem(
            config.problem,
            profile=config.profile,
            problem_seed=config.problem_seed,
            initialization_seed=config.initialization_seed,
        )
        key = (
            problem.name,
            config.profile,
            config.problem_seed,
            config.initialization_seed,
            config.sampling_seed,
            config.steps,
            config.batch_size,
            problem.population_size,
        )
        plan = plan_cache.setdefault(
            key,
            SamplePlan.generate(
                problem.population_size,
                config.steps,
                config.batch_size,
                config.sampling_seed,
            ),
        )
        results.append(
            run_experiment(config, problem=problem, sample_plan=plan)
        )
    return results


def registry_configs(
    *,
    problems=("all",),
    methods=("momentum",),
    seeds=(0,),
    **kwargs,
):
    """Expand CLI-style selections into individual configurations."""

    selected = problem_names() if "all" in problems else tuple(problems)
    return [
        ExperimentConfig(
            problem=problem,
            method=method,
            problem_seed=int(seed),
            initialization_seed=int(seed),
            sampling_seed=int(seed),
            **kwargs,
        )
        for problem in selected
        for method in methods
        for seed in seeds
    ]


def run_registry(**kwargs):
    """Convenience wrapper combining :func:`registry_configs` and run_many."""

    return run_many(registry_configs(**kwargs))
