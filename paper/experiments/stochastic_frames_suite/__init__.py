"""Reusable stochastic-FRAMES benchmark suite.

The suite is deliberately an adapter around :class:`frank_wolfe.StochasticFrames`.
It adds reproducible finite-sum sampling, benchmark definitions, structured
results, and reporting without changing the optimizer used by the QPT code.
"""

from .config import ExperimentConfig
from .io import load_result, save_result
from .problem import BenchmarkProblem, ObjectiveAdapter, SamplePlan
from .registry import create_problem, problem_names
from .results import RunResult, aggregate_results
from .runner import run_experiment, run_many

__all__ = [
    "BenchmarkProblem",
    "ExperimentConfig",
    "ObjectiveAdapter",
    "RunResult",
    "SamplePlan",
    "aggregate_results",
    "create_problem",
    "load_result",
    "problem_names",
    "run_experiment",
    "run_many",
    "save_result",
]
