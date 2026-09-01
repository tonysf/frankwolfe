"""Benchmark registry for the seven first-batch stochastic problems."""

from __future__ import annotations

from collections import OrderedDict

from .problems.factorized_correlation import make_problem as factorized_correlation
from .problems.matrix_completion import make_problem as matrix_completion
from .problems.phase_retrieval import make_problem as phase_retrieval
from .problems.robust_portfolio import make_problem as robust_portfolio
from .problems.simplex_regression import make_problem as simplex_regression
from .problems.sparse_logistic import make_problem as sparse_logistic
from .problems.tv_denoising import make_problem as tv_denoising


PROBLEM_REGISTRY = OrderedDict(
    (
        ("simplex_regression", simplex_regression),
        ("sparse_logistic", sparse_logistic),
        ("tv_denoising", tv_denoising),
        ("robust_portfolio", robust_portfolio),
        ("matrix_completion", matrix_completion),
        ("phase_retrieval", phase_retrieval),
        ("factorized_correlation", factorized_correlation),
    )
)

ALIASES = {
    "e0": "simplex_regression",
    "e1": "sparse_logistic",
    "e2": "tv_denoising",
    "e3": "robust_portfolio",
    "e4": "matrix_completion",
    "e5": "phase_retrieval",
    "e6": "factorized_correlation",
    "online-simplex-regression": "simplex_regression",
    "sparse-logistic": "sparse_logistic",
    "tv-denoising": "tv_denoising",
    "robust-portfolio": "robust_portfolio",
    "matrix-completion": "matrix_completion",
    "phase-retrieval": "phase_retrieval",
    "factorized-correlation": "factorized_correlation",
}


def problem_names():
    return tuple(PROBLEM_REGISTRY)


def canonical_problem_name(name):
    normalized = str(name).strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in PROBLEM_REGISTRY:
        available = ", ".join(PROBLEM_REGISTRY)
        raise KeyError(f"Unknown problem {name!r}. Available: {available}.")
    return normalized


def create_problem(
    name,
    *,
    profile="tiny",
    problem_seed=0,
    initialization_seed=0,
):
    canonical = canonical_problem_name(name)
    return PROBLEM_REGISTRY[canonical](
        profile=profile,
        problem_seed=problem_seed,
        initialization_seed=initialization_seed,
    )
