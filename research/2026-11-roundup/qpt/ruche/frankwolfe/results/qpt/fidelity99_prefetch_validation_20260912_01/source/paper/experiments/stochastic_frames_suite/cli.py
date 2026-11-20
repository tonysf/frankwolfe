"""Command-line interface for the stochastic-FRAMES experiment registry."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .config import SUPPORTED_METHODS
from .io import save_result
from .registry import problem_names
from .runner import registry_configs, run_many


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run reproducible stochastic-FRAMES benchmark comparisons."
    )
    parser.add_argument(
        "--problem",
        nargs="+",
        default=["all"],
        help="Registry problem name(s), E0-E6 aliases, or 'all'.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=["momentum", "no-momentum", "deterministic"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--profile", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--reference-beta", type=float, default=1.0)
    parser.add_argument("--rho-scale", type=float, default=1.0)
    parser.add_argument("--smoothing-scale", type=float, default=1.0)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-frequency", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/stochastic_frames")
    )
    parser.add_argument("--metric", default="exact_smoothed_gap")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _validate_problem_selection(parser, selected):
    if "all" in selected and len(selected) != 1:
        parser.error("'all' cannot be combined with explicit problem names.")
    # Registry aliases are resolved by the runner.  This early check keeps
    # misspelled canonical CLI names readable while retaining E0-E6 aliases.
    canonical = set(problem_names())
    aliases = {f"e{index}" for index in range(7)}
    invalid = [
        item
        for item in selected
        if item.lower() not in canonical | aliases | {"all"}
        and item.lower().replace("-", "_") not in canonical
    ]
    if invalid:
        parser.error("unknown problem(s): " + ", ".join(invalid))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_problem_selection(parser, args.problem)
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative integers.")

    configs = registry_configs(
        problems=tuple(item.lower() for item in args.problem),
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        profile=args.profile,
        steps=args.steps,
        batch_size=args.batch_size,
        beta0=args.beta0,
        reference_beta=args.reference_beta,
        rho_scale=args.rho_scale,
        smoothing_scale=args.smoothing_scale,
        step_scale=args.step_scale,
        checkpoint_frequency=args.checkpoint_frequency,
        show_progress=args.show_progress,
    )
    results = run_many(configs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for result in results:
        filename = (
            f"{result.problem_name}__{result.method}__"
            f"p{result.problem_seed}_i{result.initialization_seed}_"
            f"s{result.sampling_seed}.npz"
        )
        path = save_result(args.output_dir / filename, result)
        grouped[result.problem_name].append(result)
        print(path)

    if not args.no_plots:
        from .plotting import save_common_plot

        for problem, problem_results in grouped.items():
            plot_path = args.output_dir / f"{problem}__{args.metric}.png"
            save_common_plot(plot_path, problem_results, metric=args.metric)
            print(plot_path)
    return 0


__all__ = ["build_parser", "main"]
