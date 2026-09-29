"""CLI entry point: python -m metaopt ... or python main.py"""

from __future__ import annotations

import argparse
from pathlib import Path

from metaopt.algorithms import ALGORITHMS
from metaopt.benchmark import benchmark, print_summary_table
from metaopt.problems import PROBLEMS
from metaopt.visualize import save_boxplots, save_convergence, save_mean_bars


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Metaheuristic Optimization Benchmark Suite (GA / SA / Hill Climbing)"
    )
    p.add_argument(
        "--problem",
        choices=list(PROBLEMS.keys()) + ["all"],
        default="all",
        help="Benchmark problem (default: all)",
    )
    p.add_argument(
        "--algo",
        choices=list(ALGORITHMS.keys()) + ["all"],
        default="all",
        help="Algorithm to run (default: all)",
    )
    p.add_argument("--runs", type=int, default=30, help="Independent runs (default: 30)")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory for plots",
    )
    p.add_argument("--cities", type=int, default=15, help="TSP city count")
    p.add_argument("--dim", type=int, default=5, help="Continuous problem dimension")
    return p


def run_problem(name: str, args: argparse.Namespace) -> None:
    algos = list(ALGORITHMS.keys()) if args.algo == "all" else [args.algo]
    problem_kwargs = {}
    if name == "tsp":
        problem_kwargs["n_cities"] = args.cities
    else:
        problem_kwargs["dim"] = args.dim

    # Faster defaults for continuous GA when many runs
    algo_kwargs = {}
    if name != "tsp":
        algo_kwargs = {"generations": 80, "population_size": 40, "max_iter": 800}

    print(f"\n=== Benchmark: {name.upper()} | algorithms={algos} | runs={args.runs} ===\n")
    summaries = benchmark(
        problem_name=name,
        algorithms=algos,
        runs=args.runs,
        seed=args.seed,
        problem_kwargs=problem_kwargs,
        algo_kwargs=algo_kwargs,
    )
    print_summary_table(summaries)

    out = Path(args.output)
    c = save_convergence(summaries, out)
    b = save_boxplots(summaries, out)
    m = save_mean_bars(summaries, out)
    print(f"\nPlots saved:\n  {c}\n  {b}\n  {m}")


def main() -> None:
    args = build_parser().parse_args()
    problems = list(PROBLEMS.keys()) if args.problem == "all" else [args.problem]
    for name in problems:
        run_problem(name, args)


if __name__ == "__main__":
    main()
