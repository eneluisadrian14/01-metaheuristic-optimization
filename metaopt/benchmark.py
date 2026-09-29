"""Multi-run statistical benchmarking harness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .algorithms import ALGORITHMS
from .problems import PROBLEMS


@dataclass
class RunResult:
    algorithm: str
    problem: str
    best_fitness: float
    elapsed_sec: float
    history: List[float]


@dataclass
class BenchmarkSummary:
    algorithm: str
    problem: str
    runs: int
    fitness_values: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    sample_history: List[float] = field(default_factory=list)

    def stats(self) -> Dict[str, float]:
        arr = np.asarray(self.fitness_values, dtype=float)
        tarr = np.asarray(self.times, dtype=float)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "mean_time": float(np.mean(tarr)),
        }


def run_once(
    algo_name: str,
    problem: Any,
    seed: int,
    algo_kwargs: Optional[Dict] = None,
) -> RunResult:
    fn: Callable = ALGORITHMS[algo_name]
    kwargs = algo_kwargs or {}
    t0 = time.perf_counter()
    _sol, fitness, history = fn(problem, seed=seed, **kwargs)
    elapsed = time.perf_counter() - t0
    return RunResult(algo_name, problem.name, fitness, elapsed, history)


def benchmark(
    problem_name: str = "tsp",
    algorithms: Optional[List[str]] = None,
    runs: int = 30,
    seed: int = 42,
    problem_kwargs: Optional[Dict] = None,
    algo_kwargs: Optional[Dict] = None,
) -> List[BenchmarkSummary]:
    """Run multiple independent trials and aggregate statistics."""
    algorithms = algorithms or list(ALGORITHMS.keys())
    factory = PROBLEMS[problem_name]
    problem = factory(**(problem_kwargs or {}), seed=seed)

    summaries: List[BenchmarkSummary] = []
    for algo in algorithms:
        summary = BenchmarkSummary(algorithm=algo, problem=problem.name, runs=runs)
        for i in range(runs):
            run_seed = seed + i * 997
            # Fresh problem instance per run for TSP/continuous randomness consistency
            problem_i = factory(**(problem_kwargs or {}), seed=seed)
            result = run_once(algo, problem_i, run_seed, algo_kwargs)
            summary.fitness_values.append(result.best_fitness)
            summary.times.append(result.elapsed_sec)
            if i == 0:
                summary.sample_history = result.history
        summaries.append(summary)
    return summaries


def print_summary_table(summaries: List[BenchmarkSummary]) -> None:
    header = f"{'Algorithm':<8} {'Problem':<12} {'Min':>10} {'Mean':>10} {'Std':>10} {'Max':>10} {'Time(s)':>10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        st = s.stats()
        print(
            f"{s.algorithm:<8} {s.problem:<12} "
            f"{st['min']:10.4f} {st['mean']:10.4f} {st['std']:10.4f} "
            f"{st['max']:10.4f} {st['mean_time']:10.4f}"
        )
