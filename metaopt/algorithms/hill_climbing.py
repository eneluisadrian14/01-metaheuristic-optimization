"""Hill Climbing (local search) for discrete and continuous problems."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def hill_climbing(
    problem: Any,
    *,
    max_iter: int = 1000,
    seed: int | None = None,
    initial: np.ndarray | None = None,
    **_kwargs,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Greedy local search: always accept improving neighbors.

    Returns best solution, best fitness, and fitness history.
    """
    rng = np.random.default_rng(seed)
    current = initial.copy() if initial is not None else problem.random_solution(rng)
    current_fit = problem.evaluate(current)
    history: List[float] = [current_fit]
    best, best_fit = current.copy(), current_fit

    for _ in range(max_iter):
        candidate = problem.neighbor(current, rng)
        candidate_fit = problem.evaluate(candidate)
        if candidate_fit < current_fit:
            current, current_fit = candidate, candidate_fit
            if current_fit < best_fit:
                best, best_fit = current.copy(), current_fit
        history.append(best_fit)

    return best, best_fit, history
