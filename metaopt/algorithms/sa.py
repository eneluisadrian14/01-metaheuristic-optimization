"""Simulated Annealing for discrete and continuous problems."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def simulated_annealing(
    problem: Any,
    *,
    temp_initial: float = 1000.0,
    cooling_rate: float = 0.95,
    temp_min: float = 0.1,
    iters_per_temp: int = 50,
    seed: int | None = None,
    initial: np.ndarray | None = None,
    **_kwargs,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Simulated Annealing with exponential cooling.

    Accepts worse moves with probability exp(-delta / T).
    """
    rng = np.random.default_rng(seed)
    current = initial.copy() if initial is not None else problem.random_solution(rng)
    current_fit = problem.evaluate(current)
    best, best_fit = current.copy(), current_fit
    history: List[float] = [best_fit]
    temperature = temp_initial

    while temperature > temp_min:
        for _ in range(iters_per_temp):
            candidate = problem.neighbor(current, rng)
            candidate_fit = problem.evaluate(candidate)
            delta = candidate_fit - current_fit
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current, current_fit = candidate, candidate_fit
                if current_fit < best_fit:
                    best, best_fit = current.copy(), current_fit
            history.append(best_fit)
        temperature *= cooling_rate

    return best, best_fit, history
