"""Genetic Algorithm for discrete (permutation) and continuous problems."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def _tournament_select(fitness: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    idxs = rng.integers(0, len(fitness), size=k)
    return int(idxs[np.argmin(fitness[idxs])])


def _crossover_continuous(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.random(p1.shape)
    return alpha * p1 + (1.0 - alpha) * p2


def _crossover_order(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order crossover (OX) for TSP permutations."""
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[a : b + 1] = p1[a : b + 1]
    fill = [gene for gene in p2 if gene not in child]
    ptr = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[ptr]
            ptr += 1
    return child


def _mutate_continuous(
    ind: np.ndarray,
    rng: np.random.Generator,
    rate: float,
    bounds: tuple[float, float],
) -> np.ndarray:
    out = ind.copy()
    mask = rng.random(out.shape) < rate
    noise = rng.normal(0.0, 0.4, size=out.shape)
    out = np.where(mask, out + noise, out)
    return np.clip(out, bounds[0], bounds[1])


def _mutate_swap(ind: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    out = ind.copy()
    if rng.random() < rate:
        i, j = rng.choice(len(out), size=2, replace=False)
        out[i], out[j] = out[j], out[i]
    return out


def genetic_algorithm(
    problem: Any,
    *,
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.15,
    elite_count: int = 2,
    seed: int | None = None,
    **_kwargs,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Generational GA with tournament selection, elitism, and problem-specific operators.
    """
    rng = np.random.default_rng(seed)
    discrete = getattr(problem, "discrete", False)

    population = np.array([problem.random_solution(rng) for _ in range(population_size)])
    history: List[float] = []

    for _ in range(generations):
        fitness = np.array([problem.evaluate(ind) for ind in population])
        elite_idx = np.argsort(fitness)[:elite_count]
        elites = population[elite_idx].copy()
        best_fit = float(fitness[elite_idx[0]])
        history.append(best_fit)

        next_pop = list(elites)
        while len(next_pop) < population_size:
            i1 = _tournament_select(fitness, rng)
            i2 = _tournament_select(fitness, rng)
            p1, p2 = population[i1], population[i2]
            if discrete:
                child = _crossover_order(p1, p2, rng)
                child = _mutate_swap(child, rng, mutation_rate)
            else:
                child = _crossover_continuous(p1, p2, rng)
                child = _mutate_continuous(child, rng, mutation_rate, problem.bounds)
            next_pop.append(child)
        population = np.array(next_pop)

    fitness = np.array([problem.evaluate(ind) for ind in population])
    best_i = int(np.argmin(fitness))
    history.append(float(fitness[best_i]))
    return population[best_i].copy(), float(fitness[best_i]), history
