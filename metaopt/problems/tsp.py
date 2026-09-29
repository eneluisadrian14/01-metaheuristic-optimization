"""Traveling Salesman Problem (discrete permutation encoding)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TSPProblem:
    """TSP on a complete graph with Euclidean/random distances."""

    name: str
    distance_matrix: np.ndarray
    seed: int = 42
    discrete: bool = True

    @property
    def n_cities(self) -> int:
        return int(self.distance_matrix.shape[0])

    def evaluate(self, tour: np.ndarray) -> float:
        tour = np.asarray(tour, dtype=int)
        total = 0.0
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            total += self.distance_matrix[a, b]
        return float(total)

    def random_solution(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(self.seed)
        tour = np.arange(self.n_cities)
        rng.shuffle(tour)
        return tour

    def neighbor(self, tour: np.ndarray, rng: np.random.Generator, **_kwargs) -> np.ndarray:
        """2-opt style swap of two cities."""
        new_tour = tour.copy()
        i, j = rng.choice(self.n_cities, size=2, replace=False)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour


def create_tsp(n_cities: int = 15, seed: int = 42) -> TSPProblem:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n_cities, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, 0.0)
    return TSPProblem(name="tsp", distance_matrix=dist, seed=seed)
