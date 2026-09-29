"""Continuous benchmark functions (minimization)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


def sphere(x: np.ndarray) -> float:
    """Sphere: f(x) = sum(x_i^2). Global minimum 0 at origin."""
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin: multimodal benchmark. Global minimum 0 at origin."""
    a = 10.0
    n = x.size
    return float(a * n + np.sum(x ** 2 - a * np.cos(2.0 * np.pi * x)))


@dataclass
class ContinuousProblem:
    """Continuous optimization problem with box constraints."""

    name: str
    fitness_fn: Callable[[np.ndarray], float]
    dimension: int
    bounds: Tuple[float, float]
    seed: int = 42
    discrete: bool = False

    def evaluate(self, x: np.ndarray) -> float:
        return self.fitness_fn(np.asarray(x, dtype=float))

    def random_solution(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(self.seed)
        lo, hi = self.bounds
        return rng.uniform(lo, hi, size=self.dimension)

    def clip(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds
        return np.clip(x, lo, hi)

    def neighbor(self, x: np.ndarray, rng: np.random.Generator, step: float = 0.3) -> np.ndarray:
        noise = rng.normal(0.0, step, size=x.shape)
        return self.clip(x + noise)
