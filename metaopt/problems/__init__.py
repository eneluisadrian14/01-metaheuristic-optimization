from .continuous import sphere, rastrigin, ContinuousProblem
from .tsp import TSPProblem, create_tsp

PROBLEMS = {
    "sphere": lambda dim=5, seed=42: ContinuousProblem("sphere", sphere, dim, (-5.12, 5.12), seed),
    "rastrigin": lambda dim=5, seed=42: ContinuousProblem("rastrigin", rastrigin, dim, (-5.12, 5.12), seed),
    "tsp": lambda n_cities=15, seed=42: create_tsp(n_cities, seed),
}

__all__ = ["sphere", "rastrigin", "ContinuousProblem", "TSPProblem", "create_tsp", "PROBLEMS"]
