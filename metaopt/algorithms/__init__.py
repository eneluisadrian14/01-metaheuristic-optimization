from .ga import genetic_algorithm
from .sa import simulated_annealing
from .hill_climbing import hill_climbing

ALGORITHMS = {
    "ga": genetic_algorithm,
    "sa": simulated_annealing,
    "hc": hill_climbing,
}

__all__ = ["genetic_algorithm", "simulated_annealing", "hill_climbing", "ALGORITHMS"]
