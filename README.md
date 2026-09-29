# Metaheuristic Optimization & Genetic Algorithms

Python · NumPy · Matplotlib

Portfolio-ready benchmark suite comparing **Genetic Algorithms**, **Simulated Annealing**, and **Hill Climbing** on discrete (TSP) and continuous (Rastrigin, Sphere) problems.

## Features

- Vectorized NumPy fitness evaluation and operators
- Fair multi-run statistics (min / mean / std / max, wall-clock time)
- Convergence curves, boxplots, and mean±std comparison charts
- Reproducible seeds for demos and reports

## Setup

```bash
cd "01-metaheuristic-optimization"
pip install -r requirements.txt
```

## Run

```bash
# Full benchmark (TSP + Rastrigin + Sphere), 30 runs each
python main.py --problem all --algo all --runs 30

# TSP only, faster smoke test
python main.py --problem tsp --runs 5

# Continuous Rastrigin
python main.py --problem rastrigin --dim 5 --runs 10
```

Plots are written to `outputs/`.

## Project layout

```
metaopt/
  algorithms/   # GA, SA, Hill Climbing
  problems/     # continuous + TSP
  benchmark.py  # multi-run harness
  visualize.py  # Matplotlib figures
main.py
```

## What this demonstrates

Implemented and benchmarked core optimization algorithms across discrete and continuous spaces, with statistical evaluation over many iterations and visualization of convergence and objective efficiency.
