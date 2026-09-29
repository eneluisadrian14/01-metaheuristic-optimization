"""Matplotlib visualizations for convergence and statistical comparison."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .benchmark import BenchmarkSummary


def save_convergence(summaries: List[BenchmarkSummary], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in summaries:
        if not s.sample_history:
            continue
        ax.plot(s.sample_history, label=s.algorithm.upper(), linewidth=1.8)
    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Best Objective Value")
    ax.set_title(f"Convergence — {summaries[0].problem}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / f"convergence_{summaries[0].problem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_boxplots(summaries: List[BenchmarkSummary], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = [s.fitness_values for s in summaries]
    labels = [s.algorithm.upper() for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.set_ylabel("Final Objective Value")
    ax.set_title(f"Distribution over {summaries[0].runs} runs — {summaries[0].problem}")
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / f"boxplot_{summaries[0].problem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_mean_bars(summaries: List[BenchmarkSummary], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    means = [s.stats()["mean"] for s in summaries]
    stds = [s.stats()["std"] for s in summaries]
    labels = [s.algorithm.upper() for s in summaries]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=5, color=["#2E86AB", "#E94F37", "#44AF69"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Objective (± std)")
    ax.set_title(f"Algorithm Efficiency — {summaries[0].problem}")
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / f"mean_bars_{summaries[0].problem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
