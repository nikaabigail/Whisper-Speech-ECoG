#!/usr/bin/env python3
"""Render the two summary figures from ``reported_metrics.json``.

The script is deliberately data-only: it does not import training code or open
participant recordings/checkpoints.  Run it from any working directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_METRICS = HERE / "reported_metrics.json"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(fig)
    print(f"[saved] {path}")


def plot_synchronous(metrics: dict, output: Path) -> None:
    values = metrics["synchronous"]["single_seed_accuracy"]
    labels = list(values)
    scores = np.asarray([values[label] for label in labels], dtype=float)
    y = np.arange(len(labels))
    colors = ["#8C96A0", "#8C96A0", "#8C96A0", "#176B87"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.hlines(y, 0.70, scores, color=colors, linewidth=3.0, alpha=0.58)
    ax.scatter(scores, y, s=[78, 78, 78, 112], color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    for yi, value in zip(y, scores):
        ax.text(value + 0.004, yi, f"{100 * value:.2f}%", va="center", ha="left", fontweight="medium")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.70, 0.84)
    ax.set_xticks(np.arange(0.70, 0.841, 0.02))
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.grid(axis="x", color="#D9DDE1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Word-classification accuracy")
    ax.set_title("Synchronous Whisper decoder: fixed seed-4 test")
    ax.text(
        0.0,
        -0.23,
        "Patient alias: ivanova  |  test n=445  |  one fixed seed (no confidence interval)",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9.5,
    )
    _save(fig, output)


def plot_asynchronous_ci(metrics: dict, output: Path) -> None:
    section = metrics["asynchronous"]["continuous_head_multiseed"]
    values = section["f1_by_model"]
    labels = list(values)
    means = np.asarray([values[label]["mean"] for label in labels], dtype=float)
    lows = np.asarray([values[label]["ci95"][0] for label in labels], dtype=float)
    highs = np.asarray([values[label]["ci95"][1] for label in labels], dtype=float)
    y = np.arange(len(labels))
    colors = ["#8C96A0", "#8C96A0", "#8C96A0", "#176B87"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    for yi, mean, low, high, color in zip(y, means, lows, highs, colors):
        ax.errorbar(
            mean,
            yi,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.1,
            capsize=4,
            capthick=1.6,
            markersize=7.5 if yi < 3 else 9.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.text(high + 0.003, yi, f"{mean:.3f}", va="center", ha="left", fontweight="medium")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.535)
    ax.set_xticks(np.arange(0.40, 0.531, 0.02))
    ax.grid(axis="x", color="#D9DDE1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Event-level F1")
    ax.set_title("Asynchronous continuous decoder: mean F1 and 95% t-CI")
    ax.text(
        0.0,
        -0.23,
        "n=5 continuous-head seeds  |  frozen upstream seed 4  |  threshold selected on validation",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9.5,
    )
    _save(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--sync-output", type=Path, default=HERE / "sync" / "sync_l345_seed4.png")
    parser.add_argument("--async-output", type=Path, default=HERE / "async" / "figures" / "async_f1_ci.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.metrics.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    _style()
    plot_synchronous(metrics, args.sync_output)
    plot_asynchronous_ci(metrics, args.async_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
