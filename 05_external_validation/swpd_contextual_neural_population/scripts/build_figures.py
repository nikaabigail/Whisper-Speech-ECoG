#!/usr/bin/env python3
"""Build the frozen neural population publication figures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SUMMARY = RESULTS / "final_summary.json"
AUTHORS = RESULTS / "authors_figure4a_digitized.csv"

AUTHOR_COLOR = "#666b73"
OURS_COLOR = "#0072b2"
MEL_COLOR = "#8b949e"
LINEAR_COLOR = "#009e73"
GRID_COLOR = "#e1e5e9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_both(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{stem}.png", FIGURES / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", metadata={"Date": None})
    svg = paths[1].read_text(encoding="utf-8")
    paths[1].write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)
    return paths


def load_authors() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with AUTHORS.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "subject": row["subject"],
                    "authors": float(row["authors_published_r_approx"]),
                    "ours": float(row["ours_fixed_neural_r"]) if row["ours_fixed_neural_r"] else None,
                    "status": row["ours_status"],
                }
            )
    return rows


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = load_authors()
    shared = [row for row in rows if row["ours"] is not None]
    author_shared_mean = float(np.mean([row["authors"] for row in shared]))
    ours_shared_mean = float(summary["fixed_neural_whisper_l4"]["mean"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "svg.hashsalt": "swpd-fixed-neural-population-v1",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), gridspec_kw={"width_ratios": [1.35, 0.9]})

    ax = axes[0]
    y = np.arange(len(rows))
    for index, row in enumerate(rows):
        author_value = float(row["authors"])
        ours_value = row["ours"]
        if ours_value is not None:
            ax.plot([author_value, ours_value], [index, index], color="#c4c9ce", linewidth=1.4, zorder=1)
        ax.scatter(author_value, index, s=58, facecolor="white", edgecolor=AUTHOR_COLOR,
                   linewidth=1.7, marker="o", zorder=3)
        if ours_value is not None:
            ax.scatter(float(ours_value), index, s=54, color=OURS_COLOR, edgecolor="#26313a",
                       linewidth=0.7, marker="D", zorder=4)
        else:
            label = "development" if row["subject"] == "sub-01" else "QC-исключение"
            ax.text(0.905, index, label, ha="right", va="center", fontsize=8.5, color="#6d747b")
    ax.set_yticks(y, [str(row["subject"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.47, 0.91)
    ax.set_xlabel("Pearson r")
    ax.set_title("a  По пациентам", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    author_handle = ax.scatter([], [], s=58, facecolor="white", edgecolor=AUTHOR_COLOR,
                               linewidth=1.7, marker="o")
    ours_handle = ax.scatter([], [], s=54, color=OURS_COLOR, edgecolor="#26313a",
                             linewidth=0.7, marker="D")

    ax = axes[1]
    labels = ["Авторы\n(≈ Fig. 4a)", "Прямой\nMEL80",
              "Линейный\nWhisper L4", "Нейросетевой\nWhisper L4"]
    means = np.asarray(
        [
            author_shared_mean,
            summary["direct_mel80_control"]["mean"],
            summary["linear_whisper_l4"]["mean"],
            ours_shared_mean,
        ]
    )
    colors = [AUTHOR_COLOR, MEL_COLOR, LINEAR_COLOR, OURS_COLOR]
    positions = np.arange(len(labels))
    ax.bar(positions, means, width=0.62, color=colors, alpha=0.88)
    for index, value in enumerate(means):
        ax.text(index, value + 0.006, f"{value:.3f}".replace(".", ","),
                ha="center", va="bottom", fontweight="bold")
    ci_low, ci_high = summary["fixed_neural_whisper_l4"]["ci95_t"]
    ax.errorbar(
        positions[-1],
        ours_shared_mean,
        yerr=[[ours_shared_mean - ci_low], [ci_high - ours_shared_mean]],
        fmt="none",
        ecolor="#26313a",
        elinewidth=1.6,
        capsize=5,
        zorder=4,
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 0.85)
    ax.set_ylabel("Средний Pearson r")
    ax.set_title("b  Средние по общей группе", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.text(3.0, ci_high + 0.012, "95% CI", ha="center", fontsize=8.5, color="#565e66")

    fig.suptitle(
        "SWPD: опубликованные авторские результаты и наша последняя модель",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        [author_handle, ours_handle],
        ["Авторы, Fig. 4a (≈ с графика)", "Наша fixed-neural Whisper L4"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.01,
        "Визуальное сопоставление, не прямой statistical test: авторы — MEL23/10-fold; мы — MEL80/strict temporal block5.",
        ha="center",
        fontsize=9,
        color="#4f565d",
    )
    fig.subplots_adjust(top=0.83, bottom=0.19, left=0.08, right=0.98, wspace=0.25)
    generated = save_both(fig, "figure_01_authors_vs_latest")

    sources = [SUMMARY, AUTHORS]
    manifest = {
        "schema_version": 1,
        "sources": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sources},
        "artifacts": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in generated},
        "authors_values_note": "Approximate values visually digitized from Verwoert et al. 2022 Figure 4a; not exact source data.",
    }
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Built {len(generated)} artifacts; shared author mean~{author_shared_mean:.3f}; ours={ours_shared_mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
