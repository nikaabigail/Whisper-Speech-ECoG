#!/usr/bin/env python3
"""Build publication figures and a checksum manifest from the curated result."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "population_summary.json"
FIGURES = ROOT / "figures"

MEL_COLOR = "#59636e"
L4_COLOR = "#0072b2"
DELTA_COLOR = "#009e73"
NEGATIVE_COLOR = "#d55e00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_both(fig: plt.Figure, stem: str) -> list[Path]:
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


def main() -> int:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = payload["subject_rows"]
    subjects = [row["subject"] for row in rows]
    mel = np.asarray([row["direct_mel80_r"] for row in rows])
    l4 = np.asarray([row["whisper_l4_pca50_r"] for row in rows])
    delta = l4 - mel
    mel_low = np.asarray([row["direct_mel80_low20_r"] for row in rows])
    l4_low = np.asarray([row["whisper_l4_pca50_low20_r"] for row in rows])
    delta_low = l4_low - mel_low
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.grid": False,
        "svg.hashsalt": "swpd-contextual-frozen-v1",
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    for index, subject in enumerate(subjects):
        ax.plot([0, 1], [mel[index], l4[index]], color="#b5bbc1", linewidth=1.2, zorder=1)
        ax.scatter(0, mel[index], color=MEL_COLOR, marker="o", s=45, zorder=2)
        ax.scatter(1, l4[index], color=L4_COLOR, marker="D", s=42, zorder=2)
        ax.text(1.06, l4[index], subject, va="center", fontsize=8.5, color="#31363b")
    ax.set_xlim(-0.28, 1.42)
    ax.set_ylim(min(mel.min(), l4.min()) - 0.035, max(mel.max(), l4.max()) + 0.035)
    ax.set_xticks([0, 1], ["Прямой MEL80", "Whisper L4→PCA50"])
    ax.set_ylabel("Средняя held-out корреляция на MEL80, r")
    ax.set_title("a  Парные результаты пациентов", loc="left")
    ax.grid(axis="y", color="#e4e7ea", linewidth=0.8)

    ax = axes[1]
    positions = np.arange(len(subjects))
    colors = [DELTA_COLOR if value >= 0 else NEGATIVE_COLOR for value in delta]
    ax.axvline(0, color="#6f7780", linewidth=1)
    for y, value, color in zip(positions, delta * 1000, colors):
        ax.plot([0, value], [y, y], color=color, linewidth=1.5)
        ax.scatter(value, y, color=color, s=42, zorder=2)
        ax.text(value + 0.08, y, f"{value:+.2f}",
                ha="left", va="center", fontsize=8.5)
    summary = payload["primary_inference"]["delta_l4_minus_mel80"]
    mean = summary["mean"] * 1000
    low, high = np.asarray(summary["ci95_t"]) * 1000
    mean_y = len(subjects) + 0.6
    ax.errorbar(mean, mean_y, xerr=[[mean - low], [high - mean]], fmt="D",
                color=L4_COLOR, capsize=4, linewidth=2, markersize=6)
    ax.text(high + 0.08, mean_y, f"среднее {mean:+.2f}  [95% CI {low:+.2f}; {high:+.2f}]",
            va="center", fontsize=8.5)
    ax.set_yticks(list(positions) + [mean_y], subjects + ["Среднее"])
    ax.invert_yaxis()
    ax.set_xlabel("Δr = L4 − MEL80, ×10⁻³")
    ax.set_title("b  Эффект Whisper относительно контроля", loc="left")
    ax.grid(axis="x", color="#e4e7ea", linewidth=0.8)
    ax.text(0.02, -0.16, "6/8 выигрышей; paired t p=0,046; sign p=0,289",
            transform=ax.transAxes, fontsize=9, color="#4b535b")
    fig.suptitle("SWPD frozen contextual: Whisper L4 сопоставим с прямым MEL80", fontsize=14, fontweight="bold")
    fig.tight_layout()
    generated = save_both(fig, "figure_01_frozen_main")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    specifications = [
        (axes[0], delta, payload["primary_inference"]["delta_l4_minus_mel80"], "Все 80 MEL-бинов", "p=0,046"),
        (axes[1], delta_low, payload["lower_20_inference"]["delta_l4_minus_mel80"], "Нижние 20 MEL-бинов", "p=0,060"),
    ]
    for ax, values, summary, title, p_text in specifications:
        scaled = values * 1000
        ax.axvline(0, color="#6f7780", linewidth=1)
        ax.scatter(scaled, positions, c=[DELTA_COLOR if value >= 0 else NEGATIVE_COLOR for value in values], s=44)
        mean = summary["mean"] * 1000
        low, high = np.asarray(summary["ci95_t"]) * 1000
        mean_y = len(subjects) + 0.55
        ax.errorbar(mean, mean_y, xerr=[[mean-low], [high-mean]], fmt="D",
                    color=L4_COLOR, capsize=4, linewidth=2, markersize=6)
        ax.text(mean, mean_y - 0.55, f"Δ={mean:+.2f}×10⁻³; {p_text}",
                ha="center", va="top", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Δr = L4 − MEL80, ×10⁻³")
        ax.set_yticks(list(positions) + [mean_y], subjects + ["Среднее"])
        ax.set_ylim(-0.7, mean_y + 0.8)
        ax.grid(axis="x", color="#e4e7ea", linewidth=0.8)
    axes[0].set_ylabel("Пациент")
    fig.suptitle("Проверка эффекта по спектральному диапазону", y=0.98,
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.10, right=0.98, wspace=0.20)
    generated += save_both(fig, "figure_02_all_vs_low20")

    table_path = ROOT / "results" / "frozen_subject_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    generated.append(table_path)
    manifest = {
        "schema_version": 1,
        "source": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(RESULT),
        "artifacts": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in generated},
    }
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Built {len(generated)} publication artifacts in {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
