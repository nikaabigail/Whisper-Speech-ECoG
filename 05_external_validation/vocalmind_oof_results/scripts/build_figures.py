#!/usr/bin/env python3
"""Build the curated VocalMind OOF tables, figures, and checksum manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SOURCE_CSV = RESULTS / "vocalmind_oof_metrics.csv"
SOURCE_JSON = RESULTS / "vocalmind_oof_summary.json"

EXPECTED_CSV_SHA256 = "e1caa36898f694023e1cc7ae67007a549d06ec08e6d73ad274b0b822ea2c143f"
EXPECTED_JSON_SHA256 = "988988d5b926e6a3b7987a4dc7ec6dad6edd94f8f9baa66b88f0486cf81a4afa"
EXPECTED_RUN_FINGERPRINT = "b0ffb3306022158ad74975b8090333352894892cd904b524ec51c96158cb3980"
EXPECTED_OOF_FINGERPRINT = "c638c98a405bed7df59116c0805d2534cca2bfca8cfbfe8c29eaf7825e5960c7"

MODELS = ["L3", "L4", "L5", "L345", "MELx3"]
SEEDS = [1, 2, 3, 4, 42]
METRICS = ["accuracy", "top3_accuracy", "macro_f1"]
MODEL_LABELS = {
    "L3": "Whisper L3",
    "L4": "Whisper L4",
    "L5": "Whisper L5",
    "L345": "Whisper L3+L4+L5",
    "MELx3": "MEL × 3",
}
COLORS = {
    "L3": "#9aa2aa",
    "L4": "#737d86",
    "L5": "#bcc2c7",
    "L345": "#0072b2",
    "MELx3": "#d55e00",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources() -> dict:
    actual_csv = sha256(SOURCE_CSV)
    actual_json = sha256(SOURCE_JSON)
    if actual_csv != EXPECTED_CSV_SHA256:
        raise RuntimeError(f"Unexpected source CSV SHA-256: {actual_csv}")
    if actual_json != EXPECTED_JSON_SHA256:
        raise RuntimeError(f"Unexpected source JSON SHA-256: {actual_json}")
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    if payload["fingerprint"] != EXPECTED_OOF_FINGERPRINT:
        raise RuntimeError("Unexpected immutable OOF fingerprint")
    if payload["metrics_csv_sha256"] != actual_csv:
        raise RuntimeError("JSON does not authenticate the published metrics CSV")
    if payload["statistical_scope"]["participant_count"] != 1:
        raise RuntimeError("This publication record is explicitly limited to biological n=1")
    return payload


def load_rows() -> tuple[list[dict[str, str]], dict, dict]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fingerprints = {row["run_fingerprint"] for row in rows}
    if fingerprints != {EXPECTED_RUN_FINGERPRINT}:
        raise RuntimeError(f"Unexpected run fingerprints: {sorted(fingerprints)}")

    seed_values: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    summaries: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    contrasts: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["row_type"] == "training_seed":
            seed_values[row["model"]][row["metric"]][int(row["seed"])] = float(
                row["value"]
            )
        elif row["row_type"] == "descriptive_seed_summary":
            summaries[row["model"]][row["metric"]] = {
                "mean": float(row["mean"]),
                "sd": float(row["sample_sd"]),
                "sem": float(row["sem"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
            }
        elif row["row_type"] == "descriptive_seed_primary_contrast":
            contrasts[row["metric"]] = {
                "mean": float(row["mean"]),
                "sd": float(row["sample_sd"]),
                "sem": float(row["sem"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
            }

    for model in MODELS:
        for metric in METRICS:
            if sorted(seed_values[model][metric]) != SEEDS:
                raise RuntimeError(f"Incomplete seed surface for {model}/{metric}")
            values = np.asarray([seed_values[model][metric][seed] for seed in SEEDS])
            summary = summaries[model][metric]
            if not np.isclose(values.mean(), summary["mean"], atol=1e-14):
                raise RuntimeError(f"Mean mismatch for {model}/{metric}")
            if not np.isclose(values.std(ddof=1), summary["sd"], atol=1e-14):
                raise RuntimeError(f"SD mismatch for {model}/{metric}")
    return rows, seed_values, summaries, contrasts


def write_compact_tables(seed_values: dict, summaries: dict, contrasts: dict) -> list[Path]:
    summary_path = RESULTS / "model_summary.csv"
    summary_fields = [
        "model",
        "top1_mean",
        "top1_sd",
        "top1_ci95_low",
        "top1_ci95_high",
        "top3_mean",
        "top3_sd",
        "top3_ci95_low",
        "top3_ci95_high",
        "macro_f1_mean",
        "macro_f1_sd",
        "macro_f1_ci95_low",
        "macro_f1_ci95_high",
        "n_training_seeds",
        "biological_n",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for model in MODELS:
            top1 = summaries[model]["accuracy"]
            top3 = summaries[model]["top3_accuracy"]
            f1 = summaries[model]["macro_f1"]
            writer.writerow(
                {
                    "model": model,
                    "top1_mean": top1["mean"],
                    "top1_sd": top1["sd"],
                    "top1_ci95_low": top1["ci95_low"],
                    "top1_ci95_high": top1["ci95_high"],
                    "top3_mean": top3["mean"],
                    "top3_sd": top3["sd"],
                    "top3_ci95_low": top3["ci95_low"],
                    "top3_ci95_high": top3["ci95_high"],
                    "macro_f1_mean": f1["mean"],
                    "macro_f1_sd": f1["sd"],
                    "macro_f1_ci95_low": f1["ci95_low"],
                    "macro_f1_ci95_high": f1["ci95_high"],
                    "n_training_seeds": len(SEEDS),
                    "biological_n": 1,
                }
            )

    primary_path = RESULTS / "primary_contrast_by_seed.csv"
    primary_fields = [
        "seed",
        "l345_top1",
        "melx3_top1",
        "delta_top1",
        "l345_top3",
        "melx3_top3",
        "delta_top3",
        "l345_macro_f1",
        "melx3_macro_f1",
        "delta_macro_f1",
    ]
    with primary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=primary_fields)
        writer.writeheader()
        for seed in SEEDS:
            l_top1 = seed_values["L345"]["accuracy"][seed]
            m_top1 = seed_values["MELx3"]["accuracy"][seed]
            l_top3 = seed_values["L345"]["top3_accuracy"][seed]
            m_top3 = seed_values["MELx3"]["top3_accuracy"][seed]
            l_f1 = seed_values["L345"]["macro_f1"][seed]
            m_f1 = seed_values["MELx3"]["macro_f1"][seed]
            writer.writerow(
                {
                    "seed": seed,
                    "l345_top1": l_top1,
                    "melx3_top1": m_top1,
                    "delta_top1": l_top1 - m_top1,
                    "l345_top3": l_top3,
                    "melx3_top3": m_top3,
                    "delta_top3": l_top3 - m_top3,
                    "l345_macro_f1": l_f1,
                    "melx3_macro_f1": m_f1,
                    "delta_macro_f1": l_f1 - m_f1,
                }
            )

    expected = {
        "accuracy": np.mean(
            [
                seed_values["L345"]["accuracy"][seed]
                - seed_values["MELx3"]["accuracy"][seed]
                for seed in SEEDS
            ]
        ),
        "top3_accuracy": np.mean(
            [
                seed_values["L345"]["top3_accuracy"][seed]
                - seed_values["MELx3"]["top3_accuracy"][seed]
                for seed in SEEDS
            ]
        ),
        "macro_f1": np.mean(
            [
                seed_values["L345"]["macro_f1"][seed]
                - seed_values["MELx3"]["macro_f1"][seed]
                for seed in SEEDS
            ]
        ),
    }
    for metric, value in expected.items():
        if not np.isclose(value, contrasts[metric]["mean"], atol=1e-14):
            raise RuntimeError(f"Primary contrast mismatch for {metric}")
    return [summary_path, primary_path]


def save_both(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [FIGURES / f"{stem}.png", FIGURES / f"{stem}.svg"]
    fig.savefig(
        paths[0],
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib"},
    )
    fig.savefig(
        paths[1],
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    svg = paths[1].read_text(encoding="utf-8")
    paths[1].write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)
    return paths


def build_overview(seed_values: dict, summaries: dict) -> list[Path]:
    titles = {
        "accuracy": "Top-1 accuracy",
        "top3_accuracy": "Top-3 accuracy",
        "macro_f1": "Macro-F1",
    }
    reference = {"accuracy": 0.05, "top3_accuracy": 0.15, "macro_f1": 0.05}
    limits = {"accuracy": (0, 0.115), "top3_accuracy": (0, 0.32), "macro_f1": (0, 0.115)}
    y = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 5.6), sharey=True)
    for index, (ax, metric) in enumerate(zip(axes, METRICS)):
        means = np.asarray([summaries[model][metric]["mean"] for model in MODELS])
        sds = np.asarray([summaries[model][metric]["sd"] for model in MODELS])
        colors = [COLORS[model] for model in MODELS]
        ax.barh(y, means * 100, xerr=sds * 100, color=colors, alpha=0.88,
                error_kw={"ecolor": "#30343b", "elinewidth": 1.1, "capsize": 3})
        for model_index, model in enumerate(MODELS):
            points = np.asarray(
                [seed_values[model][metric][seed] for seed in SEEDS]
            ) * 100
            jitter = np.linspace(-0.13, 0.13, len(points))
            ax.scatter(points, model_index + jitter, s=18, facecolor="white",
                       edgecolor="#24282d", linewidth=0.7, zorder=3)
            ax.text(
                means[model_index] * 100 + sds[model_index] * 100 + 0.25,
                model_index,
                f"{means[model_index] * 100:.1f}%",
                va="center",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4},
            )
        ax.axvline(reference[metric] * 100, color="#8a9299", linestyle="--", linewidth=1.2)
        ax.text(reference[metric] * 100 + 0.18, -0.60, "1/20" if metric != "top3_accuracy" else "3/20",
                fontsize=8.5, color="#596168")
        ax.set_title(titles[metric], loc="left")
        ax.set_xlim(limits[metric][0] * 100, limits[metric][1] * 100)
        ax.set_xlabel("%")
        ax.grid(axis="x", color="#e4e7ea", linewidth=0.8)
        ax.invert_yaxis()
        if index == 0:
            ax.set_yticks(y, [MODEL_LABELS[model] for model in MODELS])
    fig.suptitle(
        "VocalMind: OOF-классификация 20 произнесённых слов",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "Столбец — среднее по 5 инициализациям, ошибка — SD, белые точки — отдельные seeds; biological n=1.",
        ha="center",
        fontsize=9.5,
        color="#4b535b",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return save_both(fig, "figure_01_oof_model_comparison")


def build_primary_contrast(seed_values: dict, contrasts: dict) -> list[Path]:
    specs = [
        ("accuracy", "Top-1 accuracy", 0.05),
        ("top3_accuracy", "Top-3 accuracy", 0.15),
        ("macro_f1", "Macro-F1", 0.05),
    ]
    y = np.arange(len(SEEDS))
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.9))
    for ax, (metric, title, chance) in zip(axes, specs):
        l345 = np.asarray([seed_values["L345"][metric][seed] for seed in SEEDS]) * 100
        mel = np.asarray([seed_values["MELx3"][metric][seed] for seed in SEEDS]) * 100
        for position, (left, right) in enumerate(zip(l345, mel)):
            ax.plot([left, right], [position, position], color="#bfc5ca", linewidth=1.8, zorder=0)
        ax.scatter(l345, y, color=COLORS["L345"], marker="D", s=54,
                   label="Whisper L3+L4+L5", zorder=2)
        ax.scatter(mel, y, color=COLORS["MELx3"], marker="o", s=54,
                   label="MEL × 3", zorder=2)
        ax.axvline(chance * 100, color="#8a9299", linestyle="--", linewidth=1.1)
        contrast = contrasts[metric]
        low = contrast["ci95_low"] * 100
        high = contrast["ci95_high"] * 100
        ax.set_title(title, loc="left")
        ax.set_yticks(y, [f"seed {seed}" for seed in SEEDS])
        ax.invert_yaxis()
        ax.set_xlabel("%")
        ax.grid(axis="x", color="#e4e7ea", linewidth=0.8)
        ax.text(
            0.03,
            0.04,
            f"Δ L345−MEL = {contrast['mean'] * 100:+.1f} п.п.\n"
            f"95% t-CI [{low:+.1f}; {high:+.1f}]",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d4d8dc"},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.955), fontsize=9.5)
    fig.suptitle(
        "Главный контраст: устойчивого преимущества Whisper над MEL не обнаружено",
        fontsize=14,
        fontweight="bold",
        y=1.06,
    )
    fig.text(
        0.5,
        -0.01,
        "Интервалы описывают разброс 5 обучений одного участника и не являются популяционными доверительными интервалами.",
        ha="center",
        fontsize=9.5,
        color="#4b535b",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    return save_both(fig, "figure_02_primary_contrast")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    validate_sources()
    _, seed_values, summaries, contrasts = load_rows()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": False,
            "svg.hashsalt": "vocalmind-oof-results-v1",
        }
    )

    generated = write_compact_tables(seed_values, summaries, contrasts)
    generated += build_overview(seed_values, summaries)
    generated += build_primary_contrast(seed_values, contrasts)
    manifest = {
        "schema_version": 1,
        "source_artifacts": {
            "results/vocalmind_oof_metrics.csv": sha256(SOURCE_CSV),
            "results/vocalmind_oof_summary.json": sha256(SOURCE_JSON),
        },
        "source_run_fingerprint": EXPECTED_RUN_FINGERPRINT,
        "immutable_oof_fingerprint": EXPECTED_OOF_FINGERPRINT,
        "artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in generated
        },
        "statistical_scope": {
            "training_seeds": SEEDS,
            "biological_n": 1,
            "seed_intervals_are_population_intervals": False,
        },
    }
    manifest_path = FIGURES / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Built {len(generated)} curated artifacts in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
