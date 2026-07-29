#!/usr/bin/env python3
"""Build publication figures for the frozen SWPD sub-01 bottleneck study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


TARGETS = ("mel80", "L3", "L4", "L5", "L345")
TARGET_LABELS = {
    "mel80": "MEL80\nконтроль",
    "L3": "Whisper\nL3",
    "L4": "Whisper\nL4",
    "L5": "Whisper\nL5",
    "L345": "Whisper\nL3+L4+L5",
}
METHODS = ("pca50", "srrr50", "clip50", "alternating50")
METHOD_LABELS = {
    "pca50": "PCA50",
    "srrr50": "обучаемый RRR50",
    "clip50": "CLIP50",
    "alternating50": "чередование50",
}
COLORS = {
    "pca50": "#0072B2",
    "srrr50": "#E69F00",
    "clip50": "#D55E00",
    "alternating50": "#009E73",
}
MARKERS = {"pca50": "o", "srrr50": "s", "clip50": "^", "alternating50": "D"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold_values(
    pca_run: Path, clip_run: Path, alternating_run: Path
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    values: dict[str, dict[str, dict[str, list[float]]]] = {
        method: {
            target: {"mel80": [], "low20": []} for target in TARGETS
        }
        for method in METHODS
    }
    for fold in range(5):
        phase = load_json(pca_run / f"fold_{fold:02d}" / "fold_result.json")
        for target in TARGETS:
            for method in ("pca50", "srrr50"):
                result = phase["results"][f"{target}__{method}"]["test"]
                values[method][target]["mel80"].append(
                    result["mel80_probe"]["fisher_z_component_correlation"]
                )
                values[method][target]["low20"].append(
                    result["mel_low20_probe"]["fisher_z_component_correlation"]
                )
            clip = load_json(clip_run / f"fold_{fold:02d}" / target / "result.json")
            values["clip50"][target]["mel80"].append(
                clip["test_mel80_probe"]["fisher_z_component_correlation"]
            )
            values["clip50"][target]["low20"].append(
                clip["test_mel_low20_probe"]["fisher_z_component_correlation"]
            )
            alternating = load_json(
                alternating_run / f"fold_{fold:02d}" / target / "result.json"
            )
            values["alternating50"][target]["mel80"].append(
                alternating["test_mel80_probe"]["fisher_z_component_correlation"]
            )
            values["alternating50"][target]["low20"].append(
                alternating["test_mel_low20_probe"]["fisher_z_component_correlation"]
            )
    return {
        method: {
            target: {
                metric: np.asarray(entries, dtype=np.float64)
                for metric, entries in metrics.items()
            }
            for target, metrics in targets.items()
        }
        for method, targets in values.items()
    }


def describe(values: np.ndarray) -> dict[str, float | int | list[float]]:
    count = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    sem = sd / np.sqrt(count)
    critical = float(stats.t.ppf(0.975, df=count - 1))
    return {
        "mean": mean,
        "sd": sd,
        "n_folds": count,
        "ci95_t": [mean - critical * sem, mean + critical * sem],
    }


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "swpd-sub01-learned-bottleneck-v1",
        }
    )


def save_figure(figure: plt.Figure, output: Path, stem: str) -> list[Path]:
    png = output / f"{stem}.png"
    svg = output / f"{stem}.svg"
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "SWPD learned-bottleneck figure builder"},
    )
    figure.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "SWPD learned-bottleneck figure builder"},
    )
    # Matplotlib emits trailing spaces in SVG path data. Normalize the text so
    # repository whitespace checks and repeated builds remain clean.
    normalized_svg = "\n".join(
        line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
    ) + "\n"
    svg.write_text(normalized_svg, encoding="utf-8", newline="\n")
    plt.close(figure)
    return [png, svg]


def performance_figure(
    data: dict[str, dict[str, dict[str, np.ndarray]]],
    output: Path,
    metric: str,
    stem: str,
    ylabel: str,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(TARGETS), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        means = []
        lower = []
        upper = []
        for target in TARGETS:
            summary = describe(data[method][target][metric])
            means.append(summary["mean"])
            lower.append(summary["mean"] - summary["ci95_t"][0])
            upper.append(summary["ci95_t"][1] - summary["mean"])
        axis.errorbar(
            x + offset,
            means,
            yerr=np.asarray([lower, upper]),
            fmt=MARKERS[method],
            color=COLORS[method],
            markerfacecolor="white" if method != "pca50" else COLORS[method],
            markeredgewidth=1.4,
            markersize=6,
            capsize=3,
            linewidth=1.5,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    axis.axvline(0.5, color="#777777", linewidth=1, linestyle="--", alpha=0.65)
    axis.text(0.02, 0.98, "акустический контроль", transform=axis.transAxes, va="top", color="#555555")
    axis.text(0.30, 0.98, "целевые признаки Whisper", transform=axis.transAxes, va="top", color="#555555")
    axis.set_xticks(x, [TARGET_LABELS[target] for target in TARGETS])
    axis.set_ylabel(ylabel)
    axis.set_title("PCA50 остаётся лучшим на общей поверхности акустического восстановления")
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.text(
        0,
        -0.20,
        "Точки: среднее по фолдам. Интервалы: описательный 95%-й t-интервал по пяти временным фолдам sub-01; не популяционный ДИ.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    return save_figure(figure, output, stem)


def delta_figure(
    data: dict[str, dict[str, dict[str, np.ndarray]]], output: Path
) -> list[Path]:
    methods = ("srrr50", "clip50", "alternating50")
    targets = ("L3", "L4", "L5", "L345")
    figure, axes = plt.subplots(1, len(targets), figsize=(11.2, 4.5), sharey=True)
    rng = np.random.default_rng(42)
    for axis, target in zip(axes, targets):
        for index, method in enumerate(methods):
            delta = data[method][target]["mel80"] - data["pca50"][target]["mel80"]
            jitter = rng.normal(0, 0.035, size=len(delta))
            axis.scatter(
                np.full(len(delta), index) + jitter,
                delta,
                s=24,
                facecolors="white",
                edgecolors=COLORS[method],
                linewidths=1.2,
                zorder=3,
            )
            summary = describe(delta)
            mean = summary["mean"]
            ci = summary["ci95_t"]
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - ci[0]], [ci[1] - mean]],
                fmt=MARKERS[method],
                color=COLORS[method],
                markerfacecolor=COLORS[method],
                capsize=3,
                markersize=6,
                linewidth=1.5,
                zorder=4,
            )
        axis.axhline(0, color="#444444", linewidth=1)
        axis.set_title(TARGET_LABELS[target].replace("\n", " "))
        axis.set_xticks(range(len(methods)), ["sRRR", "CLIP", "чередование"], rotation=30, ha="right")
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.7)
    axes[0].set_ylabel("Изменение общего MEL80 Fisher r относительно PCA50")
    figure.suptitle("Обучаемое сжатие не улучшило результат на отложенных временных блоках", y=1.02)
    figure.text(
        0.5,
        -0.04,
        "Пустые точки: пять временных фолдов. Закрашенная точка и интервал: среднее и описательный 95%-й t-интервал; не вывод на уровне пациентов.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    return save_figure(figure, output, "figure_03_delta_vs_pca50")


def write_table(
    data: dict[str, dict[str, dict[str, np.ndarray]]], output: Path
) -> Path:
    path = output / "table_01_bottleneck_performance.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method", "target", "metric", "mean", "sd", "ci95_low", "ci95_high", "n_temporal_folds"
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for method in METHODS:
            for target in TARGETS:
                for metric in ("mel80", "low20"):
                    result = describe(data[method][target][metric])
                    writer.writerow(
                        {
                            "method": method,
                            "target": target,
                            "metric": metric,
                            "mean": f"{result['mean']:.9f}",
                            "sd": f"{result['sd']:.9f}",
                            "ci95_low": f"{result['ci95_t'][0]:.9f}",
                            "ci95_high": f"{result['ci95_t'][1]:.9f}",
                            "n_temporal_folds": result["n_folds"],
                        }
                    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-run", type=Path, required=True)
    parser.add_argument("--clip-run", type=Path, required=True)
    parser.add_argument("--alternating-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "pca_srrr_summary": args.pca_run.expanduser().resolve() / "summary.json",
        "clip_summary": args.clip_run.expanduser().resolve() / "summary.json",
        "alternating_summary": args.alternating_run.expanduser().resolve() / "summary.json",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    data = load_fold_values(
        args.pca_run.expanduser().resolve(),
        args.clip_run.expanduser().resolve(),
        args.alternating_run.expanduser().resolve(),
    )
    configure_plot()
    artifacts = []
    artifacts.extend(
        performance_figure(
            data,
            output,
            "mel80",
            "figure_01_common_mel80",
            "Общее восстановление MEL80: Fisher-усреднённый r",
        )
    )
    artifacts.extend(
        performance_figure(
            data,
            output,
            "low20",
            "figure_02_lower20_mel_bins",
            "Нижние 20 MEL-бинов: Fisher-усреднённый r",
        )
    )
    artifacts.extend(delta_figure(data, output))
    artifacts.append(write_table(data, output))
    manifest = {
        "schema_version": 1,
        "kind": "swpd_sub01_learned_bottleneck_figure_manifest",
        "source_summaries": {
            name: {"run_artifact": "external frozen summary.json", "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "artifacts": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
        "statistical_unit": "temporal fold within development subject sub-01",
        "interval": "descriptive two-sided 95% t interval across five temporal folds",
        "population_inference": False,
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"COMPLETE | {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
