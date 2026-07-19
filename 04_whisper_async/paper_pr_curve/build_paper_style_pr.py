#!/usr/bin/env python3
"""Build a Figure-13-style precision-recall plot from fixed async test results.

This script is deliberately read-only with respect to training runs.  It uses the
already saved dense test curves and never retrains, reselects, or overwrites a
checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_DIR = (
    SCRIPT_DIR.parent.parent
    / "artifacts"
    / "continuous_multiseed_summaries"
)
SUMMARY_GLOB = "continuous_multiseed_*.json"
SUMMARY_KIND = "continuous_head_training_multiseed_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the frozen L3+L4+L5 asynchronous test PR curves in the style "
            "of Figure 13 and mark the published recall=0.40, precision=0.60 point."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Explicit multiseed summary JSON. If omitted, use the newest valid "
            f"{SUMMARY_GLOB} from {DEFAULT_SUMMARY_DIR}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--system", default="L3+L4+L5")
    parser.add_argument("--target-recall", type=float, default=0.40)
    parser.add_argument("--paper-precision", type=float, default=0.60)
    parser.add_argument("--xmax", type=float, default=0.60)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_summary_path(requested: Path | None) -> Path:
    """Resolve an explicit summary or deterministically discover the latest one."""
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Multiseed summary not found: {path}")
        return path

    summary_dir = DEFAULT_SUMMARY_DIR.resolve()
    if not summary_dir.is_dir():
        raise FileNotFoundError(
            "Multiseed summary directory not found: "
            f"{summary_dir}. Run run_continuous_multiseed.ps1 first or pass --summary."
        )

    candidates: list[tuple[int, str, Path]] = []
    for path in summary_dir.glob(SUMMARY_GLOB):
        if not path.is_file():
            continue
        try:
            payload = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if payload.get("kind") != SUMMARY_KIND or not payload.get("runs"):
            continue
        stat = path.stat()
        candidates.append((stat.st_mtime_ns, path.name, path.resolve()))
    if not candidates:
        raise FileNotFoundError(
            f"No valid {SUMMARY_KIND!r} JSON matching {SUMMARY_GLOB!r} in {summary_dir}. "
            "Run run_continuous_multiseed.ps1 first or pass --summary."
        )
    return max(candidates)[2]


def finite_array(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def main() -> int:
    args = parse_args()
    summary_path = resolve_summary_path(args.summary)
    output_dir = args.output_dir.resolve()
    if args.summary is None:
        print(f"[summary] auto-discovered {summary_path}")
    if not 0.0 < args.target_recall < 1.0:
        raise ValueError("--target-recall must be between 0 and 1")
    if not 0.0 < args.paper_precision < 1.0:
        raise ValueError("--paper-precision must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(summary_path)
    run_rows = sorted(summary.get("runs", []), key=lambda row: int(row["seed"]))
    if not run_rows:
        raise ValueError(f"No run rows in {summary_path}")

    curves: list[dict[str, Any]] = []
    chance_curves: list[dict[str, np.ndarray]] = []
    for run in run_rows:
        seed = int(run["seed"])
        result_path = Path(run["result"]).resolve()
        if not result_path.is_file():
            raise FileNotFoundError(f"Seed {seed} result not found: {result_path}")
        result = load_json(result_path)
        try:
            system_result = result["splits"]["test"][args.system]
            raw_curve = system_result["curve"]
            operating_point = system_result["operating_point_from_val"]
        except KeyError as exc:
            raise KeyError(
                f"Seed {seed} result lacks test system {args.system!r}: {result_path}"
            ) from exc

        threshold = finite_array(raw_curve["threshold"], f"seed {seed} threshold")
        precision = finite_array(raw_curve["precision"], f"seed {seed} precision")
        recall = finite_array(raw_curve["recall"], f"seed {seed} recall")
        f1 = finite_array(raw_curve["f1"], f"seed {seed} f1")
        fp_per_min = finite_array(raw_curve["fp_per_min"], f"seed {seed} fp_per_min")
        lengths = {a.size for a in (threshold, precision, recall, f1, fp_per_min)}
        if len(lengths) != 1:
            raise ValueError(f"Seed {seed} curve arrays have different lengths")

        target_index = int(np.argmin(np.abs(recall - args.target_recall)))
        curves.append(
            {
                "seed": seed,
                "result_path": str(result_path),
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp_per_min": fp_per_min,
                "target_index": target_index,
                "operating_point": operating_point,
            }
        )

        chance = system_result.get("chance_label_null")
        if chance:
            chance_curves.append(
                {
                    "threshold": finite_array(
                        chance["threshold"], f"seed {seed} chance threshold"
                    ),
                    "precision": finite_array(
                        chance["precision_mean"], f"seed {seed} chance precision"
                    ),
                    "recall": finite_array(
                        chance["recall_mean"], f"seed {seed} chance recall"
                    ),
                }
            )

    curve_lengths = {row["threshold"].size for row in curves}
    if len(curve_lengths) != 1:
        raise ValueError("Dense curves do not share a common threshold-grid length")
    threshold_reference = curves[0]["threshold"]
    for row in curves[1:]:
        if not np.allclose(row["threshold"], threshold_reference, atol=1e-12, rtol=0.0):
            raise ValueError("Dense curves do not use the same threshold grid")

    precision_matrix = np.vstack([row["precision"] for row in curves])
    recall_matrix = np.vstack([row["recall"] for row in curves])
    mean_precision = precision_matrix.mean(axis=0)
    mean_recall = recall_matrix.mean(axis=0)
    sd_precision = precision_matrix.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_precision)
    mean_target_index = int(np.argmin(np.abs(mean_recall - args.target_recall)))

    target_rows = []
    for row in curves:
        idx = row["target_index"]
        target_rows.append(
            {
                "seed": row["seed"],
                "threshold": float(row["threshold"][idx]),
                "precision": float(row["precision"][idx]),
                "recall": float(row["recall"][idx]),
                "f1": float(row["f1"][idx]),
                "fp_per_min": float(row["fp_per_min"][idx]),
            }
        )

    target_precision_values = np.asarray(
        [row["precision"] for row in target_rows], dtype=np.float64
    )
    target_recall_values = np.asarray(
        [row["recall"] for row in target_rows], dtype=np.float64
    )
    target_precision_mean = float(target_precision_values.mean())
    target_precision_sd = float(target_precision_values.std(ddof=1)) if len(curves) > 1 else 0.0
    target_recall_mean = float(target_recall_values.mean())

    val_points = [row["operating_point"] for row in curves]
    val_precision_mean = float(np.mean([float(point["precision"]) for point in val_points]))
    val_recall_mean = float(np.mean([float(point["recall"]) for point in val_points]))

    colors = ["#1f77b4", "#9467bd", "#17becf", "#ff7f0e", "#8c564b"]
    fig, ax = plt.subplots(figsize=(8.6, 5.7), constrained_layout=True)
    for index, row in enumerate(curves):
        order = np.argsort(row["recall"], kind="stable")
        ax.plot(
            row["recall"][order],
            row["precision"][order],
            color=colors[index % len(colors)],
            linewidth=1.0,
            alpha=0.34,
            label="individual head-seed curves" if index == 0 else None,
        )

    mean_order = np.argsort(mean_recall, kind="stable")
    sorted_recall = mean_recall[mean_order]
    sorted_precision = mean_precision[mean_order]
    sorted_sd = sd_precision[mean_order]
    ax.fill_between(
        sorted_recall,
        np.clip(sorted_precision - sorted_sd, 0.0, 1.0),
        np.clip(sorted_precision + sorted_sd, 0.0, 1.0),
        color="#0057b8",
        alpha=0.13,
        linewidth=0,
        label="±1 SD across 5 head seeds",
    )
    ax.plot(
        sorted_recall,
        sorted_precision,
        color="#0057b8",
        linewidth=2.6,
        label="Whisper L3+L4+L5 mean PR",
        zorder=5,
    )

    if chance_curves:
        chance_lengths = {row["threshold"].size for row in chance_curves}
        if len(chance_lengths) == 1:
            chance_precision = np.vstack([row["precision"] for row in chance_curves]).mean(axis=0)
            chance_recall = np.vstack([row["recall"] for row in chance_curves]).mean(axis=0)
            chance_order = np.argsort(chance_recall, kind="stable")
            ax.plot(
                chance_recall[chance_order],
                chance_precision[chance_order],
                color="#2ca02c",
                linewidth=1.7,
                linestyle="--",
                label="label-null chance (50 permutations)",
            )

    ax.axvline(args.target_recall, color="#8892a0", linewidth=1.0, linestyle=":")
    ax.scatter(
        [args.target_recall],
        [args.paper_precision],
        marker="*",
        s=170,
        color="#d62728",
        edgecolor="white",
        linewidth=0.8,
        label="paper: R=0.40, P≈0.60",
        zorder=8,
    )
    ax.errorbar(
        [target_recall_mean],
        [target_precision_mean],
        yerr=[target_precision_sd],
        fmt="o",
        markersize=7,
        color="#003f88",
        ecolor="#003f88",
        elinewidth=1.5,
        capsize=4,
        label=(
            f"ours near R=0.40: P={target_precision_mean:.3f}"
            f"±{target_precision_sd:.3f} SD"
        ),
        zorder=9,
    )
    ax.scatter(
        [val_recall_mean],
        [val_precision_mean],
        marker="D",
        s=55,
        color="#7a3e9d",
        edgecolor="white",
        linewidth=0.7,
        label=(
            f"validation-selected operating point: "
            f"R={val_recall_mean:.3f}, P={val_precision_mean:.3f}"
        ),
        zorder=8,
    )

    ax.set_xlim(0.0, args.xmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Asynchronous decoder — test precision–recall")
    ax.text(
        0.01,
        0.02,
        "Frozen upstream seed 4; five continuous-head seeds. Test used only after validation selection.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#4f5965",
    )
    ax.grid(True, which="major", color="#d9dee5", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.2)

    stem = "paper_style_async_pr_multiseed"
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    fig.savefig(png_path, dpi=220, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "threshold",
                "recall",
                "precision",
                "f1",
                "fp_per_min",
            ],
        )
        writer.writeheader()
        for row in curves:
            for index in range(row["threshold"].size):
                writer.writerow(
                    {
                        "seed": row["seed"],
                        "threshold": f"{row['threshold'][index]:.8g}",
                        "recall": f"{row['recall'][index]:.8g}",
                        "precision": f"{row['precision'][index]:.8g}",
                        "f1": f"{row['f1'][index]:.8g}",
                        "fp_per_min": f"{row['fp_per_min'][index]:.8g}",
                    }
                )

    report = {
        "schema_version": 1,
        "kind": "paper_style_async_pr_multiseed",
        "created": datetime.now().astimezone().isoformat(),
        "source_summary": str(summary_path),
        "system": args.system,
        "source_seed": summary.get("source_seed"),
        "head_seeds": [row["seed"] for row in curves],
        "n_head_seeds": len(curves),
        "target_recall": args.target_recall,
        "paper_reference": {
            "recall": args.target_recall,
            "precision_approx": args.paper_precision,
            "f1_derived": (
                2.0 * args.target_recall * args.paper_precision
                / (args.target_recall + args.paper_precision)
            ),
        },
        "ours_nearest_target_per_seed": target_rows,
        "ours_near_target_aggregate": {
            "recall_mean": target_recall_mean,
            "precision_mean": target_precision_mean,
            "precision_sd": target_precision_sd,
            "precision_sem": target_precision_sd / math.sqrt(len(curves)),
        },
        "validation_selected_test_point_mean": {
            "recall": val_recall_mean,
            "precision": val_precision_mean,
        },
        "mean_dense_curve": {
            "threshold": threshold_reference.tolist(),
            "recall": mean_recall.tolist(),
            "precision": mean_precision.tolist(),
            "precision_sd": sd_precision.tolist(),
        },
        "outputs": {
            "png": str(png_path),
            "svg": str(svg_path),
            "csv": str(csv_path),
        },
        "notes": [
            "No model was trained and no checkpoint was modified.",
            "Thin lines are the five fixed test PR curves; the thick line averages points at identical thresholds.",
            "The article reports an approximate point (recall 0.40, precision 0.60), not a machine-readable curve.",
            "Chance is this project's 50-permutation label-null estimate and may not exactly match the article's chance implementation.",
        ],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=" * 78)
    print("PAPER-STYLE ASYNCHRONOUS PRECISION-RECALL CURVE")
    print("=" * 78)
    print(f"system: {args.system} | head seeds: {[row['seed'] for row in curves]}")
    print(
        f"paper reference: recall={args.target_recall:.3f}, "
        f"precision≈{args.paper_precision:.3f}"
    )
    print(
        f"ours near recall={args.target_recall:.3f}: "
        f"precision={target_precision_mean:.4f} ± {target_precision_sd:.4f} SD"
    )
    print(
        f"validation-selected test point (mean): "
        f"recall={val_recall_mean:.4f}, precision={val_precision_mean:.4f}"
    )
    print(f"PNG:  {png_path}")
    print(f"SVG:  {svg_path}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("[safety] read-only evaluation inputs; no training and no checkpoint changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
