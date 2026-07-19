#!/usr/bin/env python3
"""Summarize the fixed test PR curves at the point nearest recall=0.40.

The script is read-only with respect to training outputs.  It also reconstructs
event matching from saved test timelines so latency is reported at the same
threshold as the requested PR-curve slice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_DIR = (
    SCRIPT_DIR.parent.parent
    / "artifacts"
    / "continuous_multiseed_summaries"
)
SUMMARY_GLOB = "continuous_multiseed_*.json"
SUMMARY_KIND = "continuous_head_training_multiseed_summary"
SYSTEMS = ("L3", "L4", "L5", "L3+L4+L5")


@dataclass(frozen=True)
class Candidate:
    file_index: int
    time_s: float
    class_index: int
    score: float
    gt_event_index: int | None
    gt_class_index: int | None
    gt_start_s: float | None


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Explicit multiseed summary JSON. If omitted, use the newest valid "
            f"{SUMMARY_GLOB} from {DEFAULT_SUMMARY_DIR}."
        ),
    )
    parser.add_argument("--target-recall", type=float, default=0.40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
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
            payload = read_json(path)
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


def file_index_from_path(path: Path) -> int:
    match = re.search(r"file_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot recover file index from {path}")
    return int(match.group(1))


def candidates_from_timeline(path: Path, system: str) -> tuple[list[Candidate], int]:
    file_index = file_index_from_path(path)
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times_s"], dtype=np.float64)
        starts = np.asarray(data["gt_start_s"], dtype=np.float64)
        ends = np.asarray(data["gt_end_s"], dtype=np.float64)
        classes = np.asarray(data["gt_class"], dtype=np.int64)
        smoothed = np.asarray(data[f"smooth_{system}"], dtype=np.float64)
    if smoothed.shape != (len(times), 27):
        raise ValueError(f"Unexpected probability shape in {path}: {smoothed.shape}")
    if not (len(starts) == len(ends) == len(classes)):
        raise ValueError(f"Ground-truth arrays do not align in {path}")

    winners = smoothed.argmax(axis=1)
    result: list[Candidate] = []
    for class_index in range(1, smoothed.shape[1]):
        peaks, _ = find_peaks(smoothed[:, class_index])
        for peak in peaks:
            if int(winners[peak]) != class_index:
                continue
            time_s = float(times[peak])
            gt_index = int(np.searchsorted(starts, time_s, side="right") - 1)
            if gt_index >= 0 and time_s <= float(ends[gt_index]):
                event_index: int | None = gt_index
                gt_class: int | None = int(classes[gt_index])
                gt_start: float | None = float(starts[gt_index])
            else:
                event_index = None
                gt_class = None
                gt_start = None
            result.append(
                Candidate(
                    file_index=file_index,
                    time_s=time_s,
                    class_index=class_index,
                    score=float(smoothed[peak, class_index]),
                    gt_event_index=event_index,
                    gt_class_index=gt_class,
                    gt_start_s=gt_start,
                )
            )
    return result, len(starts)


def score_timeline_paths(paths: list[Path], system: str, threshold: float) -> dict[str, Any]:
    candidates: list[Candidate] = []
    n_events = 0
    for path in paths:
        current, current_events = candidates_from_timeline(path, system)
        candidates.extend(current)
        n_events += current_events
    candidates.sort(key=lambda item: (item.file_index, item.time_s, -item.score))

    matched: set[tuple[int, int]] = set()
    latencies_ms: list[float] = []
    tp = fp = substitutions = duplicates = insertions = 0
    for candidate in candidates:
        if candidate.score < threshold:
            continue
        if candidate.gt_event_index is None:
            fp += 1
            insertions += 1
            continue
        if candidate.gt_class_index != candidate.class_index:
            fp += 1
            substitutions += 1
            continue
        key = (candidate.file_index, candidate.gt_event_index)
        if key in matched:
            fp += 1
            duplicates += 1
            continue
        matched.add(key)
        tp += 1
        assert candidate.gt_start_s is not None
        latencies_ms.append((candidate.time_s - candidate.gt_start_s) * 1000.0)

    recall = tp / n_events
    precision = tp / (tp + fp) if tp + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latency = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "tp": tp,
        "fp": fp,
        "fn": n_events - tp,
        "substitutions": substitutions,
        "duplicate_detections": duplicates,
        "background_insertions": insertions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_n": len(latency),
        "latency_median_ms": float(np.median(latency)) if len(latency) else float("nan"),
        "latency_q25_ms": float(np.quantile(latency, 0.25)) if len(latency) else float("nan"),
        "latency_q75_ms": float(np.quantile(latency, 0.75)) if len(latency) else float("nan"),
        "latency_p90_ms": float(np.quantile(latency, 0.90)) if len(latency) else float("nan"),
    }


def stats(values: list[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    n = len(array)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95": [mean - critical * sem, mean + critical * sem],
    }


def main() -> int:
    args = args_parser()
    summary_path = resolve_summary_path(args.summary)
    output_dir = args.output_dir.resolve()
    if args.summary is None:
        print(f"[summary] auto-discovered {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = read_json(summary_path)
    rows: list[dict[str, Any]] = []

    for run in sorted(source["runs"], key=lambda item: int(item["seed"])):
        seed = int(run["seed"])
        result_path = Path(run["result"]).resolve()
        result = read_json(result_path)
        timeline_paths = [Path(item).resolve() for item in result["timelines"]["test"]]
        for timeline in timeline_paths:
            if not timeline.is_file():
                raise FileNotFoundError(timeline)
        for system in SYSTEMS:
            system_result = result["splits"]["test"][system]
            curve = system_result["curve"]
            recalls = np.asarray(curve["recall"], dtype=np.float64)
            index = int(np.argmin(np.abs(recalls - args.target_recall)))
            threshold = float(curve["threshold"][index])
            reconstructed = score_timeline_paths(timeline_paths, system, threshold)
            expected = {
                "recall": float(curve["recall"][index]),
                "precision": float(curve["precision"][index]),
                "f1": float(curve["f1"][index]),
            }
            for metric, expected_value in expected.items():
                if not math.isclose(
                    float(reconstructed[metric]), expected_value, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"Seed {seed} {system} reconstruction mismatch for {metric}: "
                        f"{reconstructed[metric]} != {expected_value}"
                    )
            rows.append(
                {
                    "seed": seed,
                    "system": system,
                    "threshold": threshold,
                    "recall": expected["recall"],
                    "precision": expected["precision"],
                    "f1": expected["f1"],
                    "fp_per_min": float(curve["fp_per_min"][index]),
                    "pr_auc": float(system_result["pr_auc_envelope"]),
                    **reconstructed,
                    "result_path": str(result_path),
                }
            )

    aggregates: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        aggregates[system] = {
            metric: stats([float(row[metric]) for row in system_rows])
            for metric in (
                "threshold",
                "recall",
                "precision",
                "f1",
                "fp_per_min",
                "pr_auc",
                "latency_median_ms",
                "latency_q25_ms",
                "latency_q75_ms",
            )
        }

    report = {
        "schema_version": 1,
        "kind": "test_pr_slice_nearest_target_recall",
        "created": datetime.now().astimezone().isoformat(),
        "source_summary": str(summary_path),
        "target_recall": args.target_recall,
        "selection": (
            "For each head seed and system, select the saved dense test-curve point "
            "whose event recall is nearest target_recall."
        ),
        "interpretation": (
            "Descriptive paper-style test PR-curve slice; thresholds use test labels to "
            "locate recall and are not validation-fixed deployment thresholds."
        ),
        "conditional_on_frozen_upstream_seed": source.get("source_seed"),
        "rows": rows,
        "aggregate": aggregates,
    }
    json_path = output_dir / "async_recall40_multiseed.json"
    csv_path = output_dir / "async_recall40_multiseed.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 96)
    print("TEST PR SLICE NEAREST RECALL=0.40 | five continuous-head seeds")
    print("=" * 96)
    for row in rows:
        if row["system"] == "L3+L4+L5":
            print(
                f"seed {row['seed']:>2}: theta={row['threshold']:.3f} "
                f"R={row['recall']:.4f} P={row['precision']:.4f} F1={row['f1']:.4f} "
                f"FP/min={row['fp_per_min']:.2f} latency50={row['latency_median_ms']:.1f} ms"
            )
    print("-" * 96)
    for system in SYSTEMS:
        aggregate = aggregates[system]
        print(
            f"{system:>8}: R={aggregate['recall']['mean']:.4f}±{aggregate['recall']['sd']:.4f} "
            f"P={aggregate['precision']['mean']:.4f}±{aggregate['precision']['sd']:.4f} "
            f"F1={aggregate['f1']['mean']:.4f}±{aggregate['f1']['sd']:.4f} "
            f"FP/min={aggregate['fp_per_min']['mean']:.2f}±{aggregate['fp_per_min']['sd']:.2f} "
            f"latency50={aggregate['latency_median_ms']['mean']:.1f}±"
            f"{aggregate['latency_median_ms']['sd']:.1f} ms"
        )
    ensemble = aggregates["L3+L4+L5"]
    print("-" * 96)
    print(
        "Ensemble F1 95% t-CI: "
        f"[{ensemble['f1']['ci95'][0]:.4f}, {ensemble['f1']['ci95'][1]:.4f}]"
    )
    print(
        "Ensemble precision 95% t-CI: "
        f"[{ensemble['precision']['ci95'][0]:.4f}, {ensemble['precision']['ci95'][1]:.4f}]"
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("[safety] saved timelines/results read only; no model training or checkpoint changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
