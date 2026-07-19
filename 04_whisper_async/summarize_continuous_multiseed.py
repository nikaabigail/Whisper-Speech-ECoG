#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and summarize continuous-head stochastic training seeds.

The statistical unit here is a *continuous-head training seed*.  Every run is
conditional on the same immutable upstream seed-4 ECoG encoders, synchronous
word heads, hidden cache, validation recording and held-out test recordings.
This script deliberately does not describe the result as a full-pipeline
multiseed experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = ROOT.parent / "artifacts" / "continuous_multiseed_runs"
SYSTEMS = ("L3", "L4", "L5", "L3+L4+L5")
PRIMARY_SYSTEM = "L3+L4+L5"
EXPECTED_SOURCE_SEED = 4
EXPECTED_TEST_EVENTS = 430


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate validation-selected continuous decoding results across "
            "continuous-head training seeds conditional on frozen upstream seed 4."
        )
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-prefix", default="continuous_multiseed_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 42))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except Exception as exc:
        raise RuntimeError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{id(value)}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def result_path_for_run(run: dict) -> Path:
    explicit = str(run.get("result") or "").strip()
    if explicit:
        return Path(explicit).resolve()
    directory = str(run.get("run_dir") or "").strip()
    if not directory:
        raise RuntimeError(f"Run entry has neither result nor run_dir: {run}")
    return (Path(directory) / "final_result.json").resolve()


def load_run_entries(args: argparse.Namespace) -> tuple[dict, List[dict], Path | None]:
    manifest_path = args.manifest.resolve() if args.manifest else None
    if manifest_path:
        manifest = read_json(manifest_path)
        entries = list(manifest.get("runs") or [])
        if not entries:
            raise RuntimeError(f"Manifest contains no runs: {manifest_path}")
    else:
        entries = []
        for seed in sorted(set(args.seeds)):
            run_name = f"{args.run_prefix}_seed{seed}"
            run_dir = args.runs_root.resolve() / run_name
            entries.append({
                "seed": seed,
                "source_seed": EXPECTED_SOURCE_SEED,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "result": str(run_dir / "final_result.json"),
            })
        manifest = {
            "schema_version": 1,
            "kind": "continuous_head_training_multiseed_ad_hoc",
            "group_name": args.run_prefix,
            "source_seed": EXPECTED_SOURCE_SEED,
            "summary_seeds": sorted(set(args.seeds)),
            "runs": entries,
        }

    seen: set[int] = set()
    normalized = []
    for item in entries:
        seed = int(item.get("seed", -1))
        if seed < 0:
            raise RuntimeError(f"Invalid run seed: {item}")
        if seed in seen:
            raise RuntimeError(f"Duplicate run entry for seed {seed}")
        seen.add(seed)
        current = dict(item)
        current["seed"] = seed
        current["result_path"] = str(result_path_for_run(current))
        normalized.append(current)

    expected = sorted(set(int(item) for item in (manifest.get("summary_seeds") or args.seeds)))
    missing_entries = sorted(set(expected) - seen)
    if missing_entries and not args.allow_partial:
        raise RuntimeError(f"Manifest is missing expected seed entries: {missing_entries}")
    return manifest, sorted(normalized, key=lambda item: item["seed"]), manifest_path


def require_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite: {number}")
    return number


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_signature(config: dict) -> list:
    signature = []
    models = list(config.get("source_models") or [])
    for model in sorted(models, key=lambda item: int(item["layer"])):
        source_seed = int(model.get("seed", -1))
        if source_seed != EXPECTED_SOURCE_SEED:
            raise RuntimeError(
                f"Frozen source model L{model.get('layer')} has seed {source_seed}, expected 4"
            )
        signature.append({
            "layer": int(model["layer"]),
            "model_name": model.get("model_name"),
            "seed": source_seed,
            "max_words_length": int(model.get("max_words_length", -1)),
            "regression_sha256": model.get("regression_sha256"),
            "synchronous_head_sha256": model.get("synchronous_head_sha256"),
            "synchronous_accuracy": model.get("synchronous_accuracy"),
        })
    if [item["layer"] for item in signature] != [3, 4, 5]:
        raise RuntimeError(f"Expected frozen L3/L4/L5 source models, got {signature}")
    return signature


def experiment_signature(config: dict, result: dict) -> dict:
    source_hashes = config.get("source_code_sha256") or {}
    return {
        "patient": config.get("patient"),
        "mode": config.get("mode"),
        "base_project_read_only": config.get("base_project_read_only"),
        "frozen_archive_read_only": config.get("frozen_archive_read_only"),
        "cache_root": config.get("cache_root"),
        "layout": config.get("layout"),
        "architecture": config.get("architecture"),
        "sampler": config.get("sampler"),
        "optimizer": config.get("optimizer"),
        "selection": config.get("selection"),
        "final_evaluation": config.get("final_evaluation"),
        "source_models": source_signature(config),
        # The trainer SHA differs for the preserved pre-generalization seed 4.
        # Shared scientific helper code must nevertheless remain identical.
        "continuous_common_sha256": source_hashes.get("continuous_common.py"),
        "async_replay_sha256": source_hashes.get("async_replay.py"),
        "result_protocol": result.get("protocol"),
    }


def validate_operating_point(point: dict, n_events: int, label: str) -> None:
    tp = int(point.get("tp", -1))
    fp = int(point.get("fp", -1))
    fn = int(point.get("fn", -1))
    if min(tp, fp, fn) < 0:
        raise RuntimeError(f"{label}: negative or missing TP/FP/FN")
    if tp + fn != n_events:
        raise RuntimeError(f"{label}: TP+FN={tp + fn}, expected {n_events}")
    precision = require_number(point.get("precision"), f"{label}.precision")
    recall = require_number(point.get("recall"), f"{label}.recall")
    f1 = require_number(point.get("f1"), f"{label}.f1")
    expected_precision = tp / (tp + fp) if tp + fp else 1.0
    expected_recall = tp / n_events if n_events else 0.0
    expected_f1 = (
        2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall else 0.0
    )
    if not close(precision, expected_precision):
        raise RuntimeError(f"{label}: precision is inconsistent with TP/FP")
    if not close(recall, expected_recall):
        raise RuntimeError(f"{label}: recall is inconsistent with TP/N")
    if not close(f1, expected_f1):
        raise RuntimeError(f"{label}: F1 is inconsistent with precision/recall")
    if not all(0.0 <= value <= 1.0 for value in (precision, recall, f1)):
        raise RuntimeError(f"{label}: P/R/F1 outside [0, 1]")


def extract_run(entry: dict) -> tuple[dict, dict, List[dict]]:
    result_path = Path(entry["result_path"])
    if not result_path.is_file():
        raise FileNotFoundError(f"Seed {entry['seed']} is incomplete; missing {result_path}")
    result = read_json(result_path)
    seed = int(entry["seed"])
    if int(result.get("seed", -1)) != seed:
        raise RuntimeError(
            f"Seed mismatch for {result_path}: manifest={seed}, result={result.get('seed')}"
        )
    source_seed = int(result.get("source_seed", EXPECTED_SOURCE_SEED))
    if source_seed != EXPECTED_SOURCE_SEED:
        raise RuntimeError(f"Seed {seed}: result source_seed={source_seed}, expected 4")
    if result.get("primary_system") != PRIMARY_SYSTEM:
        raise RuntimeError(f"Seed {seed}: unexpected primary system {result.get('primary_system')}")

    config_path = Path(str(result.get("run_config") or ""))
    if not config_path.is_file():
        fallback = result_path.parent / "run_config.json"
        if fallback.is_file():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Seed {seed}: run_config is missing: {config_path}")
    config = read_json(config_path)
    if int(config.get("seed", -1)) != seed:
        raise RuntimeError(f"Seed {seed}: run_config seed mismatch")
    if int(config.get("source_seed", EXPECTED_SOURCE_SEED)) != EXPECTED_SOURCE_SEED:
        raise RuntimeError(f"Seed {seed}: run_config source seed mismatch")
    if config.get("mode") != "production":
        raise RuntimeError(f"Seed {seed}: refusing non-production run mode={config.get('mode')}")

    completed = result.get("completed_heads") or {}
    for layer in (3, 4, 5):
        record = completed.get(str(layer)) or completed.get(layer)
        if not record:
            raise RuntimeError(f"Seed {seed}: missing completion record for L{layer}")
        if int(record.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed {seed}: L{layer} completion seed mismatch")
        checkpoint = Path(str(record.get("best_checkpoint") or ""))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Seed {seed}: selected L{layer} checkpoint is missing: {checkpoint}")

    splits = result.get("splits") or {}
    val = splits.get("val") or {}
    test = splits.get("test") or {}
    rows = []
    for system in SYSTEMS:
        if system not in val or system not in test:
            raise RuntimeError(f"Seed {seed}: missing {system} validation/test result")
        val_best = val[system].get("best_f1_posthoc") or {}
        test_system = test[system]
        point = test_system.get("operating_point_from_val") or {}
        n_events = int(test_system.get("n_ground_truth_events", -1))
        if n_events != EXPECTED_TEST_EVENTS:
            raise RuntimeError(
                f"Seed {seed} {system}: test events={n_events}, expected {EXPECTED_TEST_EVENTS}"
            )
        validate_operating_point(point, n_events, f"seed {seed} {system}")
        threshold = require_number(point.get("threshold"), f"seed {seed} {system}.threshold")
        val_threshold = require_number(
            val_best.get("threshold"), f"seed {seed} {system}.validation threshold"
        )
        if not close(threshold, val_threshold):
            raise RuntimeError(
                f"Seed {seed} {system}: test threshold {threshold} was not selected on validation {val_threshold}"
            )
        if point.get("selected_on") != "validation":
            raise RuntimeError(f"Seed {seed} {system}: operating point is not marked validation-selected")

        latency = point.get("latency") or {}
        row = {
            "seed": seed,
            "source_seed": source_seed,
            "system": system,
            "threshold": threshold,
            "precision": require_number(point.get("precision"), "precision"),
            "recall": require_number(point.get("recall"), "recall"),
            "f1": require_number(point.get("f1"), "f1"),
            "event_pr_auc_envelope": require_number(
                test_system.get("pr_auc_envelope"), "event PR-AUC envelope"
            ),
            "false_events_per_min": require_number(
                point.get("false_events_per_min"), "false events/min"
            ),
            "background_insertions_per_min": require_number(
                point.get("background_insertions_per_min"), "background insertions/min"
            ),
            "event_error_rate": require_number(point.get("event_error_rate"), "event error rate"),
            "latency_median_ms": require_number(latency.get("median_ms"), "median latency"),
            "tp": int(point["tp"]),
            "fp": int(point["fp"]),
            "fn": int(point["fn"]),
            "validation_f1": require_number(val_best.get("f1"), "validation F1"),
            "validation_pr_auc_envelope": require_number(
                val[system].get("pr_auc_envelope"), "validation PR-AUC envelope"
            ),
            "result": str(result_path),
        }
        rows.append(row)

    primary = result.get("primary_test_operating_point") or {}
    ensemble = next(item for item in rows if item["system"] == PRIMARY_SYSTEM)
    for key in ("threshold", "precision", "recall", "f1"):
        if not close(require_number(primary.get(key), f"primary.{key}"), float(ensemble[key])):
            raise RuntimeError(f"Seed {seed}: primary operating point differs from L3+L4+L5 ({key})")

    signature = experiment_signature(config, result)
    metadata = {
        "seed": seed,
        "source_seed": source_seed,
        "result": str(result_path),
        "run_config": str(config_path),
        "script_version": result.get("script_version"),
        "config_fingerprint": result.get("config_fingerprint"),
        "runtime_hotfixes": result.get("runtime_hotfixes") or [],
        "signature": signature,
    }
    return metadata, result, rows


T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def t_critical_975(df: int) -> float:
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    larger = sorted(key for key in T_CRITICAL_975 if key >= df)
    return T_CRITICAL_975[larger[0]] if larger else 1.960


def summarize_values(values: Sequence[float]) -> dict:
    clean = [require_number(item, "aggregate value") for item in values]
    n = len(clean)
    if n == 0:
        raise RuntimeError("Cannot summarize an empty vector")
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    half = t_critical_975(n - 1) * sem if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "sem": sem,
        "ci95_t": [mean - half, mean + half],
        "min": min(clean),
        "max": max(clean),
        "values": clean,
    }


def exact_two_sided_sign_p(differences: Sequence[float]) -> dict:
    nonzero = [item for item in differences if abs(item) > 1e-12]
    wins = sum(item > 0 for item in nonzero)
    losses = sum(item < 0 for item in nonzero)
    n = len(nonzero)
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {"wins": wins, "losses": losses, "ties": len(differences) - n, "p_two_sided": p_value}


def aggregate_rows(rows: List[dict]) -> tuple[dict, dict]:
    metric_names = (
        "threshold", "precision", "recall", "f1", "event_pr_auc_envelope",
        "false_events_per_min", "background_insertions_per_min",
        "event_error_rate", "latency_median_ms",
    )
    aggregate: Dict[str, dict] = {}
    for system in SYSTEMS:
        current = [row for row in rows if row["system"] == system]
        current.sort(key=lambda row: row["seed"])
        aggregate[system] = {
            metric: summarize_values([float(row[metric]) for row in current])
            for metric in metric_names
        }

    comparisons: Dict[str, dict] = {}
    by_seed = {
        seed: {row["system"]: row for row in rows if row["seed"] == seed}
        for seed in sorted(set(row["seed"] for row in rows))
    }
    for layer in ("L3", "L4", "L5"):
        differences = [
            by_seed[seed][PRIMARY_SYSTEM]["f1"] - by_seed[seed][layer]["f1"]
            for seed in by_seed
        ]
        comparisons[f"{PRIMARY_SYSTEM}_minus_{layer}"] = {
            "metric": "validation-selected test F1",
            "difference": summarize_values(differences),
            "exact_sign_test": exact_two_sided_sign_p(differences),
        }

    best_single_differences = []
    selected = []
    for seed, systems in by_seed.items():
        best_layer = max(("L3", "L4", "L5"), key=lambda name: systems[name]["validation_f1"])
        difference = systems[PRIMARY_SYSTEM]["f1"] - systems[best_layer]["f1"]
        best_single_differences.append(difference)
        selected.append({
            "seed": seed,
            "selected_layer_on_validation": best_layer,
            "selected_layer_test_f1": systems[best_layer]["f1"],
            "ensemble_test_f1": systems[PRIMARY_SYSTEM]["f1"],
            "difference": difference,
        })
    comparisons["ensemble_minus_validation_selected_single"] = {
        "selection_rule": "choose L3/L4/L5 by validation F1, then compare test F1",
        "per_seed": selected,
        "difference": summarize_values(best_single_differences),
        "exact_sign_test": exact_two_sided_sign_p(best_single_differences),
    }
    return aggregate, comparisons


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "source_seed", "system", "threshold", "precision", "recall", "f1",
        "event_pr_auc_envelope", "false_events_per_min",
        "background_insertions_per_min", "event_error_rate", "latency_median_ms",
        "tp", "fp", "fn", "validation_f1", "validation_pr_auc_envelope", "result",
    ]
    temporary = path.with_name(f"{path.name}.partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])
    temporary.replace(path)


def print_summary(rows: List[dict], aggregate: dict) -> None:
    ensemble = sorted(
        (row for row in rows if row["system"] == PRIMARY_SYSTEM),
        key=lambda row: row["seed"],
    )
    print("=" * 86)
    print("CONTINUOUS-HEAD MULTISEED | conditional on frozen upstream encoder/head seed 4")
    print("Primary: L3+L4+L5 test operating point; threshold selected only on validation")
    print("=" * 86)
    print(f"{'seed':>6} {'P':>8} {'R':>8} {'F1':>8} {'PR-AUC':>9} {'FP/min':>9} {'theta':>8}")
    for row in ensemble:
        print(
            f"{row['seed']:>6d} {row['precision']:>8.3f} {row['recall']:>8.3f} "
            f"{row['f1']:>8.3f} {row['event_pr_auc_envelope']:>9.3f} "
            f"{row['false_events_per_min']:>9.2f} {row['threshold']:>8.3f}"
        )
    primary = aggregate[PRIMARY_SYSTEM]
    print("-" * 86)
    for metric, label in (("f1", "F1"), ("event_pr_auc_envelope", "event PR-AUC")):
        stats = primary[metric]
        print(
            f"{label}: mean={stats['mean']:.4f}, SD={stats['sample_sd']:.4f}, "
            f"SEM={stats['sem']:.4f}, 95% t-CI=[{stats['ci95_t'][0]:.4f}, "
            f"{stats['ci95_t'][1]:.4f}], n={stats['n']}"
        )


def main() -> int:
    args = parse_args()
    manifest, entries, manifest_path = load_run_entries(args)
    metadata = []
    all_rows: List[dict] = []
    missing = []
    reference_signature = None
    reference_seed = None
    for entry in entries:
        try:
            run_metadata, _result, rows = extract_run(entry)
        except (FileNotFoundError, RuntimeError) as exc:
            if args.allow_partial:
                missing.append({"seed": entry["seed"], "error": str(exc)})
                continue
            raise
        signature = canonical(run_metadata.pop("signature"))
        if reference_signature is None:
            reference_signature = signature
            reference_seed = entry["seed"]
        elif signature != reference_signature:
            raise RuntimeError(
                f"Scientific configuration mismatch between seed {reference_seed} and seed {entry['seed']}"
            )
        metadata.append(run_metadata)
        all_rows.extend(rows)

    completed_seeds = sorted(set(row["seed"] for row in all_rows))
    if not completed_seeds:
        raise RuntimeError("No completed production seeds were found")
    if len(completed_seeds) < 2 and not args.allow_partial:
        raise RuntimeError("At least two completed seeds are required for multiseed statistics")

    aggregate, comparisons = aggregate_rows(all_rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name = str(manifest.get("group_name") or args.run_prefix)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir else
        ((manifest_path.parent / "summaries") if manifest_path else (ROOT / "results" / "summaries"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"continuous_multiseed_{group_name}_{stamp}.json"
    csv_path = output_dir / f"continuous_multiseed_{group_name}_{stamp}.csv"

    payload = {
        "schema_version": 1,
        "kind": "continuous_head_training_multiseed_summary",
        "created": datetime.now().isoformat(),
        "group_name": group_name,
        "interpretation": (
            "stochastic continuous-head training seeds conditional on the same frozen "
            "upstream seed-4 encoders, synchronous heads, cache and data split"
        ),
        "not_full_pipeline_multiseed": True,
        "source_seed": EXPECTED_SOURCE_SEED,
        "completed_training_seeds": completed_seeds,
        "n_seeds": len(completed_seeds),
        "primary_system": PRIMARY_SYSTEM,
        "primary_metric_note": "test operating point uses a threshold selected only on validation",
        "validation": {
            "scientific_configuration_identical": True,
            "expected_test_events_per_seed": EXPECTED_TEST_EVENTS,
            "missing_or_invalid_runs": missing,
        },
        "runs": metadata,
        "per_seed_system_rows": all_rows,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
    }
    atomic_json(json_path, payload)
    write_csv(csv_path, all_rows)

    if manifest_path:
        latest = read_json(manifest_path)
        latest["summary"] = {
            "created": payload["created"],
            "json": str(json_path),
            "csv": str(csv_path),
            "completed_training_seeds": completed_seeds,
            "n_seeds": len(completed_seeds),
        }
        latest["updated"] = datetime.now().isoformat()
        atomic_json(manifest_path, latest)

    print_summary(all_rows, aggregate)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
