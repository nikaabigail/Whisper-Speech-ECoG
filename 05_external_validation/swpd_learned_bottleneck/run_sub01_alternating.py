#!/usr/bin/env python3
"""Run constrained alternating 50D projectors on SWPD sub-01."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from alternating_core import AlternatingConfig, fit_alternating, predict_linear  # noqa: E402
from core import (  # noqa: E402
    BLOCK_COUNT, TARGETS, atomic_json, canonical_hash, component_metrics,
    fit_linear, fit_neural_reducer, fold_indexes, load_sub01_blocks, select, sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--maximum-iterations", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(BLOCK_COUNT)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(item not in range(BLOCK_COUNT) for item in args.folds):
        raise ValueError("folds must contain only 0..4")
    config = AlternatingConfig(
        maximum_iterations=args.maximum_iterations, patience=args.patience
    )
    config.validate()
    cache = args.cache_dir.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    if run == HERE or HERE in run.parents:
        raise ValueError("Run directory must be outside the source checkout")
    contract: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_alternating50_contract",
        "development_subject": "sub-01",
        "confirmatory_subjects_read": False,
        "cache_directory": str(cache),
        "cache_sha256": {f"block_{i:02d}.npz": sha256_file(cache / f"block_{i:02d}.npz") for i in range(5)},
        "targets": args.targets,
        "folds": args.folds,
        "config": asdict(config),
        "selection": "best iteration by validation common MEL80 Fisher correlation",
        "test_gate": "test selected only after projector and decoder are fixed",
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__)),
            "core": sha256_file(HERE / "alternating_core.py"),
        },
    }
    contract["fingerprint"] = canonical_hash(contract)
    run.mkdir(parents=True, exist_ok=True)
    contract_path = run / "run_contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Existing alternating run contract differs")
    else:
        atomic_json(contract_path, contract)
    blocks = load_sub01_blocks(cache)
    completed = []
    tasks = [f"fold_{fold:02d}/{target}" for fold in args.folds for target in args.targets]
    for fold in args.folds:
        train_indexes, validation_index, test_index = fold_indexes(fold)
        train_ids, _, train_neural = select(blocks, train_indexes)
        _, _, val_neural = select(blocks, (validation_index,))
        reducer = fit_neural_reducer(train_neural, 50, 42)
        train_x = reducer.transform(train_neural)
        val_x = reducer.transform(val_neural)
        _, _, train_mel = select(blocks, train_indexes, "mel80")
        _, _, val_mel = select(blocks, (validation_index,), "mel80")
        mel_scaler = StandardScaler().fit(train_mel)
        train_mel_z = mel_scaler.transform(train_mel)
        val_mel_z = mel_scaler.transform(val_mel)
        for target in args.targets:
            task = f"fold_{fold:02d}/{target}"
            task_dir = run / f"fold_{fold:02d}" / target
            result_path = task_dir / "result.json"
            atomic_json(run / "queue_state.json", {
                "schema_version": 1, "status": "running", "current_task": task,
                "completed_tasks": completed,
                "remaining_tasks": [item for item in tasks if item not in completed and item != task],
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }, overwrite=True)
            if result_path.is_file():
                print(f"[{task}] completed -> reuse", flush=True)
                completed.append(task)
                continue
            task_dir.mkdir(parents=True, exist_ok=True)
            _, _, train_raw = select(blocks, train_indexes, target)
            _, _, val_raw = select(blocks, (validation_index,), target)
            target_scaler = StandardScaler().fit(train_raw)
            train_y = target_scaler.transform(train_raw)
            val_y = target_scaler.transform(val_raw)
            print(f"[{task}] alternating | raw_dim={train_y.shape[1]}", flush=True)
            model, training = fit_alternating(
                train_x, train_y, val_x, val_y, train_mel_z, val_mel_z, config
            )
            # Held-out test is selected only after validation fixes the iteration.
            test_ids, _, test_neural = select(blocks, (test_index,))
            _, _, test_raw = select(blocks, (test_index,), target)
            _, _, test_mel = select(blocks, (test_index,), "mel80")
            test_x = reducer.transform(test_neural)
            test_y = target_scaler.transform(test_raw)
            test_mel_z = mel_scaler.transform(test_mel)
            truth_scores = test_y @ model["projection"]
            predicted_scores = predict_linear(test_x, model["decoder_coef"], model["decoder_intercept"])
            target_probe = fit_linear(train_y @ model["projection"], train_y)
            target_prediction = target_probe.predict(predicted_scores)
            mel_prediction = predict_linear(
                predicted_scores, model["mel_probe_coef"], model["mel_probe_intercept"]
            )
            result = {
                "schema_version": 1, "kind": "swpd_sub01_alternating50_task_result",
                "contract_fingerprint": contract["fingerprint"], "fold": fold, "target": target,
                "train_blocks": list(train_indexes), "validation_block": validation_index, "test_block": test_index,
                "training": training,
                "test_score50": component_metrics(truth_scores, predicted_scores),
                "test_target_standardized": component_metrics(test_y, target_prediction),
                "test_mel80_probe": component_metrics(test_mel_z, mel_prediction),
                "test_mel_low20_probe": component_metrics(test_mel_z[:, :20], mel_prediction[:, :20]),
                "projector_orthogonality_error": float(np.linalg.norm(model["projection"].T @ model["projection"] - np.eye(50), ord="fro")),
            }
            model_path = task_dir / "fixed_model_and_preprocessing.npz"
            np.savez_compressed(
                model_path, **model,
                target_scaler_mean=target_scaler.mean_, target_scaler_scale=target_scaler.scale_,
                mel_scaler_mean=mel_scaler.mean_, mel_scaler_scale=mel_scaler.scale_,
                neural_scaler_mean=reducer.scaler.mean_, neural_scaler_scale=reducer.scaler.scale_,
                neural_pca_mean=reducer.pca.mean_, neural_pca_components=reducer.pca.components_,
                neural_pca_explained_variance=reducer.pca.explained_variance_,
                target_probe_coef=target_probe.coef_, target_probe_intercept=target_probe.intercept_,
            )
            prediction_path = task_dir / "test_predictions.npz"
            np.savez_compressed(
                prediction_path, sample_ids=test_ids, truth_scores=truth_scores.astype(np.float32),
                predicted_scores=predicted_scores.astype(np.float32), truth_mel80=test_mel_z.astype(np.float32),
                predicted_mel80=mel_prediction.astype(np.float32),
            )
            result["model_sha256"] = sha256_file(model_path)
            result["predictions_sha256"] = sha256_file(prediction_path)
            result["fingerprint"] = canonical_hash(result)
            atomic_json(result_path, result)
            completed.append(task)
            print(f"[{task}] best_iter={training['best_iteration']} test mel80-r={result['test_mel80_probe']['fisher_z_component_correlation']:.4f}", flush=True)
    rows = [json.loads((run / f"fold_{fold:02d}" / target / "result.json").read_text(encoding="utf-8")) for fold in args.folds for target in args.targets]
    aggregate = {}
    for target in args.targets:
        selected = [row for row in rows if row["target"] == target]
        aggregate[target] = {}
        for name, section in (("score50_fisher_r", "test_score50"), ("target_standardized_fisher_r", "test_target_standardized"), ("mel80_fisher_r", "test_mel80_probe"), ("mel_low20_fisher_r", "test_mel_low20_probe")):
            values = np.asarray([row[section]["fisher_z_component_correlation"] for row in selected])
            aggregate[target][name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values)>1 else 0.0, "fold_count": len(values)}
    summary = {"schema_version": 1, "kind": "swpd_sub01_alternating50_summary", "created_utc": datetime.now(timezone.utc).isoformat(), "contract_fingerprint": contract["fingerprint"], "development_subject": "sub-01", "confirmatory_subjects_read": False, "aggregate_test": aggregate, "interpretation_guard": "Development-only; five temporal folds are not five patients."}
    atomic_json(run / "summary.json", summary, overwrite=True)
    atomic_json(run / "queue_state.json", {"schema_version": 1, "status": "completed", "completed_tasks": completed, "remaining_tasks": [], "summary": str(run / "summary.json"), "updated_utc": datetime.now(timezone.utc).isoformat()}, overwrite=True)
    print(f"COMPLETE | {run / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
