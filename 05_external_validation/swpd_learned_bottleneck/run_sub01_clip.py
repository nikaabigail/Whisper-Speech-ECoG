#!/usr/bin/env python3
"""Train development-only CLIP50 projectors on SWPD sub-01."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from clip_core import ClipConfig, LinearClip, embed, embedding_diagnostics, train_clip  # noqa: E402
from core import (  # noqa: E402
    BLOCK_COUNT,
    TARGETS,
    atomic_json,
    canonical_hash,
    component_metrics,
    fit_linear,
    fit_neural_reducer,
    fold_indexes,
    load_sub01_blocks,
    select,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--maximum-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(BLOCK_COUNT)))
    return parser.parse_args()


def _contract(args: argparse.Namespace, cache: Path, config: ClipConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_clip50_contract",
        "development_subject": "sub-01",
        "confirmatory_subjects_read": False,
        "cache_directory": str(cache),
        "cache_sha256": {
            f"block_{i:02d}.npz": sha256_file(cache / f"block_{i:02d}.npz")
            for i in range(BLOCK_COUNT)
        },
        "targets": args.targets,
        "folds": args.folds,
        "device": args.device,
        "clip_config": asdict(config),
        "negative_sampling": "within each block, examples separated by at least 0.5 s",
        "selection": "best epoch by validation symmetric InfoNCE; test opened after fixed model",
        "fit_scope": "all preprocessing, initialization, CLIP and diagnostic probes train-only",
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__)),
            "clip_core": sha256_file(HERE / "clip_core.py"),
            "shared_core": sha256_file(HERE / "core.py"),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def _initial_model(train_x: np.ndarray, train_y: np.ndarray, dimension: int, seed: int) -> LinearClip:
    pca = PCA(n_components=dimension, whiten=False, svd_solver="full", random_state=seed).fit(train_y)
    projection = pca.components_.T
    target_scores = train_y @ projection
    decoder = LinearRegression(n_jobs=1).fit(train_x, target_scores)
    return LinearClip(
        train_x.shape[1],
        train_y.shape[1],
        dimension,
        target_initial_projection=projection,
        neural_initial_weight=decoder.coef_.T,
        neural_initial_bias=decoder.intercept_,
    )


def _task_fingerprint(
    contract: dict[str, Any], fold: int, target: str, train_ids: np.ndarray, val_ids: np.ndarray
) -> str:
    return canonical_hash(
        {
            "contract": contract["fingerprint"],
            "fold": fold,
            "target": target,
            "train_ids": canonical_hash(train_ids.tolist()),
            "validation_ids": canonical_hash(val_ids.tolist()),
        }
    )


def _save_task(
    directory: Path,
    model: LinearClip,
    target_scaler: StandardScaler,
    training: dict[str, Any],
    result: dict[str, Any],
    predictions: dict[str, np.ndarray],
) -> None:
    model_path = directory / "fixed_model.npz"
    np.savez_compressed(
        model_path,
        neural_weight=model.neural_weight.detach().cpu().numpy(),
        neural_bias=model.neural_bias.detach().cpu().numpy(),
        target_projection=model.target_projection.detach().cpu().numpy(),
        target_scaler_mean=target_scaler.mean_,
        target_scaler_scale=target_scaler.scale_,
    )
    prediction_path = directory / "test_predictions.npz"
    np.savez_compressed(prediction_path, **predictions)
    payload = {
        **result,
        "training": training,
        "model_sha256": sha256_file(model_path),
        "predictions_sha256": sha256_file(prediction_path),
    }
    payload["fingerprint"] = canonical_hash(payload)
    atomic_json(directory / "result.json", payload)


def _load_task_result(path: Path, expected_task_fingerprint: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.pop("fingerprint", None)
    if stored != canonical_hash(payload):
        raise RuntimeError(f"CLIP task receipt was modified: {path}")
    if expected_task_fingerprint is not None and payload.get("task_fingerprint") != expected_task_fingerprint:
        raise RuntimeError(f"CLIP task fingerprint mismatch: {path}")
    expected = {
        "fixed_model.npz": payload["model_sha256"],
        "test_predictions.npz": payload["predictions_sha256"],
    }
    for filename, checksum in expected.items():
        artifact = path.parent / filename
        if not artifact.is_file() or sha256_file(artifact) != checksum:
            raise RuntimeError(f"CLIP task artifact mismatch: {artifact}")
    payload["fingerprint"] = stored
    return payload


def main() -> int:
    args = parse_args()
    if any(fold not in range(BLOCK_COUNT) for fold in args.folds):
        raise ValueError("folds must contain only 0..4")
    config = ClipConfig(
        seed=args.seed,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    config.validate()
    cache = args.cache_dir.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    if run == HERE or HERE in run.parents:
        raise ValueError("Run directory must be outside the source checkout")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    contract = _contract(args, cache, config)
    run.mkdir(parents=True, exist_ok=True)
    contract_path = run / "run_contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Existing CLIP run contract differs; choose a new run directory")
    else:
        atomic_json(contract_path, contract)
    blocks = load_sub01_blocks(cache)
    completed: list[str] = []
    tasks = [f"fold_{fold:02d}/{target}" for fold in args.folds for target in args.targets]
    for fold in args.folds:
        train_indexes, validation_index, test_index = fold_indexes(fold)
        train_ids, train_times, train_neural = select(blocks, train_indexes)
        val_ids, val_times, val_neural = select(blocks, (validation_index,))
        neural_reducer = fit_neural_reducer(train_neural, config.dimension, seed=42)
        train_x = neural_reducer.transform(train_neural)
        val_x = neural_reducer.transform(val_neural)
        _, _, train_mel = select(blocks, train_indexes, "mel80")
        mel_scaler = StandardScaler().fit(train_mel)
        train_mel_z = mel_scaler.transform(train_mel)
        for target in args.targets:
            task_name = f"fold_{fold:02d}/{target}"
            task_dir = run / f"fold_{fold:02d}" / target
            result_path = task_dir / "result.json"
            atomic_json(
                run / "queue_state.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "current_task": task_name,
                    "completed_tasks": completed,
                    "remaining_tasks": [item for item in tasks if item not in completed and item != task_name],
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                overwrite=True,
            )
            if result_path.is_file():
                expected_fingerprint = _task_fingerprint(contract, fold, target, train_ids, val_ids)
                _load_task_result(result_path, expected_fingerprint)
                print(f"[{task_name}] validated completion -> reuse", flush=True)
                completed.append(task_name)
                continue
            task_dir.mkdir(parents=True, exist_ok=True)
            _, _, train_raw = select(blocks, train_indexes, target)
            _, _, val_raw = select(blocks, (validation_index,), target)
            target_scaler = StandardScaler().fit(train_raw)
            train_y = target_scaler.transform(train_raw)
            val_y = target_scaler.transform(val_raw)
            fingerprint = _task_fingerprint(contract, fold, target, train_ids, val_ids)
            model = _initial_model(train_x, train_y, config.dimension, args.seed)
            print(f"[{task_name}] CLIP50 train | raw_dim={train_y.shape[1]}", flush=True)
            model, training = train_clip(
                model,
                train_x,
                train_y,
                train_ids.tolist(),
                train_times,
                val_x,
                val_y,
                val_ids.tolist(),
                val_times,
                config,
                device,
                task_dir / "checkpoint.pt",
                fingerprint,
            )
            # Test is selected only after validation has fixed the model.
            test_ids, _, test_neural = select(blocks, (test_index,))
            _, _, test_raw = select(blocks, (test_index,), target)
            _, _, test_mel = select(blocks, (test_index,), "mel80")
            test_x = neural_reducer.transform(test_neural)
            test_y = target_scaler.transform(test_raw)
            test_mel_z = mel_scaler.transform(test_mel)
            train_pred_embed, train_target_embed = embed(model, train_x, train_y, device)
            test_pred_embed, test_target_embed = embed(model, test_x, test_y, device)
            target_probe = fit_linear(train_target_embed, train_y)
            mel_probe = fit_linear(train_target_embed, train_mel_z)
            test_target_prediction = target_probe.predict(test_pred_embed)
            test_mel_prediction = mel_probe.predict(test_pred_embed)
            result = {
                "schema_version": 1,
                "kind": "swpd_sub01_clip50_task_result",
                "task_fingerprint": fingerprint,
                "fold": fold,
                "target": target,
                "train_blocks": list(train_indexes),
                "validation_block": validation_index,
                "test_block": test_index,
                "test_score50": component_metrics(test_target_embed, test_pred_embed),
                "test_target_standardized": component_metrics(test_y, test_target_prediction),
                "test_mel80_probe": component_metrics(test_mel_z, test_mel_prediction),
                "test_mel_low20_probe": component_metrics(
                    test_mel_z[:, :20], test_mel_prediction[:, :20]
                ),
                "embedding_diagnostics": embedding_diagnostics(test_pred_embed, test_target_embed),
                "target_projector_orthogonality_error": float(
                    np.linalg.norm(
                        model.target_projection.detach().cpu().numpy().T
                        @ model.target_projection.detach().cpu().numpy()
                        - np.eye(config.dimension),
                        ord="fro",
                    )
                ),
            }
            _save_task(
                task_dir,
                model,
                target_scaler,
                training,
                result,
                {
                    "sample_ids": test_ids,
                    "truth_embedding": test_target_embed.astype(np.float32),
                    "predicted_embedding": test_pred_embed.astype(np.float32),
                    "truth_target_standardized": test_y.astype(np.float32),
                    "predicted_target_standardized": test_target_prediction.astype(np.float32),
                    "truth_mel80_standardized": test_mel_z.astype(np.float32),
                    "predicted_mel80_standardized": test_mel_prediction.astype(np.float32),
                },
            )
            completed.append(task_name)
            print(
                f"[{task_name}] test mel80-r="
                f"{result['test_mel80_probe']['fisher_z_component_correlation']:.4f} "
                f"low20-r={result['test_mel_low20_probe']['fisher_z_component_correlation']:.4f}",
                flush=True,
            )
    rows = []
    for fold in args.folds:
        for target in args.targets:
            payload = _load_task_result(run / f"fold_{fold:02d}" / target / "result.json")
            rows.append(payload)
    aggregate = {}
    for target in args.targets:
        selected = [row for row in rows if row["target"] == target]
        target_result = {}
        for label, section in (
            ("score50_fisher_r", "test_score50"),
            ("target_standardized_fisher_r", "test_target_standardized"),
            ("mel80_fisher_r", "test_mel80_probe"),
            ("mel_low20_fisher_r", "test_mel_low20_probe"),
        ):
            values = np.asarray(
                [row[section]["fisher_z_component_correlation"] for row in selected], dtype=float
            )
            target_result[label] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "fold_count": len(values),
            }
        aggregate[target] = target_result
    summary = {
        "schema_version": 1,
        "kind": "swpd_sub01_clip50_summary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_fingerprint": contract["fingerprint"],
        "development_subject": "sub-01",
        "confirmatory_subjects_read": False,
        "aggregate_test": aggregate,
        "interpretation_guard": "Development-only; five temporal folds are not five patients.",
    }
    atomic_json(run / "summary.json", summary, overwrite=True)
    atomic_json(
        run / "queue_state.json",
        {
            "schema_version": 1,
            "status": "completed",
            "completed_tasks": completed,
            "remaining_tasks": [],
            "summary": str(run / "summary.json"),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
        overwrite=True,
    )
    print(f"COMPLETE | {run / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FloatingPointError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
