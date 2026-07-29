#!/usr/bin/env python3
"""Run leakage-controlled PCA50 and supervised RRR50 on SWPD sub-01 caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core import (  # noqa: E402
    BLOCK_COUNT,
    TARGETS,
    aggregate_fold_metrics,
    atomic_json,
    canonical_hash,
    component_metrics,
    fit_linear,
    fit_neural_reducer,
    fit_pca_bottleneck,
    fit_supervised_rrr_bottleneck,
    fold_indexes,
    load_sub01_blocks,
    save_fold_artifacts,
    select,
    sha256_file,
)


METHODS = ("pca50", "srrr50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(BLOCK_COUNT)))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _contract(args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    cache_receipts = {}
    for index in range(BLOCK_COUNT):
        manifest = cache_dir / f"block_{index:02d}.json"
        arrays = cache_dir / f"block_{index:02d}.npz"
        cache_receipts[str(index)] = {
            "manifest_sha256": sha256_file(manifest),
            "arrays_sha256": sha256_file(arrays),
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_learned_bottleneck_contract",
        "development_subject": "sub-01",
        "confirmatory_subjects_read": False,
        "cache_directory": str(cache_dir),
        "cache_receipts": cache_receipts,
        "dimension": args.dimension,
        "seed": args.seed,
        "methods": args.methods,
        "targets": args.targets,
        "folds": args.folds,
        "split": "test=i; validation=(i+1)%5; train=remaining three blocks",
        "fit_scope": "all scalers, projectors, decoders and MEL probes fit on train blocks only",
        "metric_scope": "all frames",
        "mel_probe": "train-only linear map from true target bottleneck to standardized MEL80",
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__)),
            "core": sha256_file(HERE / "core.py"),
        },
        "python": platform.python_version(),
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def _save_neural_reducer(path: Path, reducer: Any) -> None:
    np.savez_compressed(
        path,
        scaler_mean=reducer.scaler.mean_,
        scaler_scale=reducer.scaler.scale_,
        pca_mean=reducer.pca.mean_,
        pca_components=reducer.pca.components_,
        pca_explained_variance=reducer.pca.explained_variance_,
    )


def _load_completed_fold(path: Path, contract_fingerprint: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.pop("fingerprint", None)
    if stored != canonical_hash(payload):
        raise RuntimeError(f"Completed fold receipt was modified: {path}")
    if payload.get("contract_fingerprint") != contract_fingerprint:
        raise RuntimeError(f"Completed fold contract mismatch: {path}")
    for label, result in payload.get("results", {}).items():
        artifact_directory = path.parent / label
        expected = {
            "model.npz": result["model_sha256"],
            "test_predictions.npz": result["predictions_sha256"],
        }
        for filename, checksum in expected.items():
            artifact = artifact_directory / filename
            if not artifact.is_file() or sha256_file(artifact) != checksum:
                raise RuntimeError(f"Completed fold artifact mismatch: {artifact}")
    payload["fingerprint"] = stored
    return payload


def main() -> int:
    args = parse_args()
    if args.dimension <= 0:
        raise ValueError("dimension must be positive")
    if any(fold not in range(BLOCK_COUNT) for fold in args.folds):
        raise ValueError("folds must contain only 0..4")
    if len(args.folds) != len(set(args.folds)):
        raise ValueError("folds must be unique")
    cache_dir = args.cache_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir == HERE or HERE in run_dir.parents:
        raise ValueError("Run directory must be outside the source checkout")
    contract = _contract(args, cache_dir)
    if args.plan_only:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("Existing run contract differs; choose a new run directory")
    else:
        atomic_json(contract_path, contract)

    blocks = load_sub01_blocks(cache_dir)
    completed: list[dict[str, Any]] = []
    atomic_json(
        run_dir / "queue_state.json",
        {
            "schema_version": 1,
            "status": "running",
            "current_fold": None,
            "completed_folds": [],
            "remaining_folds": args.folds,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
        overwrite=True,
    )
    for fold in args.folds:
        train_indexes, validation_index, test_index = fold_indexes(fold)
        train_ids, _, train_neural = select(blocks, train_indexes)
        val_ids, _, val_neural = select(blocks, (validation_index,))
        test_ids, _, test_neural = select(blocks, (test_index,))
        if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
            raise RuntimeError("Sample leakage between train/validation/test")
        fold_root = run_dir / f"fold_{fold:02d}"
        fold_result_path = fold_root / "fold_result.json"
        if fold_result_path.is_file():
            print(f"[fold {fold}] validated completion -> reuse", flush=True)
            completed.append(_load_completed_fold(fold_result_path, contract["fingerprint"]))
            continue
        if fold_root.exists():
            raise RuntimeError(
                f"Incomplete fold directory exists: {fold_root}. Preserve it for audit "
                "and choose a new run directory."
            )
        atomic_json(
            run_dir / "queue_state.json",
            {
                "schema_version": 1,
                "status": "running",
                "current_fold": fold,
                "completed_folds": sorted(int(item["fold"]) for item in completed),
                "remaining_folds": [item for item in args.folds if item != fold and item not in {int(done["fold"]) for done in completed}],
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            overwrite=True,
        )
        fold_root.mkdir(parents=True, exist_ok=False)
        print(
            f"[fold {fold}] train={list(train_indexes)} val={validation_index} test={test_index}",
            flush=True,
        )
        neural_reducer = fit_neural_reducer(train_neural, args.dimension, args.seed)
        _save_neural_reducer(fold_root / "neural_reducer.npz", neural_reducer)
        train_x = neural_reducer.transform(train_neural)
        val_x = neural_reducer.transform(val_neural)
        test_x = neural_reducer.transform(test_neural)
        _, _, train_mel = select(blocks, train_indexes, "mel80")
        _, _, val_mel = select(blocks, (validation_index,), "mel80")
        _, _, test_mel = select(blocks, (test_index,), "mel80")
        from sklearn.preprocessing import StandardScaler

        mel_scaler = StandardScaler().fit(train_mel)
        train_mel_z = mel_scaler.transform(train_mel)
        val_mel_z = mel_scaler.transform(val_mel)
        test_mel_z = mel_scaler.transform(test_mel)
        fold_payload: dict[str, Any] = {
            "fold": fold,
            "contract_fingerprint": contract["fingerprint"],
            "train_blocks": list(train_indexes),
            "validation_block": validation_index,
            "test_block": test_index,
            "train_frames": int(len(train_ids)),
            "validation_frames": int(len(val_ids)),
            "test_frames": int(len(test_ids)),
            "train_sample_ids_sha256": canonical_hash(train_ids.tolist()),
            "validation_sample_ids_sha256": canonical_hash(val_ids.tolist()),
            "test_sample_ids_sha256": canonical_hash(test_ids.tolist()),
            "results": {},
        }
        for target in args.targets:
            _, _, train_raw = select(blocks, train_indexes, target)
            _, _, val_raw = select(blocks, (validation_index,), target)
            _, _, test_raw = select(blocks, (test_index,), target)
            for method in args.methods:
                label = f"{target}__{method}"
                print(f"  [{label}] fit train-only projector and decoder", flush=True)
                if method == "pca50":
                    bottleneck = fit_pca_bottleneck(train_raw, args.dimension, args.seed)
                else:
                    bottleneck = fit_supervised_rrr_bottleneck(train_x, train_raw, args.dimension)
                if bottleneck.orthogonality_error() > 1e-8:
                    raise RuntimeError(f"Projector orthogonality failed for {label}")
                train_scores = bottleneck.transform(train_raw)
                val_scores = bottleneck.transform(val_raw)
                test_scores = bottleneck.transform(test_raw)
                decoder = fit_linear(train_x, train_scores)
                val_prediction = decoder.predict(val_x)
                test_prediction = decoder.predict(test_x)
                mel_probe = fit_linear(train_scores, train_mel_z)
                val_mel_prediction = mel_probe.predict(val_prediction)
                test_mel_prediction = mel_probe.predict(test_prediction)
                val_raw_prediction = bottleneck.inverse_transform(val_prediction)
                test_raw_prediction = bottleneck.inverse_transform(test_prediction)
                validation = {
                    "score50": component_metrics(val_scores, val_prediction),
                    "target_raw": component_metrics(val_raw, val_raw_prediction),
                    "mel80_probe": component_metrics(val_mel_z, val_mel_prediction),
                    "mel_low20_probe": component_metrics(val_mel_z[:, :20], val_mel_prediction[:, :20]),
                }
                test = {
                    "score50": component_metrics(test_scores, test_prediction),
                    "target_raw": component_metrics(test_raw, test_raw_prediction),
                    "mel80_probe": component_metrics(test_mel_z, test_mel_prediction),
                    "mel_low20_probe": component_metrics(test_mel_z[:, :20], test_mel_prediction[:, :20]),
                }
                artifact_hashes = save_fold_artifacts(
                    fold_root / label,
                    bottleneck,
                    decoder,
                    mel_probe,
                    test_ids,
                    test_scores,
                    test_prediction,
                    test_raw,
                    test_raw_prediction,
                    test_mel_z,
                    test_mel_prediction,
                )
                fold_payload["results"][label] = {
                    "target": target,
                    "method": method,
                    "raw_dimension": int(train_raw.shape[1]),
                    "bottleneck_dimension": args.dimension,
                    "projector_orthogonality_frobenius_error": bottleneck.orthogonality_error(),
                    "projector_retained_standardized_variance": float(np.sum(bottleneck.explained_variance_ratio)),
                    "validation": validation,
                    "test": test,
                    **artifact_hashes,
                }
                print(
                    f"    test score-r={test['score50']['fisher_z_component_correlation']:.4f} "
                    f"mel80-r={test['mel80_probe']['fisher_z_component_correlation']:.4f} "
                    f"low20-r={test['mel_low20_probe']['fisher_z_component_correlation']:.4f}",
                    flush=True,
                )
        fold_payload["fingerprint"] = canonical_hash(fold_payload)
        atomic_json(fold_result_path, fold_payload)
        completed.append(fold_payload)

    expected = set(args.folds)
    actual = {int(fold["fold"]) for fold in completed}
    if actual != expected:
        raise RuntimeError("Fold completion set mismatch")
    labels = [f"{target}__{method}" for target in args.targets for method in args.methods]
    aggregate = {
        label: aggregate_fold_metrics([fold["results"][label]["test"] for fold in completed])
        for label in labels
    }
    summary = {
        "schema_version": 1,
        "kind": "swpd_sub01_learned_bottleneck_result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_fingerprint": contract["fingerprint"],
        "development_subject": "sub-01",
        "confirmatory_subjects_read": False,
        "fold_count": len(completed),
        "aggregate_test": aggregate,
        "interpretation_guard": (
            "Development-only comparison. MEL-probe metrics are comparable across targets; "
            "target-space score correlations are representation-specific."
        ),
    }
    atomic_json(run_dir / "summary.json", summary, overwrite=True)
    atomic_json(
        run_dir / "queue_state.json",
        {
            "schema_version": 1,
            "status": "completed",
            "completed_folds": sorted(actual),
            "summary": str(run_dir / "summary.json"),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
        overwrite=True,
    )
    print(f"COMPLETE | {run_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
