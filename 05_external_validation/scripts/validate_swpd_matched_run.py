#!/usr/bin/env python3
"""Validate a completed SWPD matched PCA50 run without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGETS = ("mel80", "L3", "L4", "L5")
FOLD_COUNT = 5
TARGET_DIMENSION = 50


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(run_directory: Path) -> dict[str, int | bool]:
    root = run_directory.expanduser().resolve()
    summary = _json(root / "matched_linear_summary.json")
    manifest = _json(root / "run_manifest.json")
    if summary.get("target_dimension") != TARGET_DIMENSION:
        raise RuntimeError("Summary target dimension is not PCA50")
    if manifest.get("confirmatory_subjects_read") is not False:
        raise RuntimeError("Run claims access to confirmatory subjects")
    all_test_ids: list[str] = []
    for fold_index in range(FOLD_COUNT):
        fold = root / f"fold_{fold_index:02d}"
        result = _json(fold / "fold_result.json")
        train_hash = result["train_sample_ids_sha256"]
        train_frames = int(result["train_frames"])
        test_frames = int(result["test_frames"])
        reducers = ("neural_reducer",) + tuple(
            f"{target}_target_reducer" for target in TARGETS
        )
        for reducer in reducers:
            receipt = _json(fold / reducer / "manifest.json")
            if receipt.get("kind") != "train_only_standard_scaler_pca":
                raise RuntimeError(f"Unexpected reducer kind: {fold / reducer}")
            if receipt.get("output_dim") != TARGET_DIMENSION or receipt.get("whiten") is not True:
                raise RuntimeError(f"Reducer is not whitened PCA50: {fold / reducer}")
            if receipt.get("train_sample_ids_sha256") != train_hash:
                raise RuntimeError(f"Reducer was not fit on declared train IDs: {fold / reducer}")
            if receipt.get("train_sample_count") != train_frames:
                raise RuntimeError(f"Reducer train count mismatch: {fold / reducer}")

        target_test_ids: list[np.ndarray] = []
        for target in TARGETS:
            with np.load(fold / f"{target}_test_predictions.npz", allow_pickle=False) as arrays:
                truth = np.asarray(arrays["truth"])
                prediction = np.asarray(arrays["prediction"])
                ids = np.asarray(arrays["sample_ids"])
            expected_shape = (test_frames, TARGET_DIMENSION)
            if truth.shape != expected_shape or prediction.shape != expected_shape:
                raise RuntimeError(f"Prediction shape mismatch for fold {fold_index} {target}")
            if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
                raise RuntimeError(f"Non-finite prediction data for fold {fold_index} {target}")
            if len(np.unique(ids)) != len(ids):
                raise RuntimeError(f"Duplicate test IDs inside fold {fold_index} {target}")
            target_test_ids.append(ids)
        for ids in target_test_ids[1:]:
            np.testing.assert_array_equal(target_test_ids[0], ids)
        all_test_ids.extend(target_test_ids[0].tolist())

    if len(set(all_test_ids)) != len(all_test_ids):
        raise RuntimeError("A test frame appears in more than one outer fold")
    return {
        "folds": FOLD_COUNT,
        "targets": len(TARGETS),
        "test_rows": len(all_test_ids),
        "unique_test_rows": len(set(all_test_ids)),
        "target_dimension": TARGET_DIMENSION,
        "train_only_reducers": True,
        "finite_predictions": True,
        "confirmatory_subjects_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run_directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
