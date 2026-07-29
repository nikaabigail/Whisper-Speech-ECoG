"""Pure fitting and evaluation utilities for the SWPD protocol bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Cell:
    target: np.ndarray
    target_pca50: bool = False


def component_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[0] < 3:
        raise ValueError("truth and prediction must be aligned non-trivial 2D arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metric arrays contain NaN or Infinity")
    correlations = np.full(actual.shape[1], np.nan, dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    for index in range(actual.shape[1]):
        left = actual[:, index]
        right = predicted[:, index]
        if np.std(left) > epsilon and np.std(right) > epsilon:
            correlations[index] = np.corrcoef(left, right)[0, 1]
    valid = correlations[np.isfinite(correlations)]
    if valid.size == 0:
        raise RuntimeError("no target component has a finite correlation")

    def summarize(values: np.ndarray) -> dict[str, float | int]:
        clipped = np.clip(values, -1 + 1e-7, 1 - 1e-7)
        return {
            "mean_pearson_r": float(np.mean(values)),
            "fisher_z_mean_r": float(np.tanh(np.mean(np.arctanh(clipped)))),
            "component_count": int(values.size),
        }

    low_count = min(20, correlations.size)
    low = correlations[:low_count]
    low = low[np.isfinite(low)]
    return {
        "all_bins": summarize(valid),
        "lower_20_bins": summarize(low),
        "component_correlations": [
            float(value) if np.isfinite(value) else None for value in correlations
        ],
    }


def sequential_folds(sample_count: int, fold_count: int = 10) -> tuple[dict[str, Any], ...]:
    if sample_count < fold_count or fold_count < 2:
        raise ValueError("invalid sequential fold geometry")
    splitter = KFold(n_splits=fold_count, shuffle=False)
    return tuple(
        {
            "fold": fold,
            "train": np.asarray(train, dtype=np.int64),
            "validation": np.empty(0, dtype=np.int64),
            "test": np.asarray(test, dtype=np.int64),
            "split_kind": "non-shuffled sequential KFold",
        }
        for fold, (train, test) in enumerate(splitter.split(np.arange(sample_count)))
    )


def visual_block_folds(block_labels: np.ndarray) -> tuple[dict[str, Any], ...]:
    labels = np.asarray(block_labels, dtype=np.int64)
    if labels.ndim != 1 or set(np.unique(labels).tolist()) != set(range(5)):
        raise ValueError("block labels must contain exactly blocks 0..4")
    folds = []
    for test_block in range(5):
        validation_block = (test_block + 1) % 5
        train_blocks = tuple(
            block for block in range(5) if block not in (test_block, validation_block)
        )
        train = np.flatnonzero(np.isin(labels, train_blocks))
        validation = np.flatnonzero(labels == validation_block)
        test = np.flatnonzero(labels == test_block)
        if not train.size or not validation.size or not test.size:
            raise ValueError("an empty train/validation/test block was produced")
        if np.intersect1d(train, validation).size or np.intersect1d(train, test).size:
            raise RuntimeError("block split overlap")
        folds.append(
            {
                "fold": test_block,
                "train": train,
                "validation": validation,
                "test": test,
                "train_blocks": list(train_blocks),
                "validation_block": validation_block,
                "test_block": test_block,
                "split_kind": "five visual blocks; 3 train + 1 validation + 1 test",
            }
        )
    return tuple(folds)


def block_labels_for_times(
    times_seconds: np.ndarray,
    block_bounds: Sequence[tuple[float, float]],
    *,
    edge_guard_seconds: float,
) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64)
    if times.ndim != 1 or len(block_bounds) != 5:
        raise ValueError("five block bounds and a one-dimensional timeline are required")
    if not np.isfinite(times).all() or edge_guard_seconds < 0:
        raise ValueError("invalid timeline or edge guard")
    labels = np.full(times.shape, -1, dtype=np.int8)
    for index, (start, stop) in enumerate(block_bounds):
        if stop <= start + 2 * edge_guard_seconds:
            raise ValueError("edge guard removes an entire block")
        selected = (times >= start + edge_guard_seconds) & (
            times <= stop - edge_guard_seconds
        )
        if np.any(labels[selected] >= 0):
            raise ValueError("visual block bounds overlap")
        labels[selected] = index
    return labels


def _fit_neural_space(
    neural: np.ndarray, train: np.ndarray, test: np.ndarray, components: int
) -> tuple[np.ndarray, np.ndarray, float]:
    scaler = StandardScaler(copy=True).fit(neural[train])
    train_standardized = scaler.transform(neural[train])
    test_standardized = scaler.transform(neural[test])
    maximum = min(train_standardized.shape)
    if not 1 <= components <= maximum:
        raise ValueError(f"neural PCA components must be within 1..{maximum}")
    pca = PCA(n_components=components, whiten=False, svd_solver="full")
    train_x = pca.fit_transform(train_standardized)
    test_x = pca.transform(test_standardized)
    return train_x, test_x, float(np.sum(pca.explained_variance_ratio_))


def _fit_target(
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    target_pca50: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_y_raw = np.asarray(target[train], dtype=np.float64)
    test_y_raw = np.asarray(target[test], dtype=np.float64)
    if not target_pca50:
        estimator = LinearRegression(n_jobs=1).fit(train_x, train_y_raw)
        prediction = estimator.predict(test_x)
        return test_y_raw, prediction, {"target_transform": "raw coordinates"}

    target_scaler = StandardScaler(copy=True).fit(train_y_raw)
    train_y_standardized = target_scaler.transform(train_y_raw)
    maximum = min(train_y_standardized.shape)
    if maximum < 50:
        raise ValueError("target does not support PCA50")
    target_pca = PCA(n_components=50, whiten=True, svd_solver="full")
    train_y = target_pca.fit_transform(train_y_standardized)
    estimator = LinearRegression(n_jobs=1).fit(train_x, train_y)
    predicted_scores = estimator.predict(test_x)
    prediction = target_scaler.inverse_transform(
        target_pca.inverse_transform(predicted_scores)
    )
    return test_y_raw, prediction, {
        "target_transform": "fold-train StandardScaler + whitened PCA50 + inverse",
        "target_pca_explained_variance": float(
            np.sum(target_pca.explained_variance_ratio_)
        ),
    }


def _aggregate(folds: Sequence[Mapping[str, Any]], section: str) -> dict[str, Any]:
    mean_values = np.asarray(
        [fold["metrics"][section]["mean_pearson_r"] for fold in folds],
        dtype=np.float64,
    )
    fisher_values = np.asarray(
        [fold["metrics"][section]["fisher_z_mean_r"] for fold in folds],
        dtype=np.float64,
    )
    return {
        "fold_mean_pearson_r": {
            "mean": float(np.mean(mean_values)),
            "sd": float(np.std(mean_values, ddof=1)) if mean_values.size > 1 else 0.0,
        },
        "fold_fisher_z_mean_r": {
            "mean": float(np.mean(fisher_values)),
            "sd": float(np.std(fisher_values, ddof=1)) if fisher_values.size > 1 else 0.0,
        },
        "fold_count": int(mean_values.size),
    }


def evaluate_group(
    neural: np.ndarray,
    cells: Mapping[str, Cell],
    folds: Sequence[Mapping[str, Any]],
    sample_ids: np.ndarray,
    *,
    neural_pca_components: int = 50,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x = np.asarray(neural, dtype=np.float64)
    ids = np.asarray(sample_ids)
    if x.ndim != 2 or ids.shape != (x.shape[0],) or not np.isfinite(x).all():
        raise ValueError("invalid neural matrix or sample IDs")
    if not cells:
        raise ValueError("at least one target cell is required")
    for name, cell in cells.items():
        target = np.asarray(cell.target)
        if not name or target.ndim != 2 or target.shape[0] != x.shape[0]:
            raise ValueError(f"invalid target cell {name!r}")
        if not np.isfinite(target).all():
            raise ValueError(f"target cell {name!r} contains NaN or Infinity")

    results: dict[str, Any] = {
        name: {"folds": [], "target_pca50": bool(cell.target_pca50)}
        for name, cell in cells.items()
    }
    bundles: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"truth": [], "prediction": [], "sample_ids": [], "fold": []}
        for name in cells
    }
    for split in folds:
        train = np.asarray(split["train"], dtype=np.int64)
        test = np.asarray(split["test"], dtype=np.int64)
        if np.intersect1d(train, test).size:
            raise RuntimeError("train/test overlap")
        train_x, test_x, neural_explained = _fit_neural_space(
            x, train, test, neural_pca_components
        )
        for name, cell in cells.items():
            truth, prediction, transform = _fit_target(
                np.asarray(cell.target),
                train,
                test,
                train_x,
                test_x,
                target_pca50=cell.target_pca50,
            )
            fold_result = {
                "fold": int(split["fold"]),
                "split_kind": str(split["split_kind"]),
                "train_count": int(train.size),
                "validation_count": int(len(split.get("validation", ()))),
                "test_count": int(test.size),
                "neural_pca_explained_variance": neural_explained,
                "metrics": component_metrics(truth, prediction),
                **transform,
            }
            for key in ("train_blocks", "validation_block", "test_block"):
                if key in split:
                    fold_result[key] = split[key]
            results[name]["folds"].append(fold_result)
            bundles[name]["truth"].append(np.asarray(truth, dtype=np.float32))
            bundles[name]["prediction"].append(
                np.asarray(prediction, dtype=np.float32)
            )
            bundles[name]["sample_ids"].append(ids[test])
            bundles[name]["fold"].append(
                np.full(test.size, int(split["fold"]), dtype=np.int8)
            )

    flat: dict[str, np.ndarray] = {}
    for name in cells:
        results[name]["aggregate"] = {
            "all_bins": _aggregate(results[name]["folds"], "all_bins"),
            "lower_20_bins": _aggregate(
                results[name]["folds"], "lower_20_bins"
            ),
        }
        for field, chunks in bundles[name].items():
            flat[f"{name}__{field}"] = np.concatenate(chunks, axis=0)
    return results, flat
