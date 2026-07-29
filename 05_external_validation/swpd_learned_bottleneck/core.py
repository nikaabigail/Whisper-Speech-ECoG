"""Core transforms and metrics for the isolated SWPD bottleneck experiment.

This module intentionally does not import or modify the frozen matched-PCA50
implementation.  It consumes its immutable block caches and creates new output
artifacts in a separate run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


BLOCK_COUNT = 5
TARGETS = ("mel80", "L3", "L4", "L5", "L345")
TARGET_DIMS = {"mel80": 80, "L3": 512, "L4": 512, "L5": 512, "L345": 1536}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class CachedBlock:
    index: int
    sample_ids: np.ndarray
    frame_times_seconds: np.ndarray
    neural: np.ndarray
    targets: Mapping[str, np.ndarray]


def _validate_manifest(manifest_path: Path, arrays_path: Path, index: int) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored = payload.pop("fingerprint", None)
    if stored != canonical_hash(payload):
        raise RuntimeError(f"Cache manifest fingerprint mismatch: {manifest_path}")
    if payload.get("kind") != "swpd_matched_linear_block_cache":
        raise RuntimeError(f"Unexpected cache kind: {manifest_path}")
    if int(payload["definition"]["index"]) != index:
        raise RuntimeError(f"Cache block index mismatch: {manifest_path}")
    if payload.get("arrays_file") != arrays_path.name:
        raise RuntimeError(f"Cache arrays filename mismatch: {manifest_path}")
    if payload.get("arrays_sha256") != sha256_file(arrays_path):
        raise RuntimeError(f"Cache arrays checksum mismatch: {arrays_path}")
    return payload


def load_sub01_blocks(cache_directory: Path) -> tuple[CachedBlock, ...]:
    """Load only sub-01 caches and verify every immutable cache receipt."""

    cache_directory = cache_directory.expanduser().resolve()
    if cache_directory.name.lower() != "sub-01":
        raise ValueError("Development runner accepts only a cache directory named sub-01")
    blocks: list[CachedBlock] = []
    for index in range(BLOCK_COUNT):
        arrays_path = cache_directory / f"block_{index:02d}.npz"
        manifest_path = cache_directory / f"block_{index:02d}.json"
        if not arrays_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete block cache {index}: {cache_directory}")
        manifest = _validate_manifest(manifest_path, arrays_path, index)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            sample_ids = np.asarray(arrays["sample_ids"])
            times = np.asarray(arrays["frame_times_seconds"], dtype=np.float64)
            neural = np.asarray(arrays["neural"], dtype=np.float32)
            targets = {
                name: np.asarray(arrays[name], dtype=np.float32)
                for name in ("mel80", "L3", "L4", "L5")
            }
        targets["L345"] = np.concatenate(
            [targets["L3"], targets["L4"], targets["L5"]], axis=1
        )
        count = len(sample_ids)
        if count != int(manifest["frame_count"]) or neural.shape[0] != count:
            raise RuntimeError(f"Cache frame count mismatch: {arrays_path}")
        if len(set(sample_ids.tolist())) != count:
            raise RuntimeError(f"Duplicate sample IDs: {arrays_path}")
        if not np.all(np.diff(times) > 0):
            raise RuntimeError(f"Frame times are not strictly increasing: {arrays_path}")
        for name, dimension in TARGET_DIMS.items():
            if targets[name].shape != (count, dimension):
                raise RuntimeError(f"Unexpected {name} shape in {arrays_path}")
            if not np.isfinite(targets[name]).all():
                raise RuntimeError(f"Non-finite {name} values in {arrays_path}")
        if not np.isfinite(neural).all():
            raise RuntimeError(f"Non-finite neural values in {arrays_path}")
        blocks.append(CachedBlock(index, sample_ids, times, neural, targets))
    all_ids = np.concatenate([block.sample_ids for block in blocks]).tolist()
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Sample IDs overlap between blocks")
    return tuple(blocks)


def fold_indexes(test_index: int) -> tuple[tuple[int, ...], int, int]:
    if test_index not in range(BLOCK_COUNT):
        raise ValueError("test_index must be 0..4")
    validation = (test_index + 1) % BLOCK_COUNT
    train = tuple(i for i in range(BLOCK_COUNT) if i not in (test_index, validation))
    return train, validation, test_index


def select(
    blocks: Sequence[CachedBlock], indexes: Iterable[int], target: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = [blocks[index] for index in indexes]
    ids = np.concatenate([block.sample_ids for block in chosen])
    times = np.concatenate([block.frame_times_seconds for block in chosen])
    values = np.concatenate(
        [block.neural if target is None else block.targets[target] for block in chosen], axis=0
    )
    return ids, times, values


@dataclass(frozen=True)
class NeuralReducer:
    scaler: StandardScaler
    pca: PCA

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(values)).astype(np.float64)


def fit_neural_reducer(values: np.ndarray, dimension: int, seed: int) -> NeuralReducer:
    scaler = StandardScaler().fit(np.asarray(values, dtype=np.float64))
    standardized = scaler.transform(values)
    pca = PCA(
        n_components=dimension, whiten=True, svd_solver="full", random_state=seed
    ).fit(standardized)
    return NeuralReducer(scaler, pca)


@dataclass(frozen=True)
class TargetBottleneck:
    method: str
    raw_scaler: StandardScaler
    projection: np.ndarray  # raw standardized target -> unscaled score
    score_mean: np.ndarray
    score_scale: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def input_dimension(self) -> int:
        return int(self.projection.shape[0])

    @property
    def output_dimension(self) -> int:
        return int(self.projection.shape[1])

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = self.raw_scaler.transform(np.asarray(values, dtype=np.float64))
        scores = standardized @ self.projection
        return (scores - self.score_mean) / self.score_scale

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        unscaled = np.asarray(scores, dtype=np.float64) * self.score_scale + self.score_mean
        standardized = unscaled @ self.projection.T
        return self.raw_scaler.inverse_transform(standardized)

    def orthogonality_error(self) -> float:
        identity = np.eye(self.output_dimension)
        return float(np.linalg.norm(self.projection.T @ self.projection - identity, ord="fro"))


def _score_scale(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(scores, axis=0)
    scale = np.std(scores, axis=0, ddof=0)
    epsilon = np.finfo(np.float64).eps
    if np.any(scale <= epsilon):
        raise RuntimeError("Bottleneck produced a constant training coordinate")
    return mean, scale


def fit_pca_bottleneck(values: np.ndarray, dimension: int, seed: int) -> TargetBottleneck:
    raw = np.asarray(values, dtype=np.float64)
    scaler = StandardScaler().fit(raw)
    standardized = scaler.transform(raw)
    pca = PCA(
        n_components=dimension, whiten=False, svd_solver="full", random_state=seed
    ).fit(standardized)
    projection = np.asarray(pca.components_.T, dtype=np.float64)
    scores = standardized @ projection
    mean, scale = _score_scale(scores)
    return TargetBottleneck(
        "pca50",
        scaler,
        projection,
        mean,
        scale,
        np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )


def fit_supervised_rrr_bottleneck(
    neural_scores: np.ndarray,
    target_values: np.ndarray,
    dimension: int,
) -> TargetBottleneck:
    """Fit a deterministic train-only orthonormal supervised target projector.

    The right singular vectors of Qx.T @ standardized_target maximize target
    covariance explainable by the neural training subspace.  The projector has
    orthonormal columns, so a zero/collapsed solution is impossible.
    """

    x = np.asarray(neural_scores, dtype=np.float64)
    raw = np.asarray(target_values, dtype=np.float64)
    if x.ndim != 2 or raw.ndim != 2 or x.shape[0] != raw.shape[0]:
        raise ValueError("Aligned 2D neural and target training matrices are required")
    if dimension > min(raw.shape):
        raise ValueError("Requested bottleneck dimension exceeds target matrix rank bound")
    scaler = StandardScaler().fit(raw)
    standardized = scaler.transform(raw)
    qx, _ = np.linalg.qr(x, mode="reduced")
    cross = qx.T @ standardized
    _, singular_values, vt = np.linalg.svd(cross, full_matrices=False)
    projection = np.asarray(vt[:dimension].T, dtype=np.float64)
    scores = standardized @ projection
    mean, scale = _score_scale(scores)
    total_variance = float(np.sum(np.var(standardized, axis=0, ddof=1)))
    score_variance = np.var(scores, axis=0, ddof=1)
    explained = score_variance / total_variance if total_variance > 0 else np.zeros(dimension)
    if singular_values.size < dimension:
        raise RuntimeError("Neural-target cross-covariance has too few singular directions")
    return TargetBottleneck(
        "srrr50", scaler, projection, mean, scale, np.asarray(explained, dtype=np.float64)
    )


def component_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[0] < 3:
        raise ValueError("Metrics require aligned 2D arrays with at least three rows")
    correlations = np.full(actual.shape[1], np.nan, dtype=np.float64)
    for index in range(actual.shape[1]):
        if np.std(actual[:, index]) > 1e-12 and np.std(predicted[:, index]) > 1e-12:
            correlations[index] = np.corrcoef(actual[:, index], predicted[:, index])[0, 1]
    valid = correlations[np.isfinite(correlations)]
    fisher = None
    if valid.size:
        fisher = float(
            np.tanh(np.mean(np.arctanh(np.clip(valid, -1 + 1e-7, 1 - 1e-7))))
        )
    residual = actual - predicted
    target_variance = np.var(actual, axis=0)
    valid_variance = target_variance > np.finfo(np.float64).eps
    explained = np.full(actual.shape[1], np.nan)
    explained[valid_variance] = 1.0 - (
        np.var(residual, axis=0)[valid_variance] / target_variance[valid_variance]
    )
    return {
        "frame_count": int(actual.shape[0]),
        "dimension": int(actual.shape[1]),
        "fisher_z_component_correlation": fisher,
        "mean_pearson_correlation": float(np.nanmean(correlations)),
        "median_pearson_correlation": float(np.nanmedian(correlations)),
        "mse": float(np.mean(residual**2)),
        "explained_variance": float(np.nanmean(explained)),
        "component_correlations": [float(v) if np.isfinite(v) else None for v in correlations],
    }


def fit_linear(source: np.ndarray, target: np.ndarray) -> LinearRegression:
    return LinearRegression(n_jobs=1).fit(
        np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    )


def save_fold_artifacts(
    directory: Path,
    bottleneck: TargetBottleneck,
    decoder: LinearRegression,
    mel_probe: LinearRegression,
    sample_ids: np.ndarray,
    truth_scores: np.ndarray,
    predicted_scores: np.ndarray,
    truth_raw: np.ndarray,
    predicted_raw: np.ndarray,
    truth_mel: np.ndarray,
    predicted_mel: np.ndarray,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        "projection": bottleneck.projection,
        "raw_scaler_mean": bottleneck.raw_scaler.mean_,
        "raw_scaler_scale": bottleneck.raw_scaler.scale_,
        "score_mean": bottleneck.score_mean,
        "score_scale": bottleneck.score_scale,
        "explained_variance_ratio": bottleneck.explained_variance_ratio,
        "decoder_coef": decoder.coef_,
        "decoder_intercept": decoder.intercept_,
        "mel_probe_coef": mel_probe.coef_,
        "mel_probe_intercept": mel_probe.intercept_,
    }
    model_path = directory / "model.npz"
    np.savez_compressed(model_path, **arrays)
    prediction_path = directory / "test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=sample_ids,
        truth_scores=truth_scores.astype(np.float32),
        predicted_scores=predicted_scores.astype(np.float32),
        truth_raw=truth_raw.astype(np.float32),
        predicted_raw=predicted_raw.astype(np.float32),
        truth_mel80=truth_mel.astype(np.float32),
        predicted_mel80=predicted_mel.astype(np.float32),
    )
    return {"model_sha256": sha256_file(model_path), "predictions_sha256": sha256_file(prediction_path)}


def aggregate_fold_metrics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    paths = {
        "score50_fisher_r": ("score50", "fisher_z_component_correlation"),
        "target_raw_fisher_r": ("target_raw", "fisher_z_component_correlation"),
        "mel80_fisher_r": ("mel80_probe", "fisher_z_component_correlation"),
        "mel_low20_fisher_r": ("mel_low20_probe", "fisher_z_component_correlation"),
        "mel80_mse": ("mel80_probe", "mse"),
    }
    for output_name, (section, metric) in paths.items():
        values = np.asarray([fold[section][metric] for fold in folds], dtype=np.float64)
        result[output_name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "fold_count": int(len(values)),
        }
    return result
