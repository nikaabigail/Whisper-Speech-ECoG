"""Frozen contextual extraction, cache validation and MEL/L4 evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from swpd_protocol_bridge.bridge_core import component_metrics
from whisper_ecog_ext.integrity import atomic_write_json, fingerprint_json, read_json, sha256_file
from whisper_ecog_ext.swpd.author_mel import CONTEXT_STEP_FRAMES, MODEL_ORDER, extract_high_gamma, stack_author_context
from whisper_ecog_ext.swpd.matched_linear import AUTHOR_AUDIO_PROCESSING_RATE, EDGE_GUARD_SECONDS
from whisper_ecog_ext.swpd.nwb import SWPDRecording, recording_relative_sample_bounds


FRAME_SHIFT_SECONDS = 0.01
WINDOW_SECONDS = 0.05


@dataclass(frozen=True)
class FrozenBlock:
    index: int
    sample_ids: np.ndarray
    times: np.ndarray
    neural: np.ndarray
    mel80: np.ndarray
    l4: np.ndarray


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cache_contract(subject: str, inventory: Any, definition: Any, mel: Any, whisper: Any, batch_size: int) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "swpd_frozen_contextual_block_contract",
        "subject": subject,
        "block": asdict(definition),
        "inventory": inventory.to_dict(),
        "window_seconds": WINDOW_SECONDS,
        "base_grid_seconds": FRAME_SHIFT_SECONDS,
        "output_grid_seconds": 0.02,
        "context_model_order": MODEL_ORDER,
        "context_step_frames": CONTEXT_STEP_FRAMES,
        "edge_guard_seconds": EDGE_GUARD_SECONDS,
        "channel_batch_size": batch_size,
        "mel_provenance": mel.provenance(),
        "whisper_provenance": whisper.provenance(),
        "implementation_sha256": {
            "core": sha256_file(Path(__file__)),
            "author_mel": sha256_file(Path(extract_high_gamma.__code__.co_filename)),
        },
    }
    return payload


def load_block(cache: Path, subject: str, index: int, contract_fp: str) -> FrozenBlock:
    arrays_path = cache / f"block_{index:02d}.npz"
    manifest_path = cache / f"block_{index:02d}.json"
    payload = read_json(manifest_path)
    stored = payload.pop("fingerprint", None)
    if stored != fingerprint_json(payload) or payload.get("contract_fingerprint") != contract_fp:
        raise RuntimeError(f"Frozen contextual cache contract mismatch: {subject} block {index}")
    if payload.get("subject") != subject or int(payload.get("block", -1)) != index:
        raise RuntimeError("Frozen contextual cache subject/block mismatch")
    if sha256_file(arrays_path) != payload.get("arrays_sha256"):
        raise RuntimeError("Frozen contextual cache checksum mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        result = FrozenBlock(index, np.asarray(archive["sample_ids"]), np.asarray(archive["times"]), np.asarray(archive["neural"]), np.asarray(archive["mel80"]), np.asarray(archive["L4"]))
    rows = len(result.sample_ids)
    if result.neural.shape[0] != rows or result.mel80.shape != (rows, 80) or result.l4.shape != (rows, 512):
        raise RuntimeError("Frozen contextual cache array shape mismatch")
    if any(not np.isfinite(x).all() for x in (result.times, result.neural, result.mel80, result.l4)):
        raise RuntimeError("Frozen contextual cache contains non-finite values")
    if rows <= 50 or np.unique(result.sample_ids).size != rows or np.any(np.diff(result.times) <= 0):
        raise RuntimeError("Frozen contextual cache has an invalid or non-unique timeline")
    return result


def build_or_load_block(recording: SWPDRecording, subject: str, inventory: Any, definition: Any, cache: Path, mel: Any, whisper: Any, batch_size: int) -> tuple[FrozenBlock, dict[str, Any]]:
    contract = cache_contract(subject, inventory, definition, mel, whisper, batch_size)
    contract_fp = fingerprint_json(contract)
    arrays_path = cache / f"block_{definition.index:02d}.npz"
    manifest_path = cache / f"block_{definition.index:02d}.json"
    if arrays_path.is_file() and manifest_path.is_file():
        return load_block(cache, subject, definition.index, contract_fp), contract
    if arrays_path.exists() or manifest_path.exists():
        raise RuntimeError(f"Incomplete frozen cache: {subject} block {definition.index}")
    bounds = recording_relative_sample_bounds(definition.start_seconds, definition.stop_seconds, inventory.ieeg)
    raw = recording.read_ieeg(bounds.start_index, bounds.stop_index)
    parts = []
    for first in range(0, raw.shape[1], batch_size):
        parts.append(extract_high_gamma(raw[:, first:first + batch_size], inventory.ieeg.rate_hz, window_seconds=WINDOW_SECONDS, frame_shift_seconds=FRAME_SHIFT_SECONDS))
    high_gamma = np.concatenate(parts, axis=1)
    contextual = stack_author_context(high_gamma)
    edge = MODEL_ORDER * CONTEXT_STEP_FRAMES
    base_times = bounds.actual_start_absolute_seconds + np.arange(high_gamma.shape[0]) * FRAME_SHIFT_SECONDS + WINDOW_SECONDS / 2
    absolute_times = base_times[edge:high_gamma.shape[0] - edge]
    if contextual.shape[0] != absolute_times.size:
        raise RuntimeError("Context matrix and frame times disagree")
    relative_times = absolute_times - inventory.ieeg.starting_time_seconds
    keep = (np.arange(len(contextual)) % 2 == 0) & (relative_times >= definition.start_seconds + EDGE_GUARD_SECONDS) & (relative_times <= definition.stop_seconds - EDGE_GUARD_SECONDS)
    contextual = np.asarray(contextual[keep], dtype=np.float32)
    relative_times = relative_times[keep]
    absolute_times = absolute_times[keep]
    if contextual.shape[0] <= 50 or np.any(np.diff(relative_times) <= 0):
        raise RuntimeError(f"Block {definition.index} has too few or non-increasing contextual frames")
    audio_bounds = recording_relative_sample_bounds(definition.start_seconds, definition.stop_seconds, inventory.audio)
    audio = recording.read_audio(audio_bounds.start_index, audio_bounds.stop_index).astype(np.float32)
    local_times = absolute_times - audio_bounds.actual_start_absolute_seconds
    mel_values = mel.extract_aligned(audio, AUTHOR_AUDIO_PROCESSING_RATE, local_times)
    whisper_values = whisper.extract_aligned(audio, AUTHOR_AUDIO_PROCESSING_RATE, local_times)[4]
    if np.asarray(mel_values).shape != (len(contextual), 80) or np.asarray(whisper_values).shape != (len(contextual), 512):
        raise RuntimeError("Aligned target shape differs from frozen MEL80/L4 contract")
    ids = np.asarray([f"{subject}:block-{definition.index:02d}:context-frame-{i:05d}" for i in range(len(contextual))], dtype="U58")
    _atomic_npz(arrays_path, {"sample_ids": ids, "times": relative_times, "neural": contextual, "mel80": np.asarray(mel_values, dtype=np.float32), "L4": np.asarray(whisper_values, dtype=np.float32)})
    manifest = {
        "schema_version": 1,
        "kind": "swpd_frozen_contextual_block_cache",
        "subject": subject,
        "block": int(definition.index),
        "frame_count": int(len(ids)),
        "contract_fingerprint": contract_fp,
        "arrays_sha256": sha256_file(arrays_path),
        "arrays_file": arrays_path.name,
    }
    manifest["fingerprint"] = fingerprint_json(manifest)
    atomic_write_json(manifest_path, manifest, overwrite=False)
    return load_block(cache, subject, definition.index, contract_fp), contract


def _rows(blocks: Sequence[FrozenBlock], indexes: Sequence[int], field: str) -> np.ndarray:
    return np.concatenate([getattr(blocks[index], field) for index in indexes], axis=0)


def _aggregate(folds: Sequence[Mapping[str, Any]], system: str) -> dict[str, Any]:
    all_values = np.asarray([fold[system]["all_bins"]["mean_pearson_r"] for fold in folds])
    low_values = np.asarray([fold[system]["lower_20_bins"]["mean_pearson_r"] for fold in folds])
    return {
        "all_bins": {"mean": float(all_values.mean()), "sd": float(all_values.std(ddof=1)), "values": all_values.tolist()},
        "lower_20_bins": {"mean": float(low_values.mean()), "sd": float(low_values.std(ddof=1)), "values": low_values.tolist()},
    }


def evaluate_subject(blocks: Sequence[FrozenBlock], subject: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(blocks) != 5:
        raise ValueError("Frozen evaluation requires five blocks")
    folds = []
    chunks: dict[str, list[np.ndarray]] = {"sample_ids": [], "fold": [], "truth_mel80_z": [], "direct_mel80_prediction_z": [], "l4_pca50_prediction_z": []}
    for test in range(5):
        validation = (test + 1) % 5
        train = tuple(i for i in range(5) if i not in (test, validation))
        train_neural, test_neural = _rows(blocks, train, "neural"), _rows(blocks, (test,), "neural")
        neural_scaler = StandardScaler().fit(train_neural)
        pca_x = PCA(n_components=50, whiten=False, svd_solver="full").fit(neural_scaler.transform(train_neural))
        train_x = pca_x.transform(neural_scaler.transform(train_neural)); test_x = pca_x.transform(neural_scaler.transform(test_neural))
        train_mel, test_mel = _rows(blocks, train, "mel80"), _rows(blocks, (test,), "mel80")
        mel_scaler = StandardScaler().fit(train_mel)
        train_mel_z, test_mel_z = mel_scaler.transform(train_mel), mel_scaler.transform(test_mel)
        direct_prediction = LinearRegression(n_jobs=1).fit(train_x, train_mel_z).predict(test_x)
        train_l4, test_l4 = _rows(blocks, train, "l4"), _rows(blocks, (test,), "l4")
        l4_scaler = StandardScaler().fit(train_l4)
        pca_l4 = PCA(n_components=50, whiten=True, svd_solver="full").fit(l4_scaler.transform(train_l4))
        train_scores = pca_l4.transform(l4_scaler.transform(train_l4))
        test_scores = pca_l4.transform(l4_scaler.transform(test_l4))
        predicted_scores = LinearRegression(n_jobs=1).fit(train_x, train_scores).predict(test_x)
        l4_prediction = LinearRegression(n_jobs=1).fit(train_scores, train_mel_z).predict(predicted_scores)
        fold = {"fold": test, "train_blocks": list(train), "validation_block": validation, "test_block": test, "direct_mel80": component_metrics(test_mel_z, direct_prediction), "whisper_l4_pca50": component_metrics(test_mel_z, l4_prediction), "l4_score50": component_metrics(test_scores, predicted_scores)}
        folds.append(fold)
        ids = _rows(blocks, (test,), "sample_ids")
        chunks["sample_ids"].append(ids); chunks["fold"].append(np.full(len(ids), test, dtype=np.int8)); chunks["truth_mel80_z"].append(test_mel_z.astype(np.float32)); chunks["direct_mel80_prediction_z"].append(direct_prediction.astype(np.float32)); chunks["l4_pca50_prediction_z"].append(l4_prediction.astype(np.float32))
    result = {"subject": subject, "folds": folds, "aggregate": {"direct_mel80": _aggregate(folds, "direct_mel80"), "whisper_l4_pca50": _aggregate(folds, "whisper_l4_pca50")}}
    result["delta_l4_minus_mel80"] = float(result["aggregate"]["whisper_l4_pca50"]["all_bins"]["mean"] - result["aggregate"]["direct_mel80"]["all_bins"]["mean"])
    predictions = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    return result, predictions
