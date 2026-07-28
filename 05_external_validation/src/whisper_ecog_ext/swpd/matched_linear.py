"""Development-only matched linear MEL80 versus Whisper L3/L4/L5 analysis.

All targets share the same 20 ms high-gamma frames, visual-event block split,
fold-train neural transform, 50-dimensional whitened target space, and ordinary
least-squares decoder.  Visual events define blocks only; they are never called
acoustic speech onsets or used as a speech mask.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

from ..integrity import atomic_write_json, fingerprint_json, read_json, sha256_file
from ..reducer import fit_train_only_reducer
from .author_mel import extract_high_gamma
from .nwb import (
    PILOT_SUBJECT,
    SWPDInventory,
    SWPDRecording,
    VisualWordEvent,
    assert_series_start_alignment,
    load_visual_word_events,
    recording_relative_sample_bounds,
)


TARGET_NAMES = ("mel80", "L3", "L4", "L5")
FRAME_SHIFT_SECONDS = 0.02
WINDOW_SECONDS = 0.05
EDGE_GUARD_SECONDS = 1.0
TRIALS_PER_BLOCK = 20
BLOCK_COUNT = 5
REDUCED_DIMENSION = 50
AUTHOR_AUDIO_PROCESSING_RATE = 48_000


@dataclass(frozen=True)
class VisualBlock:
    index: int
    trial_ids: tuple[str, ...]
    first_trial_index: int
    last_trial_index: int
    start_seconds: float
    stop_seconds: float


@dataclass(frozen=True)
class MatchedBlock:
    definition: VisualBlock
    sample_ids: np.ndarray
    frame_times_seconds: np.ndarray
    neural: np.ndarray
    targets: Mapping[str, np.ndarray]


def make_visual_blocks(
    events: Sequence[VisualWordEvent], recording_stop_seconds: float
) -> tuple[VisualBlock, ...]:
    if len(events) != TRIALS_PER_BLOCK * BLOCK_COUNT:
        raise ValueError("Matched SWPD analysis requires exactly 100 visual word events")
    if recording_stop_seconds <= events[-1].onset_seconds:
        raise ValueError("Recording ends before the last visual word event")
    blocks: list[VisualBlock] = []
    for index in range(BLOCK_COUNT):
        first = index * TRIALS_PER_BLOCK
        stop_index = first + TRIALS_PER_BLOCK
        stop = (
            events[stop_index].onset_seconds
            if stop_index < len(events)
            else recording_stop_seconds
        )
        start = events[first].onset_seconds
        if stop - start <= 2 * EDGE_GUARD_SECONDS:
            raise ValueError(f"Visual block {index} is too short")
        selected = tuple(events[first:stop_index])
        blocks.append(
            VisualBlock(
                index=index,
                trial_ids=tuple(event.trial_id for event in selected),
                first_trial_index=selected[0].trial_index,
                last_trial_index=selected[-1].trial_index,
                start_seconds=start,
                stop_seconds=stop,
            )
        )
    return tuple(blocks)


def _neural_frame_times(
    sample_count: int,
    sample_rate: float,
    sample_start_seconds: float,
    frame_count: int,
) -> np.ndarray:
    starts = np.floor(
        np.arange(frame_count, dtype=np.float64) * FRAME_SHIFT_SECONDS * sample_rate
    )
    times = sample_start_seconds + (starts + WINDOW_SECONDS * sample_rate / 2.0) / sample_rate
    if np.any(np.diff(times) <= 0):
        raise RuntimeError("Neural 20 ms frame times are not increasing")
    return times


def _validate_targets(targets: Mapping[str, np.ndarray], frame_count: int) -> None:
    if tuple(targets) != TARGET_NAMES:
        raise RuntimeError(f"Target order must be {TARGET_NAMES}; got {tuple(targets)}")
    expected_dims = {"mel80": 80, "L3": 512, "L4": 512, "L5": 512}
    for name, values in targets.items():
        if values.shape != (frame_count, expected_dims[name]):
            raise RuntimeError(
                f"Unexpected {name} target shape {values.shape}; "
                f"expected {(frame_count, expected_dims[name])}"
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} target contains NaN or Inf")


def extract_one_block(
    recording: SWPDRecording,
    inventory: SWPDInventory,
    block: VisualBlock,
    *,
    mel_extractor: Any,
    whisper_extractor: Any,
    edge_guard_seconds: float = EDGE_GUARD_SECONDS,
) -> MatchedBlock:
    """Extract one independent visual block; no filter/model context crosses it."""

    assert_series_start_alignment(inventory)

    ieeg_bounds = recording_relative_sample_bounds(
        block.start_seconds, block.stop_seconds, inventory.ieeg
    )
    raw_neural = recording.read_ieeg(
        ieeg_bounds.start_index, ieeg_bounds.stop_index
    )
    neural = extract_high_gamma(
        raw_neural,
        inventory.ieeg.rate_hz,
        window_seconds=WINDOW_SECONDS,
        frame_shift_seconds=FRAME_SHIFT_SECONDS,
    )
    absolute_neural_times = _neural_frame_times(
        raw_neural.shape[0],
        inventory.ieeg.rate_hz,
        ieeg_bounds.actual_start_absolute_seconds,
        neural.shape[0],
    )
    recording_relative_times = (
        absolute_neural_times - inventory.ieeg.starting_time_seconds
    )
    valid = (recording_relative_times >= block.start_seconds + edge_guard_seconds) & (
        recording_relative_times <= block.stop_seconds - edge_guard_seconds
    )
    neural = np.asarray(neural[valid], dtype=np.float32)
    absolute_neural_times = absolute_neural_times[valid]
    recording_relative_times = recording_relative_times[valid]
    if neural.shape[0] <= REDUCED_DIMENSION:
        raise RuntimeError(f"Block {block.index} has too few valid neural frames")

    audio_bounds = recording_relative_sample_bounds(
        block.start_seconds, block.stop_seconds, inventory.audio
    )
    audio = recording.read_audio(
        audio_bounds.start_index, audio_bounds.stop_index
    ).astype(np.float32, copy=False)
    target_times_local = (
        absolute_neural_times - audio_bounds.actual_start_absolute_seconds
    )
    if np.min(target_times_local) < 0:
        raise RuntimeError("Neural target time precedes the selected audio block")

    mel = np.asarray(
        mel_extractor.extract_aligned(
            audio, AUTHOR_AUDIO_PROCESSING_RATE, target_times_local
        ),
        dtype=np.float32,
    )
    whisper = whisper_extractor.extract_aligned(
        audio, AUTHOR_AUDIO_PROCESSING_RATE, target_times_local
    )
    targets: dict[str, np.ndarray] = {
        "mel80": mel,
        "L3": np.asarray(whisper[3], dtype=np.float32),
        "L4": np.asarray(whisper[4], dtype=np.float32),
        "L5": np.asarray(whisper[5], dtype=np.float32),
    }
    _validate_targets(targets, neural.shape[0])
    ids = np.asarray(
        [
            f"{inventory.subject}:block-{block.index:02d}:frame-{frame:05d}"
            for frame in range(neural.shape[0])
        ],
        dtype="U48",
    )
    return MatchedBlock(
        definition=block,
        sample_ids=ids,
        frame_times_seconds=recording_relative_times,
        neural=neural,
        targets=targets,
    )


def _block_cache_paths(cache_directory: Path, index: int) -> tuple[Path, Path]:
    stem = f"block_{index:02d}"
    return cache_directory / f"{stem}.npz", cache_directory / f"{stem}.json"


def save_block_cache(
    block: MatchedBlock,
    cache_directory: Path,
    *,
    extraction_fingerprint: str,
) -> Path:
    cache_directory.mkdir(parents=True, exist_ok=True)
    arrays_path, manifest_path = _block_cache_paths(cache_directory, block.definition.index)
    if arrays_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Block cache already exists: {arrays_path}")
    temporary = arrays_path.with_name(arrays_path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            sample_ids=block.sample_ids,
            frame_times_seconds=block.frame_times_seconds,
            neural=block.neural,
            mel80=block.targets["mel80"],
            L3=block.targets["L3"],
            L4=block.targets["L4"],
            L5=block.targets["L5"],
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, arrays_path)
    payload = {
        "schema_version": 1,
        "kind": "swpd_matched_linear_block_cache",
        "extraction_fingerprint": extraction_fingerprint,
        "definition": asdict(block.definition),
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "frame_count": int(block.neural.shape[0]),
        "neural_dim": int(block.neural.shape[1]),
        "target_dims": {name: int(value.shape[1]) for name, value in block.targets.items()},
    }
    payload["fingerprint"] = fingerprint_json(payload)
    atomic_write_json(manifest_path, payload, overwrite=False)
    return arrays_path


def load_block_cache(
    cache_directory: Path,
    index: int,
    *,
    extraction_fingerprint: str,
) -> MatchedBlock | None:
    arrays_path, manifest_path = _block_cache_paths(cache_directory, index)
    if not arrays_path.exists() and not manifest_path.exists():
        return None
    if not arrays_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Incomplete block cache for block {index}")
    payload = read_json(manifest_path)
    stored_fingerprint = payload.pop("fingerprint", None)
    if stored_fingerprint != fingerprint_json(payload):
        raise RuntimeError(f"Block cache manifest was modified: {manifest_path}")
    if payload.get("extraction_fingerprint") != extraction_fingerprint:
        raise RuntimeError(f"Block cache fingerprint mismatch: {manifest_path}")
    if sha256_file(arrays_path) != payload.get("arrays_sha256"):
        raise RuntimeError(f"Block cache array checksum mismatch: {arrays_path}")
    definition = VisualBlock(**payload["definition"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        block = MatchedBlock(
            definition=definition,
            sample_ids=np.asarray(arrays["sample_ids"]),
            frame_times_seconds=np.asarray(arrays["frame_times_seconds"], dtype=np.float64),
            neural=np.asarray(arrays["neural"], dtype=np.float32),
            targets={
                name: np.asarray(arrays[name], dtype=np.float32) for name in TARGET_NAMES
            },
        )
    if len(block.sample_ids) != int(payload["frame_count"]):
        raise RuntimeError(f"Block cache frame count mismatch: {arrays_path}")
    _validate_targets(block.targets, block.neural.shape[0])
    return block


def build_extraction_fingerprint(
    *,
    inventory: SWPDInventory,
    events_path: Path,
    block: VisualBlock,
    mel_provenance: Mapping[str, Any],
    whisper_provenance: Mapping[str, Any],
) -> str:
    return fingerprint_json(
        {
            "schema_version": 1,
            "implementation": "swpd-matched-linear-block-v1",
            "implementation_files_sha256": {
                "matched_linear.py": sha256_file(Path(__file__)),
                "author_mel.py": sha256_file(Path(__file__).with_name("author_mel.py")),
                "targets.py": sha256_file(Path(__file__).parents[1] / "targets.py"),
            },
            "subject": inventory.subject,
            "nwb_path": inventory.nwb_path,
            "nwb_size_bytes": inventory.nwb_size_bytes,
            "events_sha256": sha256_file(events_path),
            "block": asdict(block),
            "frame_shift_seconds": FRAME_SHIFT_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "edge_guard_seconds": EDGE_GUARD_SECONDS,
            "audio_nwb_measured_rate": inventory.audio.rate_hz,
            "audio_processing_rate": AUTHOR_AUDIO_PROCESSING_RATE,
            "time_coordinate": "events_tsv_recording_relative_seconds",
            "acquisition_absolute_start_seconds": {
                "ieeg": inventory.ieeg.starting_time_seconds,
                "audio": inventory.audio.starting_time_seconds,
                "stimulus": inventory.stimulus.starting_time_seconds,
            },
            "stream_start_offsets_seconds": assert_series_start_alignment(inventory),
            "mel": dict(mel_provenance),
            "whisper": dict(whisper_provenance),
        }
    )


def _select(blocks: Sequence[MatchedBlock], indexes: Sequence[int]) -> tuple[np.ndarray, ...]:
    selected = [blocks[index] for index in indexes]
    ids = np.concatenate([block.sample_ids for block in selected])
    times = np.concatenate([block.frame_times_seconds for block in selected])
    neural = np.concatenate([block.neural for block in selected], axis=0)
    return ids, times, neural


def _target_select(
    blocks: Sequence[MatchedBlock], indexes: Sequence[int], target: str
) -> np.ndarray:
    return np.concatenate([blocks[index].targets[target] for index in indexes], axis=0)


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Metric arrays must be aligned 2D values")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (actual.shape[0],):
            raise ValueError("Speech mask must have one value per frame")
        actual = actual[selected]
        predicted = predicted[selected]
    if actual.shape[0] < 3:
        return None
    residual = actual - predicted
    component_mse = np.mean(residual**2, axis=0)
    target_variance = np.var(actual, axis=0)
    residual_variance = np.var(residual, axis=0)
    valid_variance = target_variance > np.finfo(np.float64).eps
    explained = np.full(actual.shape[1], np.nan)
    explained[valid_variance] = 1.0 - residual_variance[valid_variance] / target_variance[valid_variance]
    correlations = np.full(actual.shape[1], np.nan)
    for component in range(actual.shape[1]):
        left = actual[:, component]
        right = predicted[:, component]
        if np.std(left) > np.finfo(np.float64).eps and np.std(right) > np.finfo(np.float64).eps:
            correlations[component] = np.corrcoef(left, right)[0, 1]
    valid_r = correlations[np.isfinite(correlations)]
    fisher_r = float("nan")
    if valid_r.size:
        clipped = np.clip(valid_r, -1 + 1e-7, 1 - 1e-7)
        fisher_r = float(np.tanh(np.mean(np.arctanh(clipped))))
    def optional(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "frame_count": int(actual.shape[0]),
        "standardized_mse": float(np.mean(component_mse)),
        "explained_variance": optional(float(np.nanmean(explained))),
        "fisher_z_component_correlation": optional(fisher_r),
        "valid_correlation_components": int(valid_r.size),
        "component_mse": [optional(value) for value in component_mse],
        "component_explained_variance": [optional(value) for value in explained],
        "component_correlations": [optional(value) for value in correlations],
    }


def load_audited_speech_intervals(path: Path | None) -> tuple[tuple[float, float], ...] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"onset_seconds", "offset_seconds", "label_source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Speech mask TSV must contain {sorted(required)}")
    intervals = []
    allowed_sources = {"audio_manual", "audio_vad_audited"}
    for row in rows:
        if row["label_source"] not in allowed_sources:
            raise ValueError("Speech mask must be manually derived/audited from audio")
        onset = float(row["onset_seconds"])
        offset = float(row["offset_seconds"])
        if not np.isfinite(onset) or not np.isfinite(offset) or offset <= onset:
            raise ValueError("Speech mask contains an invalid interval")
        intervals.append((onset, offset))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("Speech-mask intervals overlap")
    return tuple(intervals)


def speech_mask_for_times(
    times: np.ndarray, intervals: Sequence[tuple[float, float]] | None
) -> np.ndarray | None:
    if intervals is None:
        return None
    result = np.zeros(len(times), dtype=bool)
    for onset, offset in intervals:
        result |= (times >= onset) & (times <= offset)
    return result


def run_matched_folds(
    blocks: Sequence[MatchedBlock],
    output_directory: Path,
    *,
    speech_intervals: Sequence[tuple[float, float]] | None = None,
    reducer_seed: int = 42,
    reduced_dimension: int = REDUCED_DIMENSION,
    subject: str = PILOT_SUBJECT,
) -> dict[str, Any]:
    if len(blocks) != BLOCK_COUNT or tuple(block.definition.index for block in blocks) != tuple(range(BLOCK_COUNT)):
        raise ValueError("Matched analysis requires ordered blocks 0..4")
    if reduced_dimension <= 0:
        raise ValueError("reduced_dimension must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, Any]] = []
    for test_index in range(BLOCK_COUNT):
        validation_index = (test_index + 1) % BLOCK_COUNT
        train_indexes = tuple(
            index for index in range(BLOCK_COUNT) if index not in (test_index, validation_index)
        )
        fold_dir = output_directory / f"fold_{test_index:02d}"
        if fold_dir.exists():
            raise FileExistsError(f"Fold output already exists: {fold_dir}")
        fold_dir.mkdir()

        train_ids, _, train_neural = _select(blocks, train_indexes)
        val_ids, val_times, val_neural = _select(blocks, (validation_index,))
        test_ids, test_times, test_neural = _select(blocks, (test_index,))
        neural_reducer = fit_train_only_reducer(
            train_neural,
            train_ids.tolist(),
            n_components=reduced_dimension,
            whiten=True,
            seed=reducer_seed,
            split_role="train",
        )
        neural_reducer.save(fold_dir / "neural_reducer")
        train_x = neural_reducer.transform(train_neural)
        val_x = neural_reducer.transform(val_neural)
        test_x = neural_reducer.transform(test_neural)
        fold_payload: dict[str, Any] = {
            "fold": test_index,
            "train_blocks": list(train_indexes),
            "validation_block": validation_index,
            "test_block": test_index,
            "train_frames": int(len(train_ids)),
            "validation_frames": int(len(val_ids)),
            "test_frames": int(len(test_ids)),
            "train_sample_ids_sha256": fingerprint_json(train_ids.tolist()),
            "validation_sample_ids_sha256": fingerprint_json(val_ids.tolist()),
            "test_sample_ids_sha256": fingerprint_json(test_ids.tolist()),
            "targets": {},
        }

        val_speech = speech_mask_for_times(val_times, speech_intervals)
        test_speech = speech_mask_for_times(test_times, speech_intervals)
        for target_name in TARGET_NAMES:
            train_y_raw = _target_select(blocks, train_indexes, target_name)
            val_y_raw = _target_select(blocks, (validation_index,), target_name)
            test_y_raw = _target_select(blocks, (test_index,), target_name)
            target_reducer = fit_train_only_reducer(
                train_y_raw,
                train_ids.tolist(),
                n_components=reduced_dimension,
                whiten=True,
                seed=reducer_seed,
                split_role="train",
            )
            target_reducer.save(fold_dir / f"{target_name}_target_reducer")
            train_y = target_reducer.transform(train_y_raw)
            val_y = target_reducer.transform(val_y_raw)
            test_y = target_reducer.transform(test_y_raw)
            estimator = LinearRegression(n_jobs=1).fit(train_x, train_y)
            val_prediction = estimator.predict(val_x).astype(np.float32)
            test_prediction = estimator.predict(test_x).astype(np.float32)

            model_path = fold_dir / f"{target_name}_linear_model.npz"
            with model_path.open("wb") as handle:
                np.savez(
                    handle,
                    coefficient=np.asarray(estimator.coef_, dtype=np.float64),
                    intercept=np.asarray(estimator.intercept_, dtype=np.float64),
                )
            prediction_path = fold_dir / f"{target_name}_test_predictions.npz"
            with prediction_path.open("wb") as handle:
                np.savez(
                    handle,
                    sample_ids=test_ids,
                    frame_times_seconds=test_times,
                    truth=test_y,
                    prediction=test_prediction,
                )
            fold_payload["targets"][target_name] = {
                "model_sha256": sha256_file(model_path),
                "test_predictions_sha256": sha256_file(prediction_path),
                "validation_all": regression_metrics(val_y, val_prediction),
                "test_all": regression_metrics(test_y, test_prediction),
                "validation_speech": regression_metrics(val_y, val_prediction, val_speech)
                if val_speech is not None
                else None,
                "test_speech": regression_metrics(test_y, test_prediction, test_speech)
                if test_speech is not None
                else None,
            }
        atomic_write_json(fold_dir / "fold_result.json", fold_payload, overwrite=False)
        fold_results.append(fold_payload)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_matched_linear_subject_result",
        "subject": subject,
        "confirmatory_subjects_read": subject != PILOT_SUBJECT,
        "target_dimension": reduced_dimension,
        "target_transform": f"fold-train StandardScaler + PCA{reduced_dimension} whitening",
        "neural_transform": f"shared fold-train StandardScaler + PCA{reduced_dimension} whitening",
        "linear_model": "ordinary least squares, identical hyperparameters per target",
        "frame_grid_ms": 20,
        "high_gamma_window_ms": 50,
        "offline_zero_phase_high_gamma": True,
        "offline_bidirectional_whisper_targets": True,
        "visual_events_used_only_for": "five adjacent 20-trial block boundaries",
        "visual_events_are_acoustic_onsets": False,
        "speech_mask_status": "audited audio mask supplied" if speech_intervals else "unavailable",
        "folds": fold_results,
        "aggregate_test": {},
    }
    for target_name in TARGET_NAMES:
        target_summary: dict[str, Any] = {}
        for scope in ("test_all", "test_speech"):
            values = [fold["targets"][target_name][scope] for fold in fold_results]
            values = [value for value in values if value is not None]
            if not values:
                target_summary[scope] = None
                continue
            target_summary[scope] = {}
            for metric in (
                "standardized_mse",
                "explained_variance",
                "fisher_z_component_correlation",
            ):
                metric_values = [value[metric] for value in values if value[metric] is not None]
                target_summary[scope][metric] = (
                    {
                        "mean": float(np.mean(metric_values)),
                        "sd": float(np.std(metric_values, ddof=1))
                        if len(metric_values) > 1
                        else 0.0,
                        "fold_count": len(metric_values),
                    }
                    if metric_values
                    else None
                )
        summary["aggregate_test"][target_name] = target_summary
    atomic_write_json(output_directory / "matched_linear_summary.json", summary, overwrite=False)
    return summary
