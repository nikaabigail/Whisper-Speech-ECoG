"""Full-neural SWPD sub-01 regression preparation on lagged one-second windows.

The same representation-independent neural preprocessing and the same
``OneSecondEcogEncoder`` input surface are used for MEL80 and Whisper
L3/L4/L5.  Visual word events define five non-overlapping blocks only.  They
are never interpreted as acoustic speech labels.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal

from ..integrity import atomic_write_json, fingerprint_json, read_json, sha256_file
from ..neural_data import (
    ChannelStandardizerArtifact,
    FrameWindowDataset,
    fit_train_only_channel_standardizer,
)
from ..protocol import SplitManifest, swpd_neural_pair_assignment
from ..reducer import ReducerArtifact, fit_train_only_reducer
from .matched_linear import TARGET_NAMES, VisualBlock
from .nwb import (
    PILOT_SUBJECT,
    SWPDInventory,
    SWPDRecording,
    assert_series_start_alignment,
    recording_relative_sample_bounds,
)


CANONICAL_NEURAL_RATE_HZ = 1000
WINDOW_SAMPLES = 1001
PRODUCTION_FRAME_HZ = 1000
FAST_SMOKE_FRAME_HZ = 50
EDGE_GUARD_SECONDS = 1.0
TARGET_DIMENSION = 50
AUTHOR_AUDIO_PROCESSING_RATE_HZ = 48_000


@dataclass(frozen=True)
class UsableChannels:
    indices: tuple[int, ...]
    names: tuple[str, ...]
    types: tuple[str, ...]
    statuses: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(
            {
                "indices": list(self.indices),
                "names": list(self.names),
                "types": list(self.types),
                "statuses": list(self.statuses),
            }
        )


def load_usable_channels(channels_tsv: Path, expected_rows: int) -> UsableChannels:
    """Pin all SEEG/ECOG rows not explicitly marked bad, preserving file order."""

    with Path(channels_tsv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != int(expected_rows):
        raise ValueError(
            f"channels.tsv has {len(rows)} rows but NWB has {expected_rows} channels"
        )
    selected: list[tuple[int, str, str, str]] = []
    for index, row in enumerate(rows):
        name = str(row.get("name", "")).strip()
        channel_type = str(row.get("type", "")).strip().upper()
        status = str(row.get("status", "")).strip().lower()
        if not name:
            raise ValueError(f"channel row {index} has no name")
        if channel_type not in {"SEEG", "ECOG"}:
            continue
        if status in {"bad", "exclude", "excluded", "noisy"}:
            continue
        if status not in {"", "n/a", "na", "good"}:
            raise ValueError(f"unrecognized channel status {status!r} for {name}")
        selected.append((index, name, channel_type, status or "unspecified"))
    if len(selected) < 2:
        raise ValueError("fewer than two usable intracranial channels remain")
    names = tuple(item[1] for item in selected)
    if len(names) != len(set(names)):
        raise ValueError("usable channel names are not unique")
    return UsableChannels(
        indices=tuple(item[0] for item in selected),
        names=names,
        types=tuple(item[2] for item in selected),
        statuses=tuple(item[3] for item in selected),
    )


@dataclass(frozen=True)
class SWPDNeuralPreprocessConfig:
    high_pass_hz: float = 10.0
    low_pass_hz: float = 200.0
    butterworth_order: int = 5
    notch_frequencies_hz: tuple[float, ...] = (50.0, 100.0, 150.0)
    notch_q: tuple[float, ...] = (25.0, 100.0, 150.0)
    output_rate_hz: int = CANONICAL_NEURAL_RATE_HZ
    common_average_reference: bool = False

    def __post_init__(self) -> None:
        if int(self.output_rate_hz) != CANONICAL_NEURAL_RATE_HZ:
            raise ValueError("SWPD full-neural output rate is fixed to exactly 1000 Hz")
        if bool(self.common_average_reference):
            raise ValueError("CAR is not present in the historical/SWPD baseline and is disabled")
        if int(self.butterworth_order) <= 0:
            raise ValueError("butterworth_order must be positive")
        if not 0 < self.high_pass_hz < self.low_pass_hz:
            raise ValueError("invalid neural band-pass limits")
        if len(self.notch_frequencies_hz) != len(self.notch_q):
            raise ValueError("each notch frequency requires one Q value")


class SWPDNeuralPreprocessor:
    """Fixed block-local filtering/resampling with no fitted split statistic."""

    def __init__(
        self,
        *,
        input_rate_hz: float,
        usable_channels: UsableChannels,
        config: SWPDNeuralPreprocessConfig = SWPDNeuralPreprocessConfig(),
    ) -> None:
        if not np.isfinite(input_rate_hz) or input_rate_hz <= 0:
            raise ValueError("input neural rate must be finite and positive")
        rounded = int(round(float(input_rate_hz)))
        if abs(float(input_rate_hz) - rounded) > 1e-9:
            raise ValueError("SWPD neural rate must be an exact integer for pinned resampling")
        if config.low_pass_hz >= rounded / 2:
            raise ValueError("low-pass must be below the native Nyquist frequency")
        self.input_rate_hz = rounded
        self.usable_channels = usable_channels
        self.config = config
        common = math.gcd(self.input_rate_hz, config.output_rate_hz)
        self.resample_up = config.output_rate_hz // common
        self.resample_down = self.input_rate_hz // common

    def transform(self, raw_block: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_block, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("raw neural block must have shape (time, channels)")
        if values.shape[1] <= max(self.usable_channels.indices):
            raise ValueError("raw neural block has fewer channels than the pinned selection")
        values = values[:, self.usable_channels.indices]
        if not np.isfinite(values).all():
            raise ValueError("raw neural block contains NaN or Infinity")

        # No CAR and no per-block/whole-recording z-score.  Filters are applied
        # independently inside each visual block, so no split boundary is crossed.
        filtered = values
        for frequency, q_value in zip(
            self.config.notch_frequencies_hz, self.config.notch_q
        ):
            numerator, denominator = signal.iirnotch(
                float(frequency), Q=float(q_value), fs=self.input_rate_hz
            )
            filtered = signal.filtfilt(numerator, denominator, filtered, axis=0)
        band = signal.butter(
            int(self.config.butterworth_order),
            (float(self.config.high_pass_hz), float(self.config.low_pass_hz)),
            btype="bandpass",
            fs=self.input_rate_hz,
            output="sos",
        )
        filtered = signal.sosfiltfilt(band, filtered, axis=0)
        resampled = signal.resample_poly(
            filtered,
            up=self.resample_up,
            down=self.resample_down,
            axis=0,
        )
        if not np.isfinite(resampled).all():
            raise FloatingPointError("SWPD neural preprocessing produced NaN or Infinity")
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "swpd_fixed_classic_neural_preprocessing",
            "input_rate_hz": self.input_rate_hz,
            "output_rate_hz": self.config.output_rate_hz,
            "usable_channel_count": len(self.usable_channels.indices),
            "usable_channel_indices": list(self.usable_channels.indices),
            "usable_channel_names_sha256": fingerprint_json(
                list(self.usable_channels.names)
            ),
            "channel_selection_fingerprint": self.usable_channels.fingerprint,
            "channel_selection": (
                "channels.tsv SEEG/ECOG rows in file order, excluding only explicit bad flags"
            ),
            "steps": [
                "zero_phase_iir_notches_at_native_rate",
                "zero_phase_butterworth_band_pass_at_native_rate",
                "deterministic_polyphase_resample_to_1000_hz",
            ],
            "notch_frequencies_hz": list(self.config.notch_frequencies_hz),
            "notch_q": list(self.config.notch_q),
            "high_pass_hz": self.config.high_pass_hz,
            "low_pass_hz": self.config.low_pass_hz,
            "butterworth_order": self.config.butterworth_order,
            "resample_up": self.resample_up,
            "resample_down": self.resample_down,
            "common_average_reference": False,
            "per_block_z_score": False,
            "whole_recording_z_score": False,
            "fitted_parameters": False,
            "downstream_scaling": "training-block-only channel standardizer",
            "representation_independent": True,
            "block_local_no_split_crossing": True,
            "filter_direction": "forward_backward_zero_phase_offline",
            "online_causality_claimed": False,
        }


@dataclass(frozen=True)
class NeuralTargetBlock:
    definition: VisualBlock
    trial_id: str
    trial_start_seconds: float
    raw_1000hz: np.ndarray
    sample_ids: np.ndarray
    frame_times_seconds: np.ndarray
    targets: Mapping[str, np.ndarray]
    extraction_fingerprint: str

    def close(self) -> None:
        """Release a read-only NumPy memory map deterministically on Windows."""

        if isinstance(self.raw_1000hz, np.memmap):
            memory_map = getattr(self.raw_1000hz, "_mmap", None)
            if memory_map is not None:
                memory_map.close()


def _validate_target_arrays(targets: Mapping[str, np.ndarray], frame_count: int) -> None:
    if tuple(targets) != TARGET_NAMES:
        raise ValueError(f"target order must be {TARGET_NAMES}")
    dimensions = {"mel80": 80, "L3": 512, "L4": 512, "L5": 512}
    for name, values in targets.items():
        if np.asarray(values).shape != (frame_count, dimensions[name]):
            raise ValueError(f"invalid {name} target shape: {np.asarray(values).shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} target contains NaN or Infinity")


def extract_neural_target_block(
    recording: SWPDRecording,
    inventory: SWPDInventory,
    definition: VisualBlock,
    *,
    preprocessor: SWPDNeuralPreprocessor,
    mel_extractor: Any,
    whisper_extractor: Any,
    extraction_fingerprint: str,
    frame_hz: int = PRODUCTION_FRAME_HZ,
) -> NeuralTargetBlock:
    """Extract one block; each acoustic target sees the identical neural array."""

    if int(frame_hz) not in {PRODUCTION_FRAME_HZ, FAST_SMOKE_FRAME_HZ}:
        raise ValueError("frame_hz must be production 1000 Hz or explicit fast-smoke 50 Hz")
    assert_series_start_alignment(inventory)
    ieeg_bounds = recording_relative_sample_bounds(
        definition.start_seconds, definition.stop_seconds, inventory.ieeg
    )
    raw_native = recording.read_ieeg(
        ieeg_bounds.start_index, ieeg_bounds.stop_index
    )
    raw_1000 = preprocessor.transform(raw_native)
    hop_samples = CANONICAL_NEURAL_RATE_HZ // int(frame_hz)
    first_end = int(round(EDGE_GUARD_SECONDS * CANONICAL_NEURAL_RATE_HZ))
    last_end_exclusive = len(raw_1000) - int(
        round(EDGE_GUARD_SECONDS * CANONICAL_NEURAL_RATE_HZ)
    )
    end_indices = np.arange(first_end, last_end_exclusive, hop_samples, dtype=np.int64)
    if end_indices.size <= TARGET_DIMENSION:
        raise RuntimeError(f"block {definition.index} has too few valid 1000 Hz windows")
    frame_times = (
        ieeg_bounds.actual_start_recording_relative_seconds
        + end_indices / CANONICAL_NEURAL_RATE_HZ
    )

    audio_bounds = recording_relative_sample_bounds(
        definition.start_seconds, definition.stop_seconds, inventory.audio
    )
    audio = recording.read_audio(
        audio_bounds.start_index, audio_bounds.stop_index
    ).astype(np.float32, copy=False)
    target_times_local = (
        frame_times - audio_bounds.actual_start_recording_relative_seconds
    )
    if target_times_local[0] < 0:
        raise RuntimeError("neural target time precedes selected audio block")
    mel = np.asarray(
        mel_extractor.extract_aligned(
            audio, AUTHOR_AUDIO_PROCESSING_RATE_HZ, target_times_local
        ),
        dtype=np.float32,
    )
    whisper = whisper_extractor.extract_aligned(
        audio, AUTHOR_AUDIO_PROCESSING_RATE_HZ, target_times_local
    )
    targets = {
        "mel80": mel,
        "L3": np.asarray(whisper[3], dtype=np.float32),
        "L4": np.asarray(whisper[4], dtype=np.float32),
        "L5": np.asarray(whisper[5], dtype=np.float32),
    }
    _validate_target_arrays(targets, len(end_indices))
    sample_ids = np.asarray(
        [
            f"{PILOT_SUBJECT}:block-{definition.index:02d}:grid-{frame_hz:03d}hz:end-{end:06d}"
            for end in end_indices
        ],
        dtype="U80",
    )
    return NeuralTargetBlock(
        definition=definition,
        trial_id=f"{PILOT_SUBJECT}:visual-block-{definition.index:02d}",
        trial_start_seconds=ieeg_bounds.actual_start_recording_relative_seconds,
        raw_1000hz=raw_1000,
        sample_ids=sample_ids,
        frame_times_seconds=np.asarray(frame_times, dtype=np.float64),
        targets=targets,
        extraction_fingerprint=extraction_fingerprint,
    )


def build_neural_extraction_fingerprint(
    *,
    inventory: SWPDInventory,
    events_path: Path,
    dataset_manifest_sha256: str,
    definition: VisualBlock,
    preprocessing_provenance: Mapping[str, Any],
    mel_provenance: Mapping[str, Any],
    whisper_provenance: Mapping[str, Any],
    frame_hz: int,
    source_fingerprint: str,
) -> str:
    return fingerprint_json(
        {
            "schema_version": 1,
            "implementation": "swpd-sub01-full-neural-cache-v1",
            "implementation_sha256": sha256_file(Path(__file__)),
            "source_fingerprint": str(source_fingerprint),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "nwb_path": inventory.nwb_path,
            "nwb_size_bytes": inventory.nwb_size_bytes,
            "events_sha256": sha256_file(events_path),
            "definition": asdict(definition),
            "time_coordinate": "events_tsv_recording_relative_seconds",
            "stream_start_offsets_seconds": assert_series_start_alignment(inventory),
            "frame_hz": int(frame_hz),
            "window_lag_ms": [-1000, 0],
            "window_samples_inclusive": WINDOW_SAMPLES,
            "edge_guard_seconds_each_block_side": EDGE_GUARD_SECONDS,
            "preprocessing": dict(preprocessing_provenance),
            "mel": dict(mel_provenance),
            "whisper": dict(whisper_provenance),
        }
    )


def _cache_paths(cache_directory: Path, index: int) -> tuple[Path, Path, Path]:
    stem = f"neural_block_{index:02d}"
    return (
        Path(cache_directory) / f"{stem}_1000hz.npy",
        Path(cache_directory) / f"{stem}_targets.npz",
        Path(cache_directory) / f"{stem}.json",
    )


def save_neural_block_cache(block: NeuralTargetBlock, cache_directory: Path) -> Path:
    raw_path, target_path, manifest_path = _cache_paths(
        Path(cache_directory), block.definition.index
    )
    if any(path.exists() for path in (raw_path, target_path, manifest_path)):
        raise FileExistsError(f"neural block cache already exists for {block.definition.index}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_temporary = raw_path.with_name(raw_path.name + ".partial")
    target_temporary = target_path.with_name(target_path.name + ".partial")
    try:
        with raw_temporary.open("wb") as handle:
            np.save(handle, block.raw_1000hz, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw_temporary, raw_path)
        with target_temporary.open("wb") as handle:
            np.savez(
                handle,
                sample_ids=block.sample_ids,
                frame_times_seconds=block.frame_times_seconds,
                mel80=block.targets["mel80"],
                L3=block.targets["L3"],
                L4=block.targets["L4"],
                L5=block.targets["L5"],
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(target_temporary, target_path)
    finally:
        for path in (raw_temporary, target_temporary):
            if path.exists():
                path.unlink()
    payload = {
        "schema_version": 1,
        "kind": "swpd_sub01_full_neural_block_cache",
        "extraction_fingerprint": block.extraction_fingerprint,
        "definition": asdict(block.definition),
        "trial_id": block.trial_id,
        "trial_start_seconds": block.trial_start_seconds,
        "raw_file": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
        "target_file": target_path.name,
        "target_sha256": sha256_file(target_path),
        "sample_count": int(len(block.sample_ids)),
        "neural_samples": int(block.raw_1000hz.shape[0]),
        "channel_count": int(block.raw_1000hz.shape[1]),
        "representation_independent_raw_cache": True,
    }
    payload["fingerprint"] = fingerprint_json(payload)
    atomic_write_json(manifest_path, payload, overwrite=False)
    return manifest_path


def load_neural_block_cache(
    cache_directory: Path,
    index: int,
    *,
    extraction_fingerprint: str,
) -> NeuralTargetBlock | None:
    raw_path, target_path, manifest_path = _cache_paths(Path(cache_directory), index)
    present = [path.exists() for path in (raw_path, target_path, manifest_path)]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"incomplete neural block cache for block {index}")
    payload = read_json(manifest_path)
    stored = payload.pop("fingerprint", None)
    if stored != fingerprint_json(payload):
        raise RuntimeError(f"neural block manifest was modified: {manifest_path}")
    if payload.get("extraction_fingerprint") != extraction_fingerprint:
        raise RuntimeError(f"neural block fingerprint mismatch: {manifest_path}")
    if sha256_file(raw_path) != payload.get("raw_sha256"):
        raise RuntimeError(f"neural block checksum mismatch: {raw_path}")
    if sha256_file(target_path) != payload.get("target_sha256"):
        raise RuntimeError(f"target block checksum mismatch: {target_path}")
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    with np.load(target_path, allow_pickle=False) as arrays:
        sample_ids = np.asarray(arrays["sample_ids"])
        times = np.asarray(arrays["frame_times_seconds"], dtype=np.float64)
        targets = {
            name: np.asarray(arrays[name], dtype=np.float32) for name in TARGET_NAMES
        }
    definition = VisualBlock(**payload["definition"])
    _validate_target_arrays(targets, len(sample_ids))
    if raw.shape != (int(payload["neural_samples"]), int(payload["channel_count"])):
        raise RuntimeError("cached neural array shape differs from its manifest")
    if len(sample_ids) != int(payload["sample_count"]):
        raise RuntimeError("cached target count differs from its manifest")
    return NeuralTargetBlock(
        definition=definition,
        trial_id=str(payload["trial_id"]),
        trial_start_seconds=float(payload["trial_start_seconds"]),
        raw_1000hz=raw,
        sample_ids=sample_ids,
        frame_times_seconds=times,
        targets=targets,
        extraction_fingerprint=extraction_fingerprint,
    )


@dataclass(frozen=True)
class NeuralSplitPlan:
    train_blocks: tuple[int, ...]
    validation_block: int
    test_block: int
    manifest: SplitManifest


def make_sub01_neural_split(
    blocks: Sequence[NeuralTargetBlock], *, dataset_manifest_sha256: str
) -> NeuralSplitPlan:
    if len(blocks) != 5 or tuple(block.definition.index for block in blocks) != tuple(range(5)):
        raise ValueError("SWPD sub-01 neural pilot requires ordered blocks 0..4")
    assignment = swpd_neural_pair_assignment(1)

    def ids(indexes: Sequence[int]) -> list[str]:
        return [
            str(sample_id)
            for index in indexes
            for sample_id in blocks[index].sample_ids.tolist()
        ]

    manifest = SplitManifest.create(
        dataset_id="swpd-sub-01",
        protocol_id=(
            "swpd-full-neural-five-visual-blocks-v1/"
            f"test-block-{assignment.test_pair_index}"
        ),
        split_seed=0,
        train_ids=ids(assignment.training_pair_indices),
        validation_ids=ids((assignment.validation_pair_index,)),
        test_ids=ids((assignment.test_pair_index,)),
        purge_gap_seconds=EDGE_GUARD_SECONDS,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    return NeuralSplitPlan(
        train_blocks=assignment.training_pair_indices,
        validation_block=assignment.validation_pair_index,
        test_block=assignment.test_pair_index,
        manifest=manifest,
    )


def fit_or_load_channel_standardizer(
    blocks: Sequence[NeuralTargetBlock],
    train_indexes: Sequence[int],
    artifact_directory: Path,
) -> ChannelStandardizerArtifact:
    path = Path(artifact_directory)
    if path.exists():
        loaded = ChannelStandardizerArtifact.load(path)
        expected = fingerprint_json([blocks[index].trial_id for index in train_indexes])
        if loaded.train_trial_ids_sha256 != expected:
            raise RuntimeError("channel standardizer was fitted on different blocks")
        return loaded
    fitted = fit_train_only_channel_standardizer(
        [blocks[index].raw_1000hz for index in train_indexes],
        [blocks[index].trial_id for index in train_indexes],
        split_role="train",
    )
    fitted.save(path)
    return fitted


def fit_or_load_target_reducer(
    blocks: Sequence[NeuralTargetBlock],
    train_indexes: Sequence[int],
    target_name: str,
    artifact_directory: Path,
    *,
    seed: int,
) -> ReducerArtifact:
    if target_name not in TARGET_NAMES:
        raise ValueError(f"unknown target {target_name}")
    train_ids = np.concatenate([blocks[index].sample_ids for index in train_indexes])
    path = Path(artifact_directory)
    if path.exists():
        loaded = ReducerArtifact.load(path)
        if loaded.train_sample_ids_sha256 != fingerprint_json(train_ids.tolist()):
            raise RuntimeError(f"{target_name} reducer was fitted on different frames")
        return loaded
    features = np.concatenate(
        [blocks[index].targets[target_name] for index in train_indexes], axis=0
    )
    fitted = fit_train_only_reducer(
        features,
        train_ids.tolist(),
        n_components=TARGET_DIMENSION,
        whiten=True,
        seed=int(seed),
        split_role="train",
    )
    fitted.save(path)
    return fitted


def standardize_blocks_once(
    blocks: Sequence[NeuralTargetBlock],
    indexes: Sequence[int],
    standardizer: ChannelStandardizerArtifact,
) -> dict[str, np.ndarray]:
    """Apply train-only channel statistics once, not per overlapping window."""

    return {
        blocks[index].trial_id: standardizer.transform(blocks[index].raw_1000hz)
        for index in indexes
    }


def make_window_dataset(
    blocks: Sequence[NeuralTargetBlock],
    indexes: Sequence[int],
    *,
    target_name: str,
    target_reducer: ReducerArtifact,
    standardized_trials: Mapping[str, np.ndarray],
    split_role: str,
    split_fingerprint: str | None = None,
) -> FrameWindowDataset:
    selected = [blocks[index] for index in indexes]
    sample_ids = np.concatenate([block.sample_ids for block in selected])
    times = np.concatenate([block.frame_times_seconds for block in selected])
    raw_targets = np.concatenate([block.targets[target_name] for block in selected], axis=0)
    targets = target_reducer.transform(raw_targets)
    trial_ids = tuple(
        block.trial_id for block in selected for _ in range(len(block.sample_ids))
    )
    starts = {block.trial_id: block.trial_start_seconds for block in selected}
    trials = {block.trial_id: standardized_trials[block.trial_id] for block in selected}
    return FrameWindowDataset(
        trials=trials,
        sample_ids=sample_ids.tolist(),
        frame_trial_ids=trial_ids,
        frame_times_s=times,
        targets=targets,
        split_role=split_role,
        sample_rate_hz=CANONICAL_NEURAL_RATE_HZ,
        window_samples=WINDOW_SAMPLES,
        trial_start_times_s=starts,
        standardizer=None,
        split_fingerprint=split_fingerprint,
    )
