"""Leakage-controlled neural standardization and lagged one-second windows.

Window indexes stop at prediction time. Dataset-specific zero-phase filters may
still make the complete offline pipeline noncausal and must be disclosed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .integrity import fingerprint_json, read_json, sha256_file


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_ROLES = frozenset({"train", "validation", "held_out_test"})


def _unique_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class ChannelStandardizerArtifact:
    """Portable per-channel statistics fitted only on neural training trials."""

    mean: np.ndarray
    scale: np.ndarray
    variance: np.ndarray
    sample_count: int
    train_trial_ids_sha256: str

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in (
            self.mean,
            self.scale,
            self.variance,
        ))
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("standardizer arrays must be one-dimensional")
        if not arrays[0].size or any(value.shape != arrays[0].shape for value in arrays):
            raise ValueError("standardizer arrays must have the same non-empty shape")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("standardizer arrays must be finite")
        if np.any(arrays[1] <= 0) or np.any(arrays[2] < 0):
            raise ValueError("standardizer scales/variances are invalid")
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be positive")
        if not _SHA256.fullmatch(str(self.train_trial_ids_sha256)):
            raise ValueError("train_trial_ids_sha256 must be a lowercase SHA256")
        for field_name, value in zip(("mean", "scale", "variance"), arrays):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, field_name, frozen)

    @property
    def channel_count(self) -> int:
        return int(self.mean.size)

    def transform(self, trial: np.ndarray) -> np.ndarray:
        values = np.asarray(trial, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channel_count:
            raise ValueError(
                f"neural trial must have shape (time, {self.channel_count}); got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("neural trial contains NaN or Infinity")
        return ((values - self.mean) / self.scale).astype(np.float32)

    def manifest_payload(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "train_only_neural_channel_standardizer",
            "channel_count": self.channel_count,
            "sample_count": int(self.sample_count),
            "train_trial_ids_sha256": self.train_trial_ids_sha256,
            "zero_variance_channel_indices": np.flatnonzero(self.variance == 0).tolist(),
        }

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"Channel standardizer already exists: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{directory.name}.partial-", dir=directory.parent)
        )
        try:
            arrays_path = temporary / "channel_standardizer_arrays.npz"
            np.savez_compressed(
                arrays_path,
                mean=self.mean,
                scale=self.scale,
                variance=self.variance,
            )
            payload = self.manifest_payload()
            payload["arrays_file"] = arrays_path.name
            payload["arrays_sha256"] = sha256_file(arrays_path)
            payload["fingerprint"] = fingerprint_json(payload)
            (temporary / "manifest.json").write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            if directory.exists():
                raise FileExistsError(f"Channel standardizer already exists: {directory}")
            os.rename(temporary, directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return directory

    @classmethod
    def load(cls, directory: Path) -> "ChannelStandardizerArtifact":
        directory = Path(directory)
        payload = read_json(directory / "manifest.json")
        fingerprint = payload.pop("fingerprint", None)
        if fingerprint != fingerprint_json(payload):
            raise RuntimeError(f"Channel-standardizer manifest mismatch: {directory}")
        if payload.get("kind") != "train_only_neural_channel_standardizer":
            raise RuntimeError(f"Unexpected standardizer artifact kind: {directory}")
        arrays_path = directory / str(payload["arrays_file"])
        if sha256_file(arrays_path) != payload.get("arrays_sha256"):
            raise RuntimeError(f"Channel-standardizer payload checksum mismatch: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            artifact = cls(
                mean=arrays["mean"],
                scale=arrays["scale"],
                variance=arrays["variance"],
                sample_count=int(payload["sample_count"]),
                train_trial_ids_sha256=str(payload["train_trial_ids_sha256"]),
            )
        if artifact.channel_count != int(payload["channel_count"]):
            raise RuntimeError("Channel count disagrees with standardizer manifest")
        return artifact


def fit_train_only_channel_standardizer(
    trials: Sequence[np.ndarray],
    trial_ids: Sequence[str],
    *,
    split_role: str,
) -> ChannelStandardizerArtifact:
    """Fit population mean/variance over time, without materializing a concatenation."""

    if split_role != "train":
        raise ValueError("Channel-standardizer fitting requires split_role='train'")
    ids = _unique_ids(trial_ids, name="training trial IDs")
    if len(trials) != len(ids):
        raise ValueError("one training trial ID is required for every trial array")

    count = 0
    mean: np.ndarray | None = None
    squared_deviation: np.ndarray | None = None
    for trial_id, trial in zip(ids, trials):
        values = np.asarray(trial, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"training trial {trial_id!r} must have shape (time, channels)")
        if not np.isfinite(values).all():
            raise ValueError(f"training trial {trial_id!r} contains NaN or Infinity")
        if mean is not None and values.shape[1] != mean.size:
            raise ValueError("all neural trials must have the same channel count")

        batch_count = int(values.shape[0])
        batch_mean = values.mean(axis=0, dtype=np.float64)
        centered = values - batch_mean
        batch_squared_deviation = np.sum(centered * centered, axis=0, dtype=np.float64)
        if mean is None:
            mean = batch_mean
            squared_deviation = batch_squared_deviation
            count = batch_count
            continue
        assert squared_deviation is not None
        delta = batch_mean - mean
        combined_count = count + batch_count
        squared_deviation = (
            squared_deviation
            + batch_squared_deviation
            + delta * delta * count * batch_count / combined_count
        )
        mean = mean + delta * batch_count / combined_count
        count = combined_count

    if mean is None or squared_deviation is None:
        raise ValueError("at least one non-empty training trial is required")
    variance = np.maximum(squared_deviation / count, 0.0)
    scale = np.sqrt(variance)
    scale[variance <= np.finfo(np.float64).eps] = 1.0
    return ChannelStandardizerArtifact(
        mean=mean,
        scale=scale,
        variance=variance,
        sample_count=count,
        train_trial_ids_sha256=fingerprint_json(list(ids)),
    )


@dataclass(frozen=True)
class WindowRecord:
    sample_id: str
    trial_id: str
    frame_time_s: float
    start_index: int
    end_index: int


class FrameWindowDataset(Dataset):
    """Slice inclusive lag[-1000 ms, 0] windows from individual whole trials.

    Frame times are relative to each trial unless ``trial_start_times_s`` is
    provided. Every frame is validated at construction. A frame that would need
    padding or touch another trial is rejected rather than silently modified.
    """

    def __init__(
        self,
        *,
        trials: Mapping[str, np.ndarray],
        sample_ids: Sequence[str],
        frame_trial_ids: Sequence[str],
        frame_times_s: Sequence[float],
        targets: np.ndarray,
        split_role: str,
        sample_rate_hz: int = 1000,
        window_samples: int = 1001,
        window_end_offset_samples: int = 0,
        alignment_tolerance_samples: float = 0.05,
        trial_start_times_s: Mapping[str, float] | None = None,
        standardizer: ChannelStandardizerArtifact | None = None,
        standardization_mode: str = "pretransform_trials",
        split_fingerprint: str | None = None,
    ) -> None:
        if split_role not in _SPLIT_ROLES:
            raise ValueError(f"split_role must be one of {sorted(_SPLIT_ROLES)}")
        if int(sample_rate_hz) != 1000:
            raise ValueError("OneSecondEcogEncoder requires neural data at exactly 1000 Hz")
        if int(window_samples) != 1001:
            raise ValueError("the inclusive one-second input contract requires 1001 samples")
        if not np.isfinite(alignment_tolerance_samples) or not 0 <= alignment_tolerance_samples < 0.5:
            raise ValueError("alignment_tolerance_samples must be finite and in [0, 0.5)")
        normalized_split_fingerprint = (
            str(split_fingerprint).lower() if split_fingerprint is not None else None
        )
        if normalized_split_fingerprint is not None and not _SHA256.fullmatch(
            normalized_split_fingerprint
        ):
            raise ValueError("split_fingerprint must be a lowercase SHA256")
        if split_role == "held_out_test" and normalized_split_fingerprint is None:
            raise ValueError("held_out_test datasets require their immutable split_fingerprint")
        if standardization_mode not in {"pretransform_trials", "per_window"}:
            raise ValueError(
                "standardization_mode must be 'pretransform_trials' or 'per_window'"
            )
        ids = _unique_ids(sample_ids, name="frame sample IDs")
        trial_ids = tuple(str(value) for value in frame_trial_ids)
        times = np.asarray(frame_times_s, dtype=np.float64)
        target_values = np.asarray(targets)
        if not (len(ids) == len(trial_ids) == len(times) == len(target_values)):
            raise ValueError("frame IDs, trial IDs, times and targets must have equal length")
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("frame times must be a finite one-dimensional sequence")
        if target_values.ndim != 2 or target_values.shape[1] == 0:
            raise ValueError("regression targets must have shape (frames, target_features)")
        if not np.isfinite(target_values).all():
            raise ValueError("regression targets contain NaN or Infinity")

        normalized_trials: dict[str, np.ndarray] = {}
        channel_count: int | None = None
        for trial_id, trial in trials.items():
            key = str(trial_id)
            if key in normalized_trials:
                raise ValueError(f"duplicate neural trial ID after string conversion: {key}")
            values = np.asarray(trial)
            if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
                raise ValueError(f"neural trial {key!r} must have shape (time, channels)")
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                raise ValueError(f"neural trial {key!r} must contain finite numeric values")
            if channel_count is None:
                channel_count = int(values.shape[1])
            elif int(values.shape[1]) != channel_count:
                raise ValueError("all neural trials must have the same channel count")
            normalized_trials[key] = values
        if not normalized_trials:
            raise ValueError("at least one whole neural trial is required")
        assert channel_count is not None
        if standardizer is not None and standardizer.channel_count != channel_count:
            raise ValueError("standardizer channel count differs from neural trials")
        if standardizer is not None and standardization_mode == "pretransform_trials":
            transformed_trials: dict[str, np.ndarray] = {}
            for trial_id, values in normalized_trials.items():
                transformed = np.ascontiguousarray(
                    standardizer.transform(values), dtype=np.float32
                )
                transformed.setflags(write=False)
                transformed_trials[trial_id] = transformed
            normalized_trials = transformed_trials

        starts = {str(key): float(value) for key, value in (trial_start_times_s or {}).items()}
        if not all(np.isfinite(value) for value in starts.values()):
            raise ValueError("trial start times must be finite")
        records: list[WindowRecord] = []
        for sample_id, trial_id, frame_time in zip(ids, trial_ids, times):
            if trial_id not in normalized_trials:
                raise ValueError(f"frame {sample_id!r} refers to unknown trial {trial_id!r}")
            relative_sample = (float(frame_time) - starts.get(trial_id, 0.0)) * sample_rate_hz
            frame_index = int(round(relative_sample))
            if abs(relative_sample - frame_index) > float(alignment_tolerance_samples):
                raise ValueError(f"frame {sample_id!r} cannot be aligned to the 1000 Hz grid")
            end_index = frame_index + int(window_end_offset_samples)
            start_index = end_index - int(window_samples) + 1
            trial_length = int(normalized_trials[trial_id].shape[0])
            if start_index < 0 or end_index >= trial_length:
                raise ValueError(
                    f"frame {sample_id!r} in trial {trial_id!r} would require boundary "
                    f"padding ({start_index}:{end_index}, trial length {trial_length})"
                )
            records.append(
                WindowRecord(
                    sample_id=sample_id,
                    trial_id=trial_id,
                    frame_time_s=float(frame_time),
                    start_index=start_index,
                    end_index=end_index,
                )
            )

        self.trials = normalized_trials
        self.records = tuple(records)
        self.sample_ids = ids
        self.targets = np.asarray(target_values, dtype=np.float32)
        self.split_role = split_role
        self.sample_rate_hz = int(sample_rate_hz)
        self.window_samples = int(window_samples)
        self.window_end_offset_samples = int(window_end_offset_samples)
        self.alignment_tolerance_samples = float(alignment_tolerance_samples)
        self.input_channels = channel_count
        self.target_dim = int(self.targets.shape[1])
        self.standardizer = standardizer
        self.standardization_mode = (
            standardization_mode if standardizer is not None else "none"
        )
        self.split_fingerprint = normalized_split_fingerprint

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[int(index)]
        window = self.trials[record.trial_id][record.start_index : record.end_index + 1]
        if int(window.shape[0]) != self.window_samples:
            raise RuntimeError("validated neural window changed size after dataset construction")
        if self.standardizer is not None and self.standardization_mode == "per_window":
            window = self.standardizer.transform(window)
        else:
            window = np.asarray(window, dtype=np.float32)
        return {
            "inputs": torch.from_numpy(np.ascontiguousarray(window.T)),
            "target": torch.from_numpy(np.array(self.targets[index], copy=True)),
            "sample_id": record.sample_id,
            "trial_id": record.trial_id,
            "frame_time_s": torch.tensor(record.frame_time_s, dtype=torch.float64),
        }

    def storage_receipt(self) -> dict:
        """Describe bounded storage; no materialized overlapping windows are retained."""

        stored_bytes = int(sum(value.nbytes for value in self.trials.values()))
        stored_samples = int(sum(value.shape[0] for value in self.trials.values()))
        return {
            "schema_version": 1,
            "kind": "whole_trial_window_dataset_storage",
            "trial_count": len(self.trials),
            "stored_time_samples": stored_samples,
            "stored_bytes": stored_bytes,
            "stored_dtype": str(next(iter(self.trials.values())).dtype),
            "materialized_window_count": 0,
            "window_samples": self.window_samples,
            "standardization_mode": self.standardization_mode,
            "per_item_standardization": self.standardization_mode == "per_window",
            "train_standardizer_trial_ids_sha256": (
                self.standardizer.train_trial_ids_sha256
                if self.standardizer is not None
                else None
            ),
        }
