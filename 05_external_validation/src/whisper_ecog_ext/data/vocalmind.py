"""Strict read-only adapter and split indexer for VocalMind overt words.

The primary protocol uses repetitions 1--5 only. For outer fold ``k``, repetition
``k`` is test, the next repetition cyclically is validation, and the other three
are training data. Repetition 6 is never allowed into a primary partition.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
import wave

import numpy as np


ADAPTER_VERSION = "vocalmind_raw_word_v1"
INDEX_SCHEMA_VERSION = 1
PRIMARY_SPLIT_POLICY = "repetitions_1_5_cyclic_train3_val1_test1_v1"
_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_AUDIO_NAME_RE = re.compile(r"^Audio_(?P<word>[A-Za-z][A-Za-z0-9]*)_(?P<rep>[1-9]\d*)\.wav$")
_EEG_NAME_RE = re.compile(
    r"^Vocalized_(?P<word>[A-Za-z][A-Za-z0-9]*)_(?P<rep>[1-9]\d*)\.csv$"
)


EXPECTED_WORDS = (
    "DaNao",
    "DiTie",
    "JiaTing",
    "KaFei",
    "MianBao",
    "MiFan",
    "NiuNai",
    "PengYou",
    "PingGuo",
    "QiChe",
    "QingCai",
    "ShouJi",
    "ShuMu",
    "WuDao",
    "XiongMao",
    "XueXiao",
    "YaChi",
    "YinHang",
    "YinYue",
    "ZhongGuo",
)

EXPECTED_CHANNEL_IDS = (
    9,
    10,
    11,
    12,
    13,
    14,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
)


class VocalMindDataError(RuntimeError):
    """The on-disk VocalMind release violates the pinned data contract."""


@dataclass(frozen=True)
class SourceArchiveSpec:
    name: str
    size_bytes: int
    md5: str
    extracted_member_count: int

    @property
    def extraction_directory(self) -> str:
        return Path(self.name).stem


@dataclass(frozen=True)
class VocalMindContract:
    words: tuple[str, ...]
    channel_ids: tuple[int, ...]
    eeg_sample_rate_hz: int
    eeg_samples: int
    audio_sample_rate_hz: int
    audio_frames: int
    audio_channels: int
    audio_sample_width_bytes: int
    primary_repetitions: tuple[int, ...] = (1, 2, 3, 4, 5)
    secondary_repetition: int = 6
    expected_missing: frozenset[tuple[str, int]] = field(default_factory=frozenset)
    source_archives: tuple[SourceArchiveSpec, ...] = ()

    def __post_init__(self) -> None:
        if len(self.words) == 0 or len(set(self.words)) != len(self.words):
            raise ValueError("words must be non-empty and unique")
        if any(not _TOKEN_RE.fullmatch(word) for word in self.words):
            raise ValueError("Every word must be a filename-safe ASCII token")
        if len(self.channel_ids) == 0 or len(set(self.channel_ids)) != len(self.channel_ids):
            raise ValueError("channel_ids must be non-empty and unique")
        if self.primary_repetitions != (1, 2, 3, 4, 5):
            raise ValueError("The frozen primary contract requires repetitions 1--5")
        if self.secondary_repetition in self.primary_repetitions:
            raise ValueError("secondary_repetition cannot be primary")
        if min(
            self.eeg_sample_rate_hz,
            self.eeg_samples,
            self.audio_sample_rate_hz,
            self.audio_frames,
            self.audio_channels,
            self.audio_sample_width_bytes,
        ) <= 0:
            raise ValueError("Rates, shapes, and sample width must be positive")
        if self.eeg_samples * self.audio_sample_rate_hz != (
            self.audio_frames * self.eeg_sample_rate_hz
        ):
            raise ValueError("EEG and audio durations must be identical")
        allowed_missing = {
            (word, self.secondary_repetition)
            for word in self.words
        }
        if not self.expected_missing.issubset(allowed_missing):
            raise ValueError("Only secondary-repetition trials may be declared missing")

    @property
    def expected_keys(self) -> frozenset[tuple[str, int]]:
        repetitions = (*self.primary_repetitions, self.secondary_repetition)
        return frozenset(
            (word, repetition)
            for word in self.words
            for repetition in repetitions
            if (word, repetition) not in self.expected_missing
        )

    @property
    def duration_seconds(self) -> float:
        return self.eeg_samples / self.eeg_sample_rate_hz


DEFAULT_VOCALMIND_CONTRACT = VocalMindContract(
    words=EXPECTED_WORDS,
    channel_ids=EXPECTED_CHANNEL_IDS,
    eeg_sample_rate_hz=1000,
    eeg_samples=3000,
    audio_sample_rate_hz=44_100,
    audio_frames=132_300,
    audio_channels=2,
    audio_sample_width_bytes=2,
    expected_missing=frozenset({("ShuMu", 6)}),
    source_archives=(
        SourceArchiveSpec(
            "Original_Audio_Word.zip",
            34_300_728,
            "504755a852fd0cb47f0f829b3da44daa",
            119,
        ),
        SourceArchiveSpec(
            "Original_sEEG_Vocalized_Word.zip",
            190_182_847,
            "1cf951dca48cea887eedf414fdd736e6",
            119,
        ),
    ),
)


@dataclass(frozen=True)
class AudioMetadata:
    channels: int
    sample_width_bytes: int
    sample_rate_hz: int
    frames: int
    compression: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.frames, self.channels


@dataclass(frozen=True)
class VocalMindTrial:
    trial_id: str
    word: str
    repetition: int
    audio_path: Path
    eeg_path: Path
    audio_relative_path: str
    eeg_relative_path: str


ProgressCallback = Callable[[int, int, VocalMindTrial], None]


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_regular_file_inside(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise VocalMindDataError(f"Expected a regular file, not a link: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise VocalMindDataError(f"File resolves outside the dataset directory: {path}") from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise VocalMindDataError(f"Cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VocalMindDataError(f"JSON receipt is not an object: {path}")
    return value


def _read_channel_ids(path: Path) -> tuple[int, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.reader(handle))
    except (OSError, StopIteration, csv.Error) as exc:
        raise VocalMindDataError(f"Cannot read CSV header from {path}: {exc}") from exc
    try:
        channel_ids = tuple(int(value) for value in row)
    except ValueError as exc:
        raise VocalMindDataError(f"Non-integer channel ID in {path}") from exc
    if len(set(channel_ids)) != len(channel_ids):
        raise VocalMindDataError(f"Duplicate channel IDs in {path}")
    return channel_ids


def _read_audio_metadata(path: Path) -> AudioMetadata:
    try:
        with wave.open(str(path), "rb") as handle:
            return AudioMetadata(
                channels=handle.getnchannels(),
                sample_width_bytes=handle.getsampwidth(),
                sample_rate_hz=handle.getframerate(),
                frames=handle.getnframes(),
                compression=handle.getcomptype(),
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise VocalMindDataError(f"Cannot read WAV metadata from {path}: {exc}") from exc


def _trial_id(word: str, repetition: int) -> str:
    return f"vocalized_word:{word}:rep{repetition:02d}"


class VocalMindAdapter:
    """Read-only access to the pinned raw VocalMind vocalized-word release."""

    def __init__(
        self,
        data_root: Path | str,
        contract: VocalMindContract = DEFAULT_VOCALMIND_CONTRACT,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.contract = contract
        self._trials: tuple[VocalMindTrial, ...] | None = None
        self._containers: dict[str, Path] | None = None
        self._receipts: tuple[dict[str, Any], ...] | None = None

    def _resolve_extraction_root(self) -> Path:
        candidates = (self.data_root / "extracted", self.data_root)
        for candidate in candidates:
            if (
                (candidate / "Original_Audio_Word").is_dir()
                and (candidate / "Original_sEEG_Vocalized_Word").is_dir()
            ):
                return candidate.resolve()
        raise VocalMindDataError(
            "Cannot find extracted/Original_Audio_Word and "
            "extracted/Original_sEEG_Vocalized_Word below the supplied data root"
        )

    def _locate_containers(self) -> dict[str, Path]:
        if self._containers is not None:
            return self._containers
        extraction_root = self._resolve_extraction_root()
        containers = {
            "audio": (extraction_root / "Original_Audio_Word").resolve(),
            "eeg": (extraction_root / "Original_sEEG_Vocalized_Word").resolve(),
        }
        for path in containers.values():
            try:
                path.relative_to(extraction_root)
            except ValueError as exc:
                raise VocalMindDataError(f"Extraction directory escapes data root: {path}") from exc
        self._containers = containers
        return containers

    def _validate_source_receipts(self) -> tuple[dict[str, Any], ...]:
        if self._receipts is not None:
            return self._receipts
        containers = self._locate_containers()
        by_stem = {path.name: path for path in containers.values()}
        validated: list[dict[str, Any]] = []
        for expected in self.contract.source_archives:
            try:
                container = by_stem[expected.extraction_directory]
            except KeyError as exc:
                raise VocalMindDataError(
                    f"Missing extraction directory for {expected.name}"
                ) from exc
            receipt = _read_json_object(container / ".extraction_receipt.json")
            checks = {
                "archive_name": expected.name,
                "archive_size_bytes": expected.size_bytes,
                "archive_md5": expected.md5,
                "member_count": expected.extracted_member_count,
            }
            for key, expected_value in checks.items():
                actual = receipt.get(key)
                if key == "archive_md5":
                    actual = str(actual).casefold()
                if actual != expected_value:
                    raise VocalMindDataError(
                        f"Receipt mismatch for {expected.name}: {key}={actual!r}, "
                        f"expected {expected_value!r}"
                    )
            validated.append(checks)
        self._receipts = tuple(validated)
        return self._receipts

    def _collect_paths(
        self,
        container: Path,
        suffix: str,
        pattern: re.Pattern[str],
        label: str,
    ) -> dict[tuple[str, int], Path]:
        paths: dict[tuple[str, int], Path] = {}
        files = sorted(container.rglob(f"*{suffix}"), key=lambda path: path.as_posix().casefold())
        if not files:
            raise VocalMindDataError(f"No {suffix} files found below {container}")
        for path in files:
            _ensure_regular_file_inside(path, container)
            match = pattern.fullmatch(path.name)
            if match is None:
                raise VocalMindDataError(f"Unexpected {label} filename: {path.name}")
            word = match.group("word")
            repetition = int(match.group("rep"))
            key = word, repetition
            if key in paths:
                raise VocalMindDataError(f"Duplicate {label} trial {key}: {path}")
            paths[key] = path
        return paths

    def discover(self) -> tuple[VocalMindTrial, ...]:
        if self._trials is not None:
            return self._trials
        self._validate_source_receipts()
        containers = self._locate_containers()
        audio = self._collect_paths(containers["audio"], ".wav", _AUDIO_NAME_RE, "audio")
        eeg = self._collect_paths(containers["eeg"], ".csv", _EEG_NAME_RE, "EEG")
        if set(audio) != set(eeg):
            audio_only = sorted(set(audio) - set(eeg))
            eeg_only = sorted(set(eeg) - set(audio))
            raise VocalMindDataError(
                f"Audio/EEG pairing mismatch; audio_only={audio_only[:5]}, eeg_only={eeg_only[:5]}"
            )
        actual_keys = frozenset(audio)
        if actual_keys != self.contract.expected_keys:
            missing = sorted(self.contract.expected_keys - actual_keys)
            unexpected = sorted(actual_keys - self.contract.expected_keys)
            raise VocalMindDataError(
                f"Trial inventory differs from the frozen contract; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

        word_order = {word: index for index, word in enumerate(self.contract.words)}
        ordered_keys = sorted(actual_keys, key=lambda key: (word_order[key[0]], key[1]))
        trials = []
        for word, repetition in ordered_keys:
            audio_path = audio[(word, repetition)]
            eeg_path = eeg[(word, repetition)]
            trials.append(
                VocalMindTrial(
                    trial_id=_trial_id(word, repetition),
                    word=word,
                    repetition=repetition,
                    audio_path=audio_path,
                    eeg_path=eeg_path,
                    audio_relative_path=audio_path.relative_to(self.data_root).as_posix(),
                    eeg_relative_path=eeg_path.relative_to(self.data_root).as_posix(),
                )
            )
        self._trials = tuple(trials)
        return self._trials

    def trial(self, trial_id: str) -> VocalMindTrial:
        matches = [trial for trial in self.discover() if trial.trial_id == trial_id]
        if len(matches) != 1:
            raise KeyError(trial_id)
        return matches[0]

    def validate_metadata(self, trial: VocalMindTrial) -> AudioMetadata:
        channel_ids = _read_channel_ids(trial.eeg_path)
        if channel_ids != self.contract.channel_ids:
            raise VocalMindDataError(
                f"Channel order mismatch for {trial.trial_id}: got {channel_ids}, "
                f"expected {self.contract.channel_ids}"
            )
        metadata = _read_audio_metadata(trial.audio_path)
        expected = AudioMetadata(
            channels=self.contract.audio_channels,
            sample_width_bytes=self.contract.audio_sample_width_bytes,
            sample_rate_hz=self.contract.audio_sample_rate_hz,
            frames=self.contract.audio_frames,
            compression="NONE",
        )
        if metadata != expected:
            raise VocalMindDataError(
                f"WAV contract mismatch for {trial.trial_id}: got {metadata}, expected {expected}"
            )
        return metadata

    def load_eeg(self, trial: VocalMindTrial, dtype: np.dtype[Any] = np.dtype("float32")) -> np.ndarray:
        channel_ids = _read_channel_ids(trial.eeg_path)
        if channel_ids != self.contract.channel_ids:
            raise VocalMindDataError(f"Channel order mismatch for {trial.trial_id}")
        try:
            values = np.loadtxt(
                trial.eeg_path,
                delimiter=",",
                skiprows=1,
                dtype=np.float64,
                ndmin=2,
            )
        except (OSError, ValueError) as exc:
            raise VocalMindDataError(f"Cannot parse EEG CSV for {trial.trial_id}: {exc}") from exc
        expected_shape = self.contract.eeg_samples, len(self.contract.channel_ids)
        if values.shape != expected_shape:
            raise VocalMindDataError(
                f"EEG shape mismatch for {trial.trial_id}: got {values.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise VocalMindDataError(f"Non-finite EEG values in {trial.trial_id}")
        result = np.asarray(values, dtype=dtype, order="C")
        result.setflags(write=False)
        return result

    def load_audio(self, trial: VocalMindTrial) -> np.ndarray:
        metadata = self.validate_metadata(trial)
        try:
            with wave.open(str(trial.audio_path), "rb") as handle:
                frames = handle.readframes(metadata.frames)
        except (OSError, EOFError, wave.Error) as exc:
            raise VocalMindDataError(f"Cannot read WAV samples for {trial.trial_id}: {exc}") from exc
        expected_bytes = metadata.frames * metadata.channels * metadata.sample_width_bytes
        if len(frames) != expected_bytes:
            raise VocalMindDataError(
                f"WAV payload size mismatch for {trial.trial_id}: "
                f"got {len(frames)}, expected {expected_bytes}"
            )
        if metadata.sample_width_bytes != 2:
            raise VocalMindDataError("Only pinned 16-bit PCM VocalMind audio is supported")
        pcm = np.frombuffer(frames, dtype="<i2").reshape(metadata.shape)
        result = pcm.astype(np.float32) / 32768.0
        if not np.isfinite(result).all():
            raise VocalMindDataError(f"Non-finite audio values in {trial.trial_id}")
        result.setflags(write=False)
        return result

    def build_index(
        self,
        *,
        deep: bool = True,
        hash_files: bool = True,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        trials = self.discover()
        records: list[dict[str, Any]] = []
        for position, trial in enumerate(trials, start=1):
            audio_metadata = self.validate_metadata(trial)
            if deep:
                self.load_eeg(trial)
                self.load_audio(trial)
            eeg_record: dict[str, Any] = {
                "path": trial.eeg_relative_path,
                "size_bytes": trial.eeg_path.stat().st_size,
                "shape": [self.contract.eeg_samples, len(self.contract.channel_ids)],
                "sample_rate_hz": self.contract.eeg_sample_rate_hz,
            }
            audio_record: dict[str, Any] = {
                "path": trial.audio_relative_path,
                "size_bytes": trial.audio_path.stat().st_size,
                "shape": list(audio_metadata.shape),
                "sample_rate_hz": audio_metadata.sample_rate_hz,
                "sample_width_bytes": audio_metadata.sample_width_bytes,
            }
            if hash_files:
                eeg_record["sha256"] = _sha256_file(trial.eeg_path)
                audio_record["sha256"] = _sha256_file(trial.audio_path)
            records.append(
                {
                    "trial_id": trial.trial_id,
                    "word": trial.word,
                    "repetition": trial.repetition,
                    "primary_eligible": trial.repetition in self.contract.primary_repetitions,
                    "eeg": eeg_record,
                    "audio": audio_record,
                }
            )
            if progress is not None:
                progress(position, len(trials), trial)

        primary_count = len(self.contract.words) * len(self.contract.primary_repetitions)
        if sum(record["primary_eligible"] for record in records) != primary_count:
            raise VocalMindDataError("Primary trial count is not balanced across repetitions 1--5")
        core: dict[str, Any] = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "adapter_version": ADAPTER_VERSION,
            "dataset": "VocalMind-v2",
            "task": "vocalized_word",
            "validation": {
                "deep_numeric_shapes_and_finite": deep,
                "individual_file_sha256": hash_files,
                "strict_pairing": True,
                "channel_order": True,
                "wav_metadata": True,
            },
            "contract": {
                "words": list(self.contract.words),
                "word_count": len(self.contract.words),
                "primary_repetitions": list(self.contract.primary_repetitions),
                "secondary_repetition": self.contract.secondary_repetition,
                "expected_missing": [
                    {"word": word, "repetition": repetition}
                    for word, repetition in sorted(self.contract.expected_missing)
                ],
                "eeg_sample_rate_hz": self.contract.eeg_sample_rate_hz,
                "eeg_shape": [self.contract.eeg_samples, len(self.contract.channel_ids)],
                "channel_ids": list(self.contract.channel_ids),
                "audio_sample_rate_hz": self.contract.audio_sample_rate_hz,
                "audio_shape": [self.contract.audio_frames, self.contract.audio_channels],
                "audio_sample_width_bytes": self.contract.audio_sample_width_bytes,
                "duration_seconds": self.contract.duration_seconds,
            },
            "source_archives": list(self._validate_source_receipts()),
            "counts": {
                "all_paired_trials": len(records),
                "primary_trials_reps_1_5": primary_count,
                "secondary_rep6_trials": sum(
                    record["repetition"] == self.contract.secondary_repetition
                    for record in records
                ),
            },
            "trials": records,
        }
        result = dict(core)
        result["dataset_index_sha256"] = _fingerprint(core)
        return result


def _ordered_ids_for_repetitions(
    records: Sequence[Mapping[str, Any]],
    repetitions: set[int],
) -> list[str]:
    if not repetitions.issubset({1, 2, 3, 4, 5}):
        raise VocalMindDataError(
            f"Primary partitions may only use repetitions 1--5, got {sorted(repetitions)}"
        )
    return [
        str(record["trial_id"])
        for record in records
        if int(record["repetition"]) in repetitions
    ]


def build_primary_split_manifest(
    dataset_index: Mapping[str, Any],
    contract: VocalMindContract = DEFAULT_VOCALMIND_CONTRACT,
) -> dict[str, Any]:
    records_value = dataset_index.get("trials")
    if not isinstance(records_value, list):
        raise VocalMindDataError("Dataset index has no trial list")
    records: list[Mapping[str, Any]] = records_value
    primary_records = [
        record
        for record in records
        if int(record["repetition"]) in contract.primary_repetitions
    ]
    expected_primary = len(contract.words) * len(contract.primary_repetitions)
    if len(primary_records) != expected_primary:
        raise VocalMindDataError(
            f"Expected {expected_primary} primary trials, got {len(primary_records)}"
        )
    for repetition in contract.primary_repetitions:
        words = {
            str(record["word"])
            for record in primary_records
            if int(record["repetition"]) == repetition
        }
        if words != set(contract.words):
            raise VocalMindDataError(
                f"Primary repetition {repetition} is not a complete {len(contract.words)}-class block"
            )

    folds = []
    all_primary_ids = {str(record["trial_id"]) for record in primary_records}
    for position, test_repetition in enumerate(contract.primary_repetitions):
        validation_repetition = contract.primary_repetitions[
            (position + 1) % len(contract.primary_repetitions)
        ]
        train_repetitions = set(contract.primary_repetitions) - {
            test_repetition,
            validation_repetition,
        }
        train_ids = _ordered_ids_for_repetitions(records, train_repetitions)
        validation_ids = _ordered_ids_for_repetitions(records, {validation_repetition})
        test_ids = _ordered_ids_for_repetitions(records, {test_repetition})
        partitions = [set(train_ids), set(validation_ids), set(test_ids)]
        if any(left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
            raise VocalMindDataError(f"Overlap detected in fold {position + 1}")
        if set().union(*partitions) != all_primary_ids:
            raise VocalMindDataError(f"Fold {position + 1} does not cover every primary trial")
        if any("rep06" in trial_id for trial_id in train_ids + validation_ids + test_ids):
            raise VocalMindDataError("Repetition 6 entered a primary partition")
        folds.append(
            {
                "fold": position + 1,
                "test_repetition": test_repetition,
                "validation_repetition": validation_repetition,
                "train_repetitions": sorted(train_repetitions),
                "train_trial_ids": train_ids,
                "validation_trial_ids": validation_ids,
                "test_trial_ids": test_ids,
                "counts": {
                    "train": len(train_ids),
                    "validation": len(validation_ids),
                    "test": len(test_ids),
                },
            }
        )

    secondary_ids = [
        str(record["trial_id"])
        for record in records
        if int(record["repetition"]) == contract.secondary_repetition
    ]
    core: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset": "VocalMind-v2",
        "dataset_index_sha256": dataset_index.get("dataset_index_sha256"),
        "split_policy": PRIMARY_SPLIT_POLICY,
        "class_order": list(contract.words),
        "primary_repetitions": list(contract.primary_repetitions),
        "forbidden_primary_repetitions": [contract.secondary_repetition],
        "expected_missing_secondary": [
            {"word": word, "repetition": repetition}
            for word, repetition in sorted(contract.expected_missing)
        ],
        "all_primary_trial_ids": [
            str(record["trial_id"])
            for record in records
            if int(record["repetition"]) in contract.primary_repetitions
        ],
        "excluded_secondary_trial_ids": secondary_ids,
        "folds": folds,
    }
    result = dict(core)
    result["split_manifest_sha256"] = _fingerprint(core)
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise VocalMindDataError(f"Output exists; use --force-output to replace it: {path}")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise VocalMindDataError(f"Temporary output already exists: {temporary}")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _progress(position: int, total: int, trial: VocalMindTrial) -> None:
    print(
        f"[{position:03d}/{total:03d}] {trial.trial_id}",
        file=sys.stderr,
        flush=True,
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="skip numeric CSV/audio payload shape and finite-value checks",
    )
    parser.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="skip per-trial SHA-256 (archive receipt checks still run)",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="validate and summarize read-only data")
    _add_common_arguments(inventory_parser)
    inventory_parser.add_argument("--json-out", type=Path)
    inventory_parser.add_argument("--force-output", action="store_true")

    index_parser = subparsers.add_parser("index", help="write deterministic dataset and split manifests")
    _add_common_arguments(index_parser)
    index_parser.add_argument("--index-out", type=Path, required=True)
    index_parser.add_argument("--splits-out", type=Path, required=True)
    index_parser.add_argument("--force-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    adapter = VocalMindAdapter(args.data_root)
    dataset_index = adapter.build_index(
        deep=not args.metadata_only,
        hash_files=not args.skip_file_hashes,
        progress=_progress,
    )
    split_manifest = build_primary_split_manifest(dataset_index)
    summary = {
        "dataset": dataset_index["dataset"],
        "adapter_version": dataset_index["adapter_version"],
        "counts": dataset_index["counts"],
        "validation": dataset_index["validation"],
        "dataset_index_sha256": dataset_index["dataset_index_sha256"],
        "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        "fold_counts": [fold["counts"] for fold in split_manifest["folds"]],
        "rep6_primary_forbidden": True,
    }

    if args.command == "inventory":
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        if args.json_out:
            _atomic_write_json(args.json_out, dataset_index, args.force_output)
            print(f"[saved index] {args.json_out}", file=sys.stderr)
    else:
        _atomic_write_json(args.index_out, dataset_index, args.force_output)
        _atomic_write_json(args.splits_out, split_manifest, args.force_output)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"[saved index] {args.index_out}", file=sys.stderr)
        print(f"[saved splits] {args.splits_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VocalMindDataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
