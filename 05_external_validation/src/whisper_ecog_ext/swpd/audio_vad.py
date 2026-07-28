"""Deterministic audio-only speech candidates for manual SWPD audit.

This module deliberately does not accept visual events.  Its TSV output is a
candidate annotation, not ground truth, and cannot authorize asynchronous
evaluation until a human audit receipt has been supplied.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..integrity import atomic_write_json, fingerprint_json, read_json, sha256_file
from ..targets import resample_audio_polyphase


CANDIDATE_LABEL_SOURCE = "audio_energy_candidate_unreviewed"
AUDITED_LABEL_SOURCES = frozenset({"audio_manual", "audio_vad_audited"})


class AudioAuditRequired(PermissionError):
    """Raised when an event-level operation is attempted before audio audit."""


@dataclass(frozen=True)
class EnergyVadConfig:
    processing_input_rate_hz: int = 48_000
    analysis_rate_hz: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    noise_percentile: float = 20.0
    onset_db_above_noise: float = 12.0
    offset_db_above_noise: float = 8.0
    minimum_speech_ms: float = 100.0
    merge_gap_ms: float = 150.0

    def __post_init__(self) -> None:
        if int(self.processing_input_rate_hz) <= 0 or int(self.analysis_rate_hz) <= 0:
            raise ValueError("audio sample rates must be positive")
        for name in ("frame_ms", "hop_ms", "minimum_speech_ms", "merge_gap_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= float(self.noise_percentile) <= 100:
            raise ValueError("noise_percentile must be in [0, 100]")
        if not (
            math.isfinite(float(self.onset_db_above_noise))
            and math.isfinite(float(self.offset_db_above_noise))
            and self.onset_db_above_noise > self.offset_db_above_noise >= 0
        ):
            raise ValueError("VAD onset threshold must exceed its non-negative offset threshold")

    @property
    def frame_samples(self) -> int:
        return int(round(self.analysis_rate_hz * self.frame_ms / 1000.0))

    @property
    def hop_samples(self) -> int:
        return int(round(self.analysis_rate_hz * self.hop_ms / 1000.0))


@dataclass(frozen=True)
class AudioInterval:
    onset_seconds: float
    offset_seconds: float
    peak_db: float


def _frame_rms_db(audio: np.ndarray, config: EnergyVadConfig) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame RMS in bounded-memory chunks."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size < config.frame_samples:
        raise ValueError("audio is shorter than one VAD frame")
    if not np.isfinite(values).all():
        raise ValueError("audio contains NaN or Infinity")
    peak = float(np.max(np.abs(values)))
    if peak <= np.finfo(np.float32).eps:
        normalized = np.zeros_like(values)
    else:
        normalized = values / peak
    starts = np.arange(
        0,
        values.size - config.frame_samples + 1,
        config.hop_samples,
        dtype=np.int64,
    )
    rms = np.empty(starts.size, dtype=np.float64)
    chunk_frames = 4096
    for first in range(0, starts.size, chunk_frames):
        selected = starts[first : first + chunk_frames]
        block_start = int(selected[0])
        block_stop = int(selected[-1]) + config.frame_samples
        block = np.asarray(normalized[block_start:block_stop], dtype=np.float64)
        squared_prefix = np.empty(block.size + 1, dtype=np.float64)
        squared_prefix[0] = 0.0
        np.cumsum(block * block, out=squared_prefix[1:])
        local = selected - block_start
        energy = (
            squared_prefix[local + config.frame_samples] - squared_prefix[local]
        ) / config.frame_samples
        rms[first : first + len(selected)] = np.sqrt(np.maximum(energy, 0.0))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return starts.astype(np.float64) / config.analysis_rate_hz, db


def detect_audio_energy_candidates(
    audio: np.ndarray,
    *,
    config: EnergyVadConfig = EnergyVadConfig(),
) -> tuple[tuple[AudioInterval, ...], dict]:
    """Return deterministic candidates on the recording-relative audio clock."""

    audio16 = resample_audio_polyphase(
        np.asarray(audio, dtype=np.float32),
        config.processing_input_rate_hz,
        config.analysis_rate_hz,
    )
    frame_starts, frame_db = _frame_rms_db(audio16, config)
    noise_db = float(np.percentile(frame_db, config.noise_percentile))
    onset_threshold = noise_db + config.onset_db_above_noise
    offset_threshold = noise_db + config.offset_db_above_noise

    raw: list[tuple[int, int]] = []
    active_start: int | None = None
    for index, value in enumerate(frame_db):
        if active_start is None:
            if value >= onset_threshold:
                active_start = index
        elif value < offset_threshold:
            raw.append((active_start, index))
            active_start = None
    if active_start is not None:
        raw.append((active_start, len(frame_db) - 1))

    converted: list[AudioInterval] = []
    for start_index, stop_index in raw:
        onset = float(frame_starts[start_index])
        offset = float(frame_starts[stop_index] + config.frame_ms / 1000.0)
        if (offset - onset) * 1000.0 < config.minimum_speech_ms:
            continue
        peak_db = float(np.max(frame_db[start_index : stop_index + 1]))
        candidate = AudioInterval(onset, offset, peak_db)
        if converted and onset - converted[-1].offset_seconds <= config.merge_gap_ms / 1000.0:
            previous = converted.pop()
            candidate = AudioInterval(
                previous.onset_seconds,
                offset,
                max(previous.peak_db, peak_db),
            )
        converted.append(candidate)

    provenance = {
        "schema_version": 1,
        "kind": "deterministic_audio_energy_vad_candidates",
        "algorithm": "RMS_dB_hysteresis_v1",
        "config": asdict(config),
        "frame_samples": config.frame_samples,
        "hop_samples": config.hop_samples,
        "noise_db": noise_db,
        "onset_threshold_db": onset_threshold,
        "offset_threshold_db": offset_threshold,
        "frame_count": int(frame_db.size),
        "candidate_count": len(converted),
        "time_coordinate": "seconds_relative_to_audio_recording_start",
        "visual_events_used": False,
        "ground_truth_status": "unreviewed_candidate_only",
    }
    provenance["fingerprint"] = fingerprint_json(provenance)
    return tuple(converted), provenance


def write_audio_candidates(
    intervals: Sequence[AudioInterval],
    provenance: dict,
    *,
    tsv_path: Path,
    metadata_path: Path,
    measured_nwb_rate_hz: float,
) -> tuple[Path, Path]:
    """Create immutable candidate files for a later manual listen-and-audit pass."""

    tsv_path = Path(tsv_path)
    metadata_path = Path(metadata_path)
    if tsv_path.exists() or metadata_path.exists():
        raise FileExistsError("audio candidate output already exists")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tsv_path.with_name(f".{tsv_path.name}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "candidate_id",
                    "onset_seconds",
                    "offset_seconds",
                    "peak_db",
                    "label_source",
                    "audit_decision",
                    "auditor_notes",
                ),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for index, interval in enumerate(intervals):
                writer.writerow(
                    {
                        "candidate_id": f"audio-candidate-{index:04d}",
                        "onset_seconds": f"{interval.onset_seconds:.6f}",
                        "offset_seconds": f"{interval.offset_seconds:.6f}",
                        "peak_db": f"{interval.peak_db:.6f}",
                        "label_source": CANDIDATE_LABEL_SOURCE,
                        "audit_decision": "UNREVIEWED",
                        "auditor_notes": "",
                    }
                )
        temporary.replace(tsv_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    payload = dict(provenance)
    payload.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_tsv": str(tsv_path.resolve()),
            "candidate_tsv_sha256": sha256_file(tsv_path),
            "measured_nwb_audio_rate_hz": float(measured_nwb_rate_hz),
            "processing_input_rate_hz": int(
                provenance["config"]["processing_input_rate_hz"]
            ),
            "manual_audit_required": True,
            "event_evaluation_authorized": False,
        }
    )
    payload["fingerprint"] = fingerprint_json(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    atomic_write_json(metadata_path, payload, overwrite=False)
    return tsv_path, metadata_path


def validate_audio_candidate_bundle(tsv_path: Path, metadata_path: Path) -> dict:
    """Verify immutable unreviewed candidates before reuse."""

    tsv_path = Path(tsv_path)
    metadata_path = Path(metadata_path)
    if not tsv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("audio candidate TSV/metadata bundle is incomplete")
    payload = read_json(metadata_path)
    stored = payload.pop("fingerprint", None)
    if stored != fingerprint_json(payload):
        raise RuntimeError("audio candidate metadata fingerprint mismatch")
    if (
        payload.get("kind") != "deterministic_audio_energy_vad_candidates"
        or payload.get("candidate_tsv_sha256") != sha256_file(tsv_path)
        or payload.get("manual_audit_required") is not True
        or payload.get("event_evaluation_authorized") is not False
        or payload.get("visual_events_used") is not False
    ):
        raise RuntimeError("audio candidate metadata provenance mismatch")
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows and any(
        row.get("label_source") != CANDIDATE_LABEL_SOURCE
        or row.get("audit_decision") != "UNREVIEWED"
        for row in rows
    ):
        raise RuntimeError("candidate TSV was edited; create a separate audited TSV instead")
    result = dict(payload)
    result["fingerprint"] = stored
    return result


def load_audited_audio_intervals(
    audited_tsv: Path,
    audit_receipt: Path,
    *,
    candidate_tsv: Path,
) -> tuple[tuple[float, float], ...]:
    """Validate human approval; an edited TSV alone is intentionally insufficient."""

    audited_tsv = Path(audited_tsv)
    audit_receipt = Path(audit_receipt)
    candidate_tsv = Path(candidate_tsv)
    missing = [path for path in (audited_tsv, audit_receipt, candidate_tsv) if not path.is_file()]
    if missing:
        raise AudioAuditRequired("audio audit is incomplete: " + ", ".join(map(str, missing)))
    receipt = read_json(audit_receipt)
    stored = receipt.pop("fingerprint", None)
    if stored != fingerprint_json(receipt):
        raise RuntimeError("audio audit receipt fingerprint mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "human_audio_interval_audit"
        or receipt.get("review_status") != "approved"
        or not str(receipt.get("reviewer", "")).strip()
        or receipt.get("candidate_tsv_sha256") != sha256_file(candidate_tsv)
        or receipt.get("audited_tsv_sha256") != sha256_file(audited_tsv)
    ):
        raise AudioAuditRequired("audio audit receipt is absent, incomplete, or belongs to other files")

    with audited_tsv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"onset_seconds", "offset_seconds", "label_source"}
    if not rows or not required.issubset(rows[0]):
        raise AudioAuditRequired(f"audited TSV must contain {sorted(required)}")
    intervals: list[tuple[float, float]] = []
    for row in rows:
        if row["label_source"] not in AUDITED_LABEL_SOURCES:
            raise AudioAuditRequired("all event intervals must be manually derived/audited from audio")
        onset = float(row["onset_seconds"])
        offset = float(row["offset_seconds"])
        if not np.isfinite(onset) or not np.isfinite(offset) or offset <= onset:
            raise AudioAuditRequired("audited audio TSV contains an invalid interval")
        intervals.append((onset, offset))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        raise AudioAuditRequired("audited audio intervals overlap")
    return tuple(intervals)


def closed_event_gate_payload(
    *,
    candidate_tsv: Path | None,
    regression_units: Iterable[str],
) -> dict:
    """Describe, but never open, the post-regression asynchronous gate."""

    candidate = Path(candidate_tsv) if candidate_tsv is not None else None
    payload = {
        "schema_version": 1,
        "kind": "swpd_asynchronous_event_gate",
        "open": False,
        "visual_events_are_ground_truth": False,
        "audio_candidate_tsv": str(candidate.resolve()) if candidate and candidate.exists() else None,
        "audio_candidate_tsv_sha256": sha256_file(candidate)
        if candidate and candidate.is_file()
        else None,
        "completed_regression_units": sorted(str(value) for value in regression_units),
        "required_before_open": [
            "human-approved audio-only interval TSV plus bound audit receipt",
            "continuous heads trained and fixed on train/validation only",
            "fixed L3+L4+L5 arithmetic probability ensemble",
            "separate explicit final-test authorization",
        ],
        "blocked_operations": [
            "event metrics",
            "continuous asynchronous final test",
            "L3+L4+L5 event probability ensemble evaluation",
        ],
    }
    payload["fingerprint"] = fingerprint_json(payload)
    return payload
