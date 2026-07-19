#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared, read-only-base helpers for continuous-window word-head training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import async_replay as ar


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
DEFAULT_ARCHIVE = REPOSITORY_ROOT / "checkpoints" / "frozen_seed4"
DEFAULT_CACHE = REPOSITORY_ROOT / "artifacts" / "async_hidden_cache"
CACHE_VERSION = 3
PATIENT = "ivanova"
SEED = 4
LAYERS = (3, 4, 5)
WINDOW_FRAMES = 52


@dataclass(frozen=True)
class CachedFile:
    file_index: int
    hidden_path: Path
    labels_path: Path
    metadata_path: Path
    n_frames: int
    hidden_dim: int
    frame_hz: float
    first_endpoint: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(ar.json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_files(archive: Path, paths: Sequence[Path]) -> None:
    manifest_path = archive / "MANIFEST.sha256.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen archive manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if int(manifest.get("seed", -1)) != SEED:
        raise RuntimeError(f"Frozen archive has unexpected source seed: {manifest_path}")
    entries = {
        str(item["relative_path"]).replace("\\", "/"): item
        for item in manifest.get("files", [])
    }
    for path in paths:
        relative = path.relative_to(archive).as_posix()
        entry = entries.get(relative)
        if not entry:
            raise RuntimeError(f"Frozen file is absent from manifest: {relative}")
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Frozen file failed SHA256 verification: {path}")


def archive_spec(archive: Path, layer: int) -> ar.ModelSpec:
    layer_dir = archive / "upstream" / "checkpoints" / f"L{layer}"
    result_dir = archive / "upstream" / "results" / f"L{layer}"
    regression_files = list(layer_dir.glob("regression___*.pth"))
    classifier_files = list(layer_dir.glob("classification_hidden___*.pth"))
    classification_results = list(result_dir.glob("classification_hidden___*.json"))
    if not (len(regression_files) == len(classifier_files) == len(classification_results) == 1):
        raise RuntimeError(f"Frozen L{layer} archive is ambiguous or incomplete: {layer_dir}")
    result = read_json(classification_results[0])
    config = result.get("config") or {}
    if int(config.get("seed", -1)) != SEED:
        raise RuntimeError(f"Frozen L{layer} has unexpected seed: {config.get('seed')}")
    if (config.get("patient") != PATIENT or config.get("mode") != "classification_hidden"
            or config.get("control", "none") != "none"
            or config.get("augment", "none") != "none"):
        raise RuntimeError(f"Frozen L{layer} experiment identity/config is unexpected")
    model_name = str(config["regression_model"])
    date = classification_results[0].stem.rsplit("___", 1)[-1]
    if not model_name.endswith(f"WHISPER_BASE_L{layer}"):
        raise RuntimeError(f"Frozen result/model mismatch for L{layer}: {model_name}")
    if int(config.get("max_words_length", -1)) != WINDOW_FRAMES:
        raise RuntimeError(
            f"Frozen L{layer} word window is {config.get('max_words_length')}, "
            f"expected {WINDOW_FRAMES} frames"
        )
    if not all(date in path.stem and model_name in path.stem
               for path in (regression_files[0], classifier_files[0])):
        raise RuntimeError(f"Frozen L{layer} checkpoint/result dates or models do not match")
    expected_names = (
        f"regression___{PATIENT}___{model_name}___{date}.pth",
        f"classification_hidden___{PATIENT}___{model_name}___{date}.pth",
        f"classification_hidden___{PATIENT}___{model_name}___{date}.json",
    )
    actual_names = (
        regression_files[0].name, classifier_files[0].name, classification_results[0].name
    )
    if actual_names != expected_names:
        raise RuntimeError(f"Frozen L{layer} filenames do not match result provenance")
    verify_frozen_files(
        archive, (regression_files[0], classifier_files[0], classification_results[0])
    )
    return ar.ModelSpec(
        layer=layer,
        model_name=model_name,
        date=date,
        seed=SEED,
        max_words_length=int(config["max_words_length"]),
        regression_path=regression_files[0],
        classifier_path=classifier_files[0],
        result_path=classification_results[0],
        synchronous_accuracy=float(result["test_accuracy_full"]),
    )


def load_frozen_bundle(api, patient: dict, archive: Path, layer: int) -> ar.ModelBundle:
    spec = archive_spec(archive, layer)
    bundle = ar.load_bundle(api, patient, spec, allow_nonzero_lead=False)
    bundle.regression.model.eval()
    bundle.regression.model.requires_grad_(False)
    return bundle


def cache_manifest_path(cache_root: Path, layer: int) -> Path:
    return cache_root / f"L{layer}" / "manifest.json"


def labels_path(cache_root: Path, file_index: int) -> Path:
    return cache_root / "metadata" / f"file_{file_index:02d}_labels.npy"


def metadata_path(cache_root: Path, file_index: int) -> Path:
    return cache_root / "metadata" / f"file_{file_index:02d}.json"


def hidden_path(cache_root: Path, layer: int, file_index: int) -> Path:
    return cache_root / f"L{layer}" / f"file_{file_index:02d}.npy"


def build_labels_and_metadata(api, patient: dict, regression, file_index: int,
                              n_frames: int, raw_samples: int) -> Tuple[np.ndarray, dict]:
    filepath = Path(patient["files_list"][file_index])
    words_file = Path(api.get_words_filepath(str(filepath)))
    words = api.load_words_info(str(words_file))
    raw_fs = float(patient["sampling_rate"])
    effective_downsampling = int(regression.downsampling_coef * api.HIDDEN_STRIDE)
    frame_hz = raw_fs / effective_downsampling
    first_valid_hidden = int(math.ceil(regression.LAG_BACKWARD / api.HIDDEN_STRIDE))
    first_endpoint = first_valid_hidden + WINDOW_FRAMES - 1
    labels = np.zeros(n_frames, dtype=np.int16)
    events = []
    for event_index, (raw_start, raw_end, word) in enumerate(words):
        raw_start = int(raw_start)
        raw_end = int(raw_end)
        if raw_end > raw_samples:
            continue
        start_frame = max(0, int(raw_start / effective_downsampling))
        end_frame = min(n_frames, int(raw_end / effective_downsampling))
        class_index = int(api.WORDS_REMAP[str(word)])
        if end_frame > start_frame:
            if np.any(labels[start_frame:end_frame] != 0):
                raise RuntimeError(f"Overlapping word annotations in file {file_index}")
            labels[start_frame:end_frame] = class_index
        events.append({
            "event_index": int(event_index),
            "word": str(word),
            "class_index": class_index,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "start_s": raw_start / raw_fs,
            "end_s": raw_end / raw_fs,
            "hidden_start": start_frame,
            "hidden_end": end_frame,
        })
    metadata = {
        "version": CACHE_VERSION,
        "patient": patient["name"],
        "file_index": file_index,
        "raw_file": str(filepath),
        "raw_file_bytes": filepath.stat().st_size,
        "raw_file_mtime_ns": filepath.stat().st_mtime_ns,
        "words_file": str(words_file),
        "words_file_bytes": words_file.stat().st_size,
        "words_file_mtime_ns": words_file.stat().st_mtime_ns,
        "words_file_sha256": sha256_file(words_file),
        "raw_samples": int(raw_samples),
        "raw_fs": raw_fs,
        "duration_s": raw_samples / raw_fs,
        "n_hidden_frames": int(n_frames),
        "hidden_frame_hz": frame_hz,
        "effective_downsampling_raw_samples": effective_downsampling,
        "hidden_stride": int(api.HIDDEN_STRIDE),
        "word_window_frames": WINDOW_FRAMES,
        "first_valid_hidden_frame": first_valid_hidden,
        "first_endpoint": first_endpoint,
        "events": events,
    }
    return labels, metadata


def ground_truth_from_metadata(metadata: dict) -> List[ar.GroundTruth]:
    file_index = int(metadata["file_index"])
    return [
        ar.GroundTruth(
            file_index=file_index,
            event_index=int(event["event_index"]),
            start_s=float(event["start_s"]),
            end_s=float(event["end_s"]),
            class_index=int(event["class_index"]),
            word=str(event["word"]),
        )
        for event in metadata["events"]
    ]


def load_cached_file(cache_root: Path, layer: int, file_index: int,
                     verify_sha: bool = False) -> CachedFile:
    manifest_file = cache_manifest_path(cache_root, layer)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Cache manifest is missing: {manifest_file}")
    manifest = read_json(manifest_file)
    entry = (manifest.get("files") or {}).get(str(file_index))
    if not entry or not entry.get("complete"):
        raise RuntimeError(f"L{layer} file {file_index} cache is incomplete")
    h_path = Path(entry["hidden_path"])
    l_path = labels_path(cache_root, file_index)
    m_path = metadata_path(cache_root, file_index)
    for path in (h_path, l_path, m_path):
        if not path.is_file():
            raise FileNotFoundError(f"Cache artifact is missing: {path}")
    if h_path.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"Cache size mismatch: {h_path}")
    expected_paths = (
        hidden_path(cache_root, layer, file_index), l_path, m_path
    )
    recorded_paths = (
        h_path, Path(entry.get("labels_path", "")), Path(entry.get("metadata_path", ""))
    )
    if any(recorded.resolve() != expected.resolve()
           for recorded, expected in zip(recorded_paths, expected_paths)):
        raise RuntimeError(f"Cache path identity mismatch for L{layer} file {file_index}")
    if (l_path.stat().st_size != int(entry.get("labels_bytes", -1))
            or m_path.stat().st_size != int(entry.get("metadata_bytes", -1))):
        raise RuntimeError(f"Cache metadata/labels size mismatch for file {file_index}")
    if verify_sha and sha256_file(h_path) != entry["sha256"]:
        raise RuntimeError(f"Cache SHA256 mismatch: {h_path}")
    hidden = np.load(h_path, mmap_mode="r")
    metadata = read_json(m_path)
    labels = np.load(l_path, mmap_mode="r")
    expected_shape = tuple(int(item) for item in entry["shape"])
    if hidden.dtype != np.float32 or hidden.shape != expected_shape:
        raise RuntimeError(
            f"Cache dtype/shape mismatch for {h_path}: {hidden.dtype}/{hidden.shape}"
        )
    if int(metadata["n_hidden_frames"]) != hidden.shape[0]:
        raise RuntimeError(f"Metadata/cache frame mismatch: {h_path}")
    if int(metadata.get("version", -1)) != CACHE_VERSION:
        raise RuntimeError(f"Metadata version mismatch: {m_path}")
    if metadata.get("patient") != PATIENT or int(metadata.get("file_index", -1)) != file_index:
        raise RuntimeError(f"Metadata identity mismatch: {m_path}")
    if labels.dtype != np.int16 or labels.shape != (hidden.shape[0],):
        raise RuntimeError(f"Labels dtype/shape mismatch: {l_path}")
    raw_file = Path(metadata["raw_file"])
    words_file = Path(metadata["words_file"])
    if (str(raw_file.resolve()) != str(Path(entry["raw_file"]).resolve())
            or raw_file.stat().st_size != int(entry["raw_file_bytes"])
            or raw_file.stat().st_mtime_ns != int(entry["raw_file_mtime_ns"])):
        raise RuntimeError(f"Raw source/manifest mismatch: {raw_file}")
    for source, size_key, mtime_key in (
        (raw_file, "raw_file_bytes", "raw_file_mtime_ns"),
        (words_file, "words_file_bytes", "words_file_mtime_ns"),
    ):
        if (not source.is_file() or source.stat().st_size != int(metadata[size_key])
                or source.stat().st_mtime_ns != int(metadata[mtime_key])):
            raise RuntimeError(f"Source changed after cache creation: {source}")
    if verify_sha:
        for path, key in ((l_path, "labels_sha256"), (m_path, "metadata_sha256")):
            expected = entry.get(key)
            if not expected or sha256_file(path) != expected:
                raise RuntimeError(f"Cache SHA256 mismatch: {path}")
        if sha256_file(words_file) != metadata["words_file_sha256"]:
            raise RuntimeError(f"Annotation SHA256 mismatch: {words_file}")
    return CachedFile(
        file_index=file_index,
        hidden_path=h_path,
        labels_path=l_path,
        metadata_path=m_path,
        n_frames=int(hidden.shape[0]),
        hidden_dim=int(hidden.shape[1]),
        frame_hz=float(metadata["hidden_frame_hz"]),
        first_endpoint=int(metadata["first_endpoint"]),
    )


def open_hidden(cached: CachedFile):
    return np.load(cached.hidden_path, mmap_mode="r")


def open_labels(cached: CachedFile):
    return np.load(cached.labels_path, mmap_mode="r")


def gather_windows(hidden_by_file: Dict[int, np.ndarray], references: np.ndarray,
                   window_frames: int = WINDOW_FRAMES) -> np.ndarray:
    if references.ndim != 2 or references.shape[1] < 2:
        raise ValueError("references must contain file_index and endpoint columns")
    first_hidden = hidden_by_file[int(references[0, 0])]
    batch = np.empty(
        (len(references), first_hidden.shape[1], window_frames), dtype=np.float32
    )
    for row, reference in enumerate(references):
        file_index = int(reference[0])
        endpoint = int(reference[1])
        source = hidden_by_file[file_index]
        start = endpoint - window_frames + 1
        if start < 0 or endpoint >= source.shape[0]:
            raise IndexError(f"Invalid window endpoint file={file_index} endpoint={endpoint}")
        batch[row] = source[start:endpoint + 1].T
    return batch


def infer_dense(api, head, hidden: np.ndarray, first_endpoint: int,
                batch_size: int, step_frames: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    torch = api.torch
    endpoints = np.arange(first_endpoint, hidden.shape[0], step_frames, dtype=np.int64)
    probabilities = np.empty((len(endpoints), len(api.WORDS_REMAP)), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(endpoints), batch_size):
            current = endpoints[start:start + batch_size]
            references = np.column_stack(
                [np.zeros(len(current), dtype=np.int64), current]
            )
            windows = gather_windows({0: hidden}, references)
            tensor = torch.from_numpy(windows).to(api.DEVICE)
            logits = head(tensor)
            probabilities[start:start + len(current)] = (
                torch.softmax(logits, dim=1).cpu().numpy()
            )
    return endpoints, probabilities


def file_profile(api, metadata: dict, endpoints: np.ndarray,
                 probabilities: Dict[str, np.ndarray], smooth_ms: float = 200.0,
                 smoothing: str = "centered") -> ar.FileProfile:
    frame_hz = float(metadata["hidden_frame_hz"])
    if len(endpoints) < 2:
        output_hz = frame_hz
    else:
        output_hz = frame_hz / float(np.median(np.diff(endpoints)))
    width = max(1, int(round(smooth_ms / 1000.0 * output_hz)))
    smoothed = {
        name: ar.smooth_probabilities(api, values, width, smoothing)
        for name, values in probabilities.items()
    }
    return ar.FileProfile(
        file_index=int(metadata["file_index"]),
        filename=str(metadata["raw_file"]),
        duration_s=float(metadata["duration_s"]),
        times_s=endpoints.astype(np.float64) / frame_hz,
        probabilities=probabilities,
        smoothed=smoothed,
        ground_truth=ground_truth_from_metadata(metadata),
        diagnostics={
            "output_hz": output_hz,
            "smoothing_frames": width,
            "n_ground_truth_words": len(metadata["events"]),
        },
    )


def split_indices(api, patient: dict) -> Dict[str, List[int]]:
    train, val, test = api.make_split(
        len(patient["files_list"]), patient["test_start_file_classification_index"]
    )
    return {"train": list(train), "val": list(val), "test": list(test)}
