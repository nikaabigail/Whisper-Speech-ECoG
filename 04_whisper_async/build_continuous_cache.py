#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build persistent FP32 hidden trajectories for continuous-head training.

The source project and HDF5 recordings are opened read-only. Expensive hidden
trajectories are written once to D: with per-file SHA256 and atomic completion,
so an interrupted run resumes without recomputing completed series.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

import async_replay as ar
import continuous_common as cc

sys.dont_write_bytecode = True

FROZEN_BASE_FILES = (
    "bench_models_regression.py",
    "common_preprocessing.py",
    "models_classification.py",
    "models_regression.py",
    "patients.json",
    "runner_classification.py",
    "runner_common.py",
    "runtime.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-project", type=Path, default=ar.DEFAULT_BASE)
    parser.add_argument("--archive", type=Path, default=cc.DEFAULT_ARCHIVE)
    parser.add_argument("--cache-root", type=Path, default=cc.DEFAULT_CACHE)
    parser.add_argument("--layers", nargs="+", type=int, default=cc.LAYERS)
    parser.add_argument("--files", nargs="+", type=int)
    parser.add_argument("--verify-existing-sha", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def source_fingerprint(base: Path, archive: Path) -> dict:
    live_dir = base / "library"
    frozen_dir = archive / "source" / "base_library"
    result = {}
    for name in FROZEN_BASE_FILES:
        live = live_dir / name
        frozen = frozen_dir / name
        if not live.is_file():
            raise FileNotFoundError(f"Missing synchronous source file: {live}")
        live_sha = cc.sha256_file(live)
        if frozen.is_file() and live_sha != cc.sha256_file(frozen):
            raise RuntimeError(
                f"Live base source differs from the frozen baseline: {live}. "
                "Refusing to build a mixed-version cache."
            )
        result[name] = live_sha
    return result


def runtime_fingerprint(api) -> dict:
    import scipy
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": api.torch.__version__,
        "sklearn": api.sklearn.__version__,
        "scipy": scipy.__version__,
        "h5py": api.h5py.__version__,
    }


def manifest_template(layer: int, spec: ar.ModelSpec, checkpoint_sha: str,
                      source_sha: dict, versions: dict, hidden_stride: int) -> dict:
    return {
        "version": cc.CACHE_VERSION,
        "kind": "scaled_fp32_continuous_hidden_cache",
        "patient": cc.PATIENT,
        "seed": cc.SEED,
        "layer": layer,
        "model_name": spec.model_name,
        "regression_checkpoint": str(spec.regression_path),
        "regression_sha256": checkpoint_sha,
        "hidden_stride": int(hidden_stride),
        "word_window_frames": cc.WINDOW_FRAMES,
        "scaling": "sklearn.preprocessing.scale per complete file; float32",
        "frozen_base_source_sha256": source_sha,
        "runtime_versions": versions,
        "files": {},
    }


def load_or_create_manifest(cache_root: Path, layer: int, spec: ar.ModelSpec,
                            source_sha: dict, versions: dict, hidden_stride: int) -> dict:
    path = cc.cache_manifest_path(cache_root, layer)
    checkpoint_sha = cc.sha256_file(spec.regression_path)
    if path.is_file():
        manifest = cc.read_json(path)
        expected = manifest_template(
            layer, spec, checkpoint_sha, source_sha, versions, hidden_stride
        )
        if manifest.get("version") != cc.CACHE_VERSION and not manifest.get("files"):
            # Safe schema upgrade: the previous preflight created only an empty
            # manifest, so there is no cached payload to reinterpret.
            cc.atomic_json(path, expected)
            return expected
        for key in (
            "version", "kind", "patient", "seed", "layer", "model_name",
            "regression_sha256", "hidden_stride", "word_window_frames", "scaling",
            "frozen_base_source_sha256", "runtime_versions",
        ):
            if manifest.get(key) != expected.get(key):
                raise RuntimeError(
                    f"Cache manifest fingerprint mismatch L{layer}, key={key}: "
                    f"{manifest.get(key)!r} != {expected.get(key)!r}"
                )
        return manifest
    manifest = manifest_template(
        layer, spec, checkpoint_sha, source_sha, versions, hidden_stride
    )
    cc.atomic_json(path, manifest)
    return manifest


def entry_is_valid(cache_root: Path, layer: int, file_index: int, manifest: dict,
                   verify_sha: bool, expected_raw: Path) -> bool:
    entry = (manifest.get("files") or {}).get(str(file_index))
    if not entry or not entry.get("complete"):
        return False
    try:
        path = Path(entry["hidden_path"])
        label_file = cc.labels_path(cache_root, file_index)
        metadata_file = cc.metadata_path(cache_root, file_index)
        if (path.resolve() != cc.hidden_path(cache_root, layer, file_index).resolve()
                or Path(entry["labels_path"]).resolve() != label_file.resolve()
                or Path(entry["metadata_path"]).resolve() != metadata_file.resolve()):
            return False
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            return False
        if not label_file.is_file() or not metadata_file.is_file():
            return False
        if (label_file.stat().st_size != int(entry["labels_bytes"])
                or metadata_file.stat().st_size != int(entry["metadata_bytes"])):
            return False
        array = np.load(path, mmap_mode="r")
        labels = np.load(label_file, mmap_mode="r")
        metadata = cc.read_json(metadata_file)
        raw_file = Path(metadata["raw_file"])
        words_file = Path(metadata["words_file"])
        valid = (
            array.dtype == np.float32
            and list(array.shape) == [int(item) for item in entry["shape"]]
            and labels.dtype == np.int16
            and labels.shape == (array.shape[0],)
            and int(metadata.get("version", -1)) == cc.CACHE_VERSION
            and int(metadata.get("file_index", -1)) == file_index
            and int(metadata.get("n_hidden_frames", -1)) == array.shape[0]
            and Path(metadata["raw_file"]).resolve() == expected_raw.resolve()
            and Path(entry["raw_file"]).resolve() == expected_raw.resolve()
            and raw_file.is_file()
            and raw_file.stat().st_size == int(metadata["raw_file_bytes"])
            and raw_file.stat().st_mtime_ns == int(metadata["raw_file_mtime_ns"])
            and raw_file.stat().st_size == int(entry["raw_file_bytes"])
            and raw_file.stat().st_mtime_ns == int(entry["raw_file_mtime_ns"])
            and words_file.is_file()
            and words_file.stat().st_size == int(metadata["words_file_bytes"])
            and words_file.stat().st_mtime_ns == int(metadata["words_file_mtime_ns"])
        )
        del array, labels
    except (OSError, ValueError, EOFError, KeyError, TypeError):
        return False
    if not valid:
        return False
    if verify_sha:
        try:
            checks = (
                (path, entry["sha256"]),
                (label_file, entry["labels_sha256"]),
                (metadata_file, entry["metadata_sha256"]),
                (words_file, metadata["words_file_sha256"]),
            )
            if any(cc.sha256_file(item) != expected for item, expected in checks):
                return False
        except (OSError, KeyError, TypeError):
            return False
    return True


def verify_preprocessing_compatibility(bundles) -> None:
    first = bundles[0].regression
    keys = (
        "downsampling_coef", "LAG_BACKWARD", "LAG_FORWARD", "ECOG_LEAD_MS",
        "HIGH_PASS_HZ", "LOW_PASS_HZ", "SELECTED_CHANNELS",
    )
    for bundle in bundles[1:]:
        current = bundle.regression
        for key in keys:
            if getattr(current, key, None) != getattr(first, key, None):
                raise RuntimeError(
                    f"L{bundle.spec.layer} preprocessing differs at {key}; "
                    "shared preprocessing would be invalid"
                )
        if list(current.selected_channels) != list(first.selected_channels):
            raise RuntimeError(f"L{bundle.spec.layer} effective selected channels differ")
        if current.preprocess_ecog.__func__ is not first.preprocess_ecog.__func__:
            raise RuntimeError(f"L{bundle.spec.layer} preprocessing implementation differs")


def main() -> int:
    args = parse_args()
    layers = sorted(set(args.layers))
    if any(layer not in cc.LAYERS for layer in layers):
        raise ValueError("Only frozen layers 3, 4 and 5 are supported")
    base = args.base_project.resolve()
    archive = args.archive.resolve()
    cache_root = args.cache_root.resolve()
    api = ar.load_base_api(base, cc.SEED)
    api.set_seed(cc.SEED)
    patient = ar.load_patient(base, cc.PATIENT)
    splits = cc.split_indices(api, patient)
    all_indices = splits["train"] + splits["val"] + splits["test"]
    file_indices = list(dict.fromkeys(args.files if args.files is not None else all_indices))
    if any(index not in all_indices for index in file_indices):
        raise ValueError(f"Unexpected file index in {file_indices}")
    for file_index in file_indices:
        raw_file = Path(patient["files_list"][file_index])
        words_file = Path(api.get_words_filepath(str(raw_file)))
        if not raw_file.is_file() or not words_file.is_file():
            raise FileNotFoundError(
                f"Raw/annotation input is missing for file {file_index}: "
                f"{raw_file} / {words_file}"
            )

    specs = [cc.archive_spec(archive, layer) for layer in layers]
    ar.validate_checkpoint_splits(api, patient, specs)
    source_sha = source_fingerprint(base, archive)
    versions = runtime_fingerprint(api)
    checkpoint_shas = {spec.layer: cc.sha256_file(spec.regression_path) for spec in specs}
    print(f"[runtime] {api.device_str()}")
    print(f"[archive] {archive}")
    print(f"[cache] {cache_root}")
    print(f"[files] {file_indices} | layers={layers}")
    print(f"[disk] free={shutil.disk_usage(cache_root.anchor).free / 2**30:.1f} GiB")
    for spec in specs:
        print(
            f"[L{spec.layer}] encoder={spec.regression_path.name} "
            f"sha256={checkpoint_shas[spec.layer][:12]}..."
        )
    if args.preflight:
        print("[preflight] OK; no cache/HDF5 write and no checkpoint loaded")
        return 0


    manifests = {
        spec.layer: load_or_create_manifest(
            cache_root, spec.layer, spec, source_sha, versions, api.HIDDEN_STRIDE
        )
        for spec in specs
    }

    missing_by_file = {
        file_index: [
            layer for layer in layers
            if not entry_is_valid(
                cache_root, layer, file_index, manifests[layer], args.verify_existing_sha,
                Path(patient["files_list"][file_index]),
            )
        ]
        for file_index in file_indices
    }
    if not any(missing_by_file.values()):
        for file_index in file_indices:
            print(f"[file {file_index:02d}] complete -> skip")
        print("[done] persistent cache complete for requested files/layers")
        print("[safety] source project and frozen archive were read only; no checkpoint deleted")
        return 0

    bundles = [cc.load_frozen_bundle(api, patient, archive, layer) for layer in layers]
    verify_preprocessing_compatibility(bundles)
    preprocessing_regression = bundles[0].regression

    for file_index in file_indices:
        missing_layers = missing_by_file[file_index]
        if not missing_layers:
            print(f"[file {file_index:02d}] complete -> skip")
            continue

        filepath = Path(patient["files_list"][file_index])
        print(f"[file {file_index:02d}] read-only HDF5 -> preprocess | missing={missing_layers}")
        with api.h5py.File(filepath, "r") as handle:
            samples = handle["RawData"]["Samples"]
            raw_samples = int(samples.shape[0])
            ecog = samples[:, patient["ecog_channels"]].astype("double")
        preprocessed = preprocessing_regression.preprocess_ecog(
            ecog, patient["sampling_rate"]
        ).astype("float32")
        preprocessed = api.shift_ecog_lead(
            preprocessed,
            getattr(preprocessing_regression, "ECOG_LEAD_MS", 0),
            patient["sampling_rate"] / preprocessing_regression.downsampling_coef,
        )
        del ecog

        expected_frames = None
        for bundle in bundles:
            layer = bundle.spec.layer
            if layer not in missing_layers:
                continue
            print(f"  [L{layer}] frozen encoder -> hidden 3030D", flush=True)
            hidden = api.predict_regression_hidden(
                bundle.regression, preprocessed, api.HIDDEN_STRIDE
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scaled = api.sklearn.preprocessing.scale(hidden, copy=False)
            if scaled.dtype != np.float32:
                scaled = scaled.astype(np.float32)
            if not np.isfinite(scaled).all():
                raise FloatingPointError(f"Non-finite hidden cache L{layer} file {file_index}")
            if expected_frames is None:
                expected_frames = int(scaled.shape[0])
            elif scaled.shape[0] != expected_frames:
                raise RuntimeError("Layer hidden timelines are not aligned")

            label_file = cc.labels_path(cache_root, file_index)
            metadata_file = cc.metadata_path(cache_root, file_index)
            if not label_file.is_file() or not metadata_file.is_file():
                labels, metadata = cc.build_labels_and_metadata(
                    api, patient, bundle.regression, file_index,
                    int(scaled.shape[0]), raw_samples,
                )
                cc.atomic_numpy(label_file, labels)
                cc.atomic_json(metadata_file, metadata)
            else:
                metadata = cc.read_json(metadata_file)
                labels = np.load(label_file, mmap_mode="r")
                if (int(metadata.get("version", -1)) != cc.CACHE_VERSION
                        or int(metadata["n_hidden_frames"]) != int(scaled.shape[0])
                        or labels.dtype != np.int16
                        or labels.shape != (int(scaled.shape[0]),)):
                    raise RuntimeError(f"Existing metadata mismatch for file {file_index}")
                del labels

            output = cc.hidden_path(cache_root, layer, file_index)
            cc.atomic_numpy(output, scaled)
            checksum = cc.sha256_file(output)
            manifests[layer].setdefault("files", {})[str(file_index)] = {
                "complete": True,
                "hidden_path": str(output),
                "shape": [int(item) for item in scaled.shape],
                "dtype": str(scaled.dtype),
                "bytes": output.stat().st_size,
                "sha256": checksum,
                "labels_path": str(label_file),
                "labels_bytes": label_file.stat().st_size,
                "labels_sha256": cc.sha256_file(label_file),
                "metadata_path": str(metadata_file),
                "metadata_bytes": metadata_file.stat().st_size,
                "metadata_sha256": cc.sha256_file(metadata_file),
                "raw_file": str(filepath),
                "raw_file_bytes": filepath.stat().st_size,
                "raw_file_mtime_ns": filepath.stat().st_mtime_ns,
                "scale_warnings": [str(item.message) for item in caught],
            }
            cc.atomic_json(cc.cache_manifest_path(cache_root, layer), manifests[layer])
            print(
                f"  [saved L{layer}] {output} | {output.stat().st_size / 2**20:.1f} MiB "
                f"| sha={checksum[:12]}...",
                flush=True,
            )
            del hidden, scaled
            gc.collect()
            if api.torch.cuda.is_available():
                api.torch.cuda.empty_cache()
        del preprocessed
        gc.collect()

    print("[done] persistent cache complete for requested files/layers")
    print("[safety] source project and frozen archive were read only; no checkpoint deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
