#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune frozen synchronous word heads on continuous hidden trajectories.

The three Whisper-guided ECoG encoders remain frozen.  Each Mel2WordHidden
classifier is warm-started from the immutable seed-4 synchronous checkpoint and
is trained on trailing 52-frame windows sampled from continuous recordings.
The upstream seed and the stochastic continuous-head training seed are tracked
separately so repeated head fits can reuse exactly the same frozen features.

The test split is protected by a structural gate: no test cache is opened until
all three production heads have independent completion records.  A smoke run is
isolated in its own directory, uses only L3/files 0 and 9, and never evaluates
test data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import async_replay as ar
import continuous_common as cc

sys.dont_write_bytecode = True


SCRIPT_VERSION = 2
RUNS_ROOT = cc.REPOSITORY_ROOT / "artifacts" / "continuous_runs"
PRODUCTION_LAYERS = (3, 4, 5)
ENSEMBLE_NAME = "L3+L4+L5"
AMP_OVERFLOW_HOTFIX_ID = "amp_grad_scaler_overflow_v1"
AMP_OVERFLOW_HOTFIX_MANIFEST = cc.ROOT / "amp_overflow_hotfix_manifest.json"
AMP_OVERFLOW_COMPATIBLE_PREVIOUS_SHA256 = {
    "76717e66d1d880ce523fea25bc42844335d237c8a205c1c5c0649c5a62115982",
}
PLOT_RENDER_HOTFIX_ID = "plot_title_seed_scope_v1"
PLOT_RENDER_COMPATIBLE_PREVIOUS_SHA256 = {
    # First generalized multiseed trainer. Training and fixed-test metrics are
    # valid; evaluation stopped only while formatting the PR-plot title.
    "42004c10977196683287532df9bb98cab420654fb452191701c0faf665c08d41",
}
MAX_AMP_OVERFLOW_SKIPS_PER_EPOCH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm-start and fine-tune L3/L4/L5 word heads on continuous windows; "
            "frozen encoders and source project are read-only."
        )
    )
    parser.add_argument("--base-project", type=Path, default=ar.DEFAULT_BASE)
    parser.add_argument("--archive", type=Path, default=cc.DEFAULT_ARCHIVE)
    parser.add_argument("--cache-root", type=Path, default=cc.DEFAULT_CACHE)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--run-name", default="continuous_finetune_seed4_v1")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--train-only", action="store_true",
        help=(
            "Train/validate and fix all heads, then exit before opening test caches. "
            "A later invocation without this flag reuses completed heads and evaluates test."
        ),
    )
    parser.add_argument("--verify-cache-sha", action="store_true")

    parser.add_argument(
        "--seed", type=int, default=cc.SEED,
        help="Stochastic seed for continuous-head sampling and optimization.",
    )
    parser.add_argument(
        "--source-seed", type=int, default=cc.SEED,
        help="Provenance seed of the frozen encoder/head archive and hidden cache.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dense-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--positives-per-word", type=int, default=4)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    parser.add_argument("--hard-negative-ms", type=float, default=500.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--smooth-ms", type=float, default=200.0)
    parser.add_argument("--smoothing", choices=("centered", "causal"), default="centered")
    parser.add_argument("--threshold-points", type=int, default=201)
    parser.add_argument("--null-permutations", type=int, default=50)
    parser.add_argument("--allow-cpu", action="store_true")

    # Smoke settings are intentionally bounded and do not alter production defaults.
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-val-windows", type=int, default=2048)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.source_seed != cc.SEED:
        raise ValueError(
            f"This frozen upstream experiment is seed={cc.SEED}; "
            f"got source-seed={args.source_seed}"
        )
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("continuous-head seed must be in [0, 2**32 - 1]")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("run-name may contain only letters, digits, dot, dash and underscore")
    positive_ints = {
        "batch-size": args.batch_size,
        "dense-batch-size": args.dense_batch_size,
        "epochs": args.epochs,
        "min-epochs": args.min_epochs,
        "patience": args.patience,
        "positives-per-word": args.positives_per_word,
        "smoke-steps": args.smoke_steps,
        "smoke-val-windows": args.smoke_val_windows,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.min_epochs > args.epochs:
        raise ValueError("min-epochs cannot exceed epochs")
    if not 0.0 <= args.hard_negative_fraction <= 1.0:
        raise ValueError("hard-negative-fraction must be in [0, 1]")
    if args.hard_negative_ms < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid negative-sampling or optimizer parameters")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label-smoothing must be in [0, 1)")
    if args.grad_clip <= 0 or args.smooth_ms < 0:
        raise ValueError("grad-clip must be positive and smooth-ms non-negative")
    if args.threshold_points < 2 or args.null_permutations < 0:
        raise ValueError("threshold-points must be >=2 and null-permutations non-negative")


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        ar.json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def torch_load_checkpoint(torch, path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_torch_save(torch, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_selected_checkpoint(torch, path: Path, device, fingerprint: str,
                                 layer: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Selected best checkpoint is missing: {path}")
    payload = torch_load_checkpoint(torch, path, device)
    if payload.get("kind") != "continuous_word_head_best":
        raise RuntimeError(f"Unexpected selected-checkpoint kind: {path}")
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError(f"Selected checkpoint config mismatch: {path}")
    if int(payload.get("layer", -1)) != int(layer):
        raise RuntimeError(f"Selected checkpoint layer mismatch: {path}")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"Selected checkpoint contains no model state: {path}")
    return payload


def experiment_layout(args: argparse.Namespace, splits: dict) -> dict:
    if args.smoke:
        return {
            "mode": "smoke",
            "layers": [3],
            "train_files": [int(splits["train"][0])],
            "val_files": [int(splits["val"][0])],
            "test_files": [],
            "max_epochs": 1,
            "min_epochs": 1,
            "patience": 1,
            "max_steps_per_epoch": int(args.smoke_steps),
            "max_val_windows": int(args.smoke_val_windows),
        }
    return {
        "mode": "production",
        "layers": list(PRODUCTION_LAYERS),
        "train_files": [int(item) for item in splits["train"]],
        "val_files": [int(item) for item in splits["val"]],
        # This list is provenance only. No test cache is touched before the gate.
        "test_files": [int(item) for item in splits["test"]],
        "max_epochs": int(args.epochs),
        "min_epochs": int(args.min_epochs),
        "patience": int(args.patience),
        "max_steps_per_epoch": None,
        "max_val_windows": None,
    }


def load_experiment_context(args: argparse.Namespace):
    base = args.base_project.resolve()
    archive = args.archive.resolve()
    cache_root = args.cache_root.resolve()
    api = ar.load_base_api(base, args.source_seed)
    api.set_seed(args.seed)
    patient = ar.load_patient(base, cc.PATIENT)
    splits = cc.split_indices(api, patient)
    layout = experiment_layout(args, splits)
    specs = {layer: cc.archive_spec(archive, layer) for layer in layout["layers"]}
    ar.validate_checkpoint_splits(api, patient, list(specs.values()))
    for layer, spec in specs.items():
        if spec.seed != args.source_seed or spec.max_words_length != cc.WINDOW_FRAMES:
            raise RuntimeError(
                f"Frozen L{layer} provenance mismatch: seed={spec.seed}, "
                f"expected source seed={args.source_seed}, "
                f"window={spec.max_words_length}"
            )
    return api, patient, base, archive, cache_root, splits, layout, specs


def inspect_cache_set(cache_root: Path, layers: Sequence[int], file_indices: Sequence[int],
                      specs: Dict[int, ar.ModelSpec], verify_sha: bool,
                      strict: bool) -> dict:
    """Inspect train/validation caches only; this function is never given test indices."""
    summary = {"complete": [], "missing": [], "errors": []}
    shapes: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for layer in layers:
        manifest_path = cc.cache_manifest_path(cache_root, layer)
        if manifest_path.is_file():
            manifest = None
            try:
                manifest = cc.read_json(manifest_path)
                expected_manifest = {
                    "version": cc.CACHE_VERSION,
                    "kind": "scaled_fp32_continuous_hidden_cache",
                    "patient": cc.PATIENT,
                    "seed": cc.SEED,
                    "layer": layer,
                    "model_name": specs[layer].model_name,
                    "regression_sha256": cc.sha256_file(specs[layer].regression_path),
                    "hidden_stride": 10,
                    "word_window_frames": cc.WINDOW_FRAMES,
                }
                for key, expected in expected_manifest.items():
                    if manifest.get(key) != expected:
                        raise RuntimeError(
                            f"manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}"
                        )
            except Exception as exc:
                destination = (
                    summary["missing"]
                    if isinstance(manifest, dict) and not (manifest.get("files") or {})
                    else summary["errors"]
                )
                destination.append({
                    "cache": f"L{layer}/manifest", "reason": str(exc)
                })
        for file_index in file_indices:
            label = f"L{layer}/file_{file_index:02d}"
            try:
                cached = cc.load_cached_file(
                    cache_root, layer, file_index, verify_sha=verify_sha
                )
                labels = cc.open_labels(cached)
                metadata = cc.read_json(cached.metadata_path)
                if labels.dtype != np.int16 or labels.shape != (cached.n_frames,):
                    raise RuntimeError(
                        f"labels dtype/shape mismatch: {labels.dtype}/{labels.shape}"
                    )
                if int(metadata.get("word_window_frames", -1)) != cc.WINDOW_FRAMES:
                    raise RuntimeError("metadata window fingerprint mismatch")
                if cached.first_endpoint < cc.WINDOW_FRAMES - 1:
                    raise RuntimeError("invalid first continuous endpoint")
                shapes[(layer, file_index)] = (cached.n_frames, cached.hidden_dim)
                summary["complete"].append(label)
                del labels
            except FileNotFoundError as exc:
                summary["missing"].append({"cache": label, "reason": str(exc)})
            except RuntimeError as exc:
                if "cache is incomplete" in str(exc):
                    summary["missing"].append({"cache": label, "reason": str(exc)})
                else:
                    summary["errors"].append({"cache": label, "reason": str(exc)})
            except Exception as exc:
                summary["errors"].append({"cache": label, "reason": str(exc)})

    for file_index in file_indices:
        available = [shapes[(layer, file_index)] for layer in layers if (layer, file_index) in shapes]
        if available and any(shape != available[0] for shape in available[1:]):
            summary["errors"].append({
                "cache": f"file_{file_index:02d}",
                "reason": f"layer timelines differ: {available}",
            })
    hidden_dims = {shape[1] for shape in shapes.values()}
    if len(hidden_dims) > 1:
        summary["errors"].append({"cache": "all", "reason": f"hidden dims differ: {hidden_dims}"})

    if strict and (summary["missing"] or summary["errors"]):
        first = (summary["errors"] + summary["missing"])[0]
        raise RuntimeError(
            f"Required train/validation cache is not ready ({first['cache']}): "
            f"{first['reason']}"
        )
    return summary


def build_run_config(args: argparse.Namespace, base: Path, archive: Path, cache_root: Path,
                     layout: dict, specs: Dict[int, ar.ModelSpec]) -> dict:
    source_models = []
    for layer in layout["layers"]:
        spec = specs[layer]
        source_models.append({
            "layer": layer,
            "model_name": spec.model_name,
            "seed": spec.seed,
            "max_words_length": spec.max_words_length,
            "regression_checkpoint": str(spec.regression_path),
            "regression_sha256": cc.sha256_file(spec.regression_path),
            "synchronous_head": str(spec.classifier_path),
            "synchronous_head_sha256": cc.sha256_file(spec.classifier_path),
            "synchronous_accuracy": spec.synchronous_accuracy,
        })
    payload = {
        "script_version": SCRIPT_VERSION,
        "kind": "continuous_word_head_finetune",
        "patient": cc.PATIENT,
        "seed": args.seed,
        "seed_role": "continuous_head_training_seed",
        "source_seed": args.source_seed,
        "source_seed_role": "frozen_encoder_and_synchronous_head_seed",
        "mode": layout["mode"],
        "base_project_read_only": str(base),
        "frozen_archive_read_only": str(archive),
        "cache_root": str(cache_root),
        "runs_root": str(args.runs_root.resolve()),
        "source_code_sha256": {
            "train_continuous_heads.py": cc.sha256_file(Path(__file__).resolve()),
            "continuous_common.py": cc.sha256_file(Path(cc.__file__).resolve()),
            "async_replay.py": cc.sha256_file(Path(ar.__file__).resolve()),
        },
        "layout": layout,
        "architecture": {
            "frozen_encoders": True,
            "head": "Mel2WordHidden, all head parameters trainable",
            "input": "trailing scaled hidden window [3030, 52] at endpoint e",
            "endpoint_label": "word class while endpoint is inside a word; otherwise class 0",
            "word_boundaries_used_at_inference": False,
        },
        "sampler": {
            "positives_per_word": args.positives_per_word,
            "positive_rule": "one random endpoint from each word quartile",
            "negative_to_positive_ratio": 1.0,
            "hard_negative_fraction": args.hard_negative_fraction,
            "hard_negative_radius_ms": args.hard_negative_ms,
            "hard_negative_rule": "true silence within radius of annotated boundaries",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "label_smoothing": args.label_smoothing,
            "gradient_clip_norm": args.grad_clip,
            "amp_on_cuda": True,
        },
        "selection": {
            "primary": "validation event F1",
            "tie_breakers": ["validation event PR-AUC", "lower validation dense CE"],
            "smooth_ms": args.smooth_ms,
            "smoothing": args.smoothing,
            "threshold_points": args.threshold_points,
        },
        "final_evaluation": {
            "production_system": ENSEMBLE_NAME,
            "ensemble": "arithmetic mean of L3/L4/L5 softmax probabilities",
            "threshold_selected_on": "validation",
            "test_open_gate": "all three per-layer completed records",
            "null_permutations": args.null_permutations,
        },
        "source_models": source_models,
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def effective_run_dir(args: argparse.Namespace) -> Path:
    suffix = "__smoke" if args.smoke else ""
    return args.runs_root.resolve() / f"{args.run_name}{suffix}"


def _config_without_trainer_sha(config: dict) -> Tuple[dict, Optional[str]]:
    normalized = copy.deepcopy(config)
    normalized.pop("fingerprint", None)
    sources = normalized.get("source_code_sha256") or {}
    trainer_sha = sources.pop("train_continuous_heads.py", None)
    return normalized, trainer_sha


def ensure_run_config(run_dir: Path, config: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_config.json"
    if path.is_file():
        existing = cc.read_json(path)
        if existing.get("fingerprint") == config["fingerprint"]:
            return existing

        existing_payload = copy.deepcopy(existing)
        existing_fingerprint = existing_payload.pop("fingerprint", None)
        if canonical_hash(existing_payload) != existing_fingerprint:
            raise RuntimeError(f"Existing run config fingerprint is invalid: {path}")
        existing_normalized, previous_trainer_sha = _config_without_trainer_sha(existing)
        current_normalized, current_trainer_sha = _config_without_trainer_sha(config)
        hotfix_manifest = (
            cc.read_json(AMP_OVERFLOW_HOTFIX_MANIFEST)
            if AMP_OVERFLOW_HOTFIX_MANIFEST.is_file() else {}
        )
        manifest_is_exact = (
            hotfix_manifest.get("hotfix_id") == AMP_OVERFLOW_HOTFIX_ID
            and hotfix_manifest.get("previous_trainer_sha256") == previous_trainer_sha
            and hotfix_manifest.get("patched_trainer_sha256") == current_trainer_sha
            and hotfix_manifest.get("allowed_run_fingerprint") == existing_fingerprint
            and hotfix_manifest.get("allowed_run_name") == run_dir.name
        )
        compatible_amp_hotfix = (
            previous_trainer_sha in AMP_OVERFLOW_COMPATIBLE_PREVIOUS_SHA256
            and existing_normalized == current_normalized
            and manifest_is_exact
        )
        plot_hotfix_dir = run_dir / "runtime_hotfixes"
        plot_hotfix_path = plot_hotfix_dir / f"{PLOT_RENDER_HOTFIX_ID}.json"
        plot_hotfix_record = (
            cc.read_json(plot_hotfix_path) if plot_hotfix_path.is_file() else {}
        )
        first_plot_hotfix_application = (
            previous_trainer_sha in PLOT_RENDER_COMPATIBLE_PREVIOUS_SHA256
            and existing_normalized == current_normalized
            and not plot_hotfix_record
        )
        recorded_plot_hotfix_application = (
            plot_hotfix_record.get("hotfix_id") == PLOT_RENDER_HOTFIX_ID
            and plot_hotfix_record.get("previous_trainer_sha256") == previous_trainer_sha
            and plot_hotfix_record.get("patched_trainer_sha256") == current_trainer_sha
            and plot_hotfix_record.get("allowed_run_fingerprint") == existing_fingerprint
            and plot_hotfix_record.get("allowed_run_name") == run_dir.name
            and existing_normalized == current_normalized
        )
        compatible_plot_hotfix = (
            first_plot_hotfix_application or recorded_plot_hotfix_application
        )
        if not compatible_amp_hotfix and not compatible_plot_hotfix:
            raise RuntimeError(
                f"Run directory belongs to a different configuration: {run_dir}. "
                "Use a new --run-name."
            )

        if compatible_plot_hotfix:
            now = datetime.now().isoformat(timespec="seconds")
            record = plot_hotfix_record or {
                "kind": "runtime_compatibility_hotfix",
                "hotfix_id": PLOT_RENDER_HOTFIX_ID,
                "first_applied": now,
                "run_config_fingerprint_preserved": existing_fingerprint,
                "allowed_run_fingerprint": existing_fingerprint,
                "allowed_run_name": run_dir.name,
                "previous_trainer_sha256": previous_trainer_sha,
                "patched_trainer_sha256": current_trainer_sha,
                "reason": (
                    "The fixed-test metrics were computed successfully, but PR-plot "
                    "title formatting referenced args outside its function scope."
                ),
                "behavior": (
                    "Pass source/head seeds explicitly to the plot helper and treat "
                    "plot rendering as a non-scientific optional artifact."
                ),
                "scientific_state_preserved": (
                    "Data, split, sampling, architecture, trained weights, selected "
                    "checkpoints, thresholds and all fixed-test metrics are unchanged."
                ),
            }
            record["last_verified"] = now
            record["current_trainer_sha256"] = current_trainer_sha
            cc.atomic_json(plot_hotfix_path, record)
            print(
                f"[hotfix] {PLOT_RENDER_HOTFIX_ID}: preserving existing run "
                "fingerprint and reusing fixed heads",
                flush=True,
            )
            return existing

        hotfix_dir = run_dir / "runtime_hotfixes"
        hotfix_path = hotfix_dir / f"{AMP_OVERFLOW_HOTFIX_ID}.json"
        now = datetime.now().isoformat(timespec="seconds")
        record = cc.read_json(hotfix_path) if hotfix_path.is_file() else {
            "kind": "runtime_compatibility_hotfix",
            "hotfix_id": AMP_OVERFLOW_HOTFIX_ID,
            "first_applied": now,
            "run_config_fingerprint_preserved": existing_fingerprint,
            "previous_trainer_sha256": previous_trainer_sha,
            "compatibility_manifest": str(AMP_OVERFLOW_HOTFIX_MANIFEST),
            "compatibility_manifest_sha256": cc.sha256_file(
                AMP_OVERFLOW_HOTFIX_MANIFEST
            ),
            "reason": (
                "CUDA AMP GradScaler doubled from 65536 to 131072 after 2000 "
                "successful steps; an expected overflow was incorrectly fatal."
            ),
            "behavior": (
                "On a non-finite AMP gradient, GradScaler skips exactly that optimizer "
                "step and backs off its scale. FP32 non-finite gradients remain fatal; "
                "repeated AMP overflows remain bounded and fatal. Resume RNG tensors are "
                "restored on the device format required by PyTorch."
            ),
            "scientific_state_preserved": (
                "Data, split, sampling, architecture, model weights, optimizer state, "
                "validation selection and test gate are unchanged."
            ),
        }
        record["last_verified"] = now
        record["current_trainer_sha256"] = current_trainer_sha
        cc.atomic_json(hotfix_path, record)
        print(
            f"[hotfix] {AMP_OVERFLOW_HOTFIX_ID}: preserving existing run fingerprint "
            "and resuming committed checkpoints",
            flush=True,
        )
        return existing
    else:
        cc.atomic_json(path, config)
        return config


def load_warm_head(api, spec: ar.ModelSpec, hidden_dim: int):
    head = api.mcls.Mel2WordHidden(hidden_dim, len(api.WORDS_REMAP)).to(api.DEVICE)
    source_state = ar.torch_load_weights(api.torch, spec.classifier_path, api.DEVICE)
    head.load_state_dict(source_state, strict=True)
    head.requires_grad_(True)
    return head


def prepare_training_data(cache_root: Path, layer: int, file_indices: Sequence[int],
                          verify_sha: bool, hard_negative_ms: float) -> dict:
    cached_by_file = {}
    hidden_by_file = {}
    labels_by_file = {}
    metadata_by_file = {}
    positive_segments: List[Tuple[int, int, int, int, int]] = []
    hard_pools = []
    far_pools = []
    skipped_events = 0

    for file_index in file_indices:
        cached = cc.load_cached_file(cache_root, layer, file_index, verify_sha=verify_sha)
        hidden = cc.open_hidden(cached)
        labels = cc.open_labels(cached)
        metadata = cc.read_json(cached.metadata_path)
        cached_by_file[file_index] = cached
        hidden_by_file[file_index] = hidden
        labels_by_file[file_index] = labels
        metadata_by_file[file_index] = metadata

        for event in metadata["events"]:
            start = max(cached.first_endpoint, int(event["hidden_start"]))
            end = min(cached.n_frames, int(event["hidden_end"]))
            class_index = int(event["class_index"])
            if end <= start:
                skipped_events += 1
                continue
            if not np.all(labels[start:end] == class_index):
                raise RuntimeError(
                    f"Positive label mismatch L{layer} file={file_index} "
                    f"event={event['event_index']}"
                )
            positive_segments.append(
                (file_index, start, end, class_index, int(event["event_index"]))
            )

        valid = np.zeros(cached.n_frames, dtype=bool)
        valid[cached.first_endpoint:] = True
        background = valid & (labels == 0)
        hard = np.zeros(cached.n_frames, dtype=bool)
        radius = int(round(hard_negative_ms / 1000.0 * cached.frame_hz))
        for event in metadata["events"]:
            for boundary in (int(event["hidden_start"]), int(event["hidden_end"])):
                left = max(cached.first_endpoint, boundary - radius)
                right = min(cached.n_frames, boundary + radius + 1)
                if right > left:
                    hard[left:right] = True
        hard &= background
        far = background & ~hard
        hard_endpoints = np.flatnonzero(hard).astype(np.int64, copy=False)
        far_endpoints = np.flatnonzero(far).astype(np.int64, copy=False)
        if len(hard_endpoints):
            hard_pools.append(np.column_stack([
                np.full(len(hard_endpoints), file_index, dtype=np.int64), hard_endpoints
            ]))
        if len(far_endpoints):
            far_pools.append(np.column_stack([
                np.full(len(far_endpoints), file_index, dtype=np.int64), far_endpoints
            ]))

    if not positive_segments:
        raise RuntimeError(f"No valid positive word segments for L{layer}")
    hard_pool = np.concatenate(hard_pools, axis=0) if hard_pools else np.empty((0, 2), np.int64)
    far_pool = np.concatenate(far_pools, axis=0) if far_pools else np.empty((0, 2), np.int64)
    if len(hard_pool) + len(far_pool) == 0:
        raise RuntimeError(f"No valid background endpoints for L{layer}")
    return {
        "cached": cached_by_file,
        "hidden": hidden_by_file,
        "labels": labels_by_file,
        "metadata": metadata_by_file,
        "positive_segments": positive_segments,
        "hard_pool": hard_pool,
        "far_pool": far_pool,
        "skipped_events": skipped_events,
    }


def choose_from_pool(rng: np.random.Generator, pool: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.int64)
    if len(pool) == 0:
        raise RuntimeError("Cannot sample from an empty background pool")
    selected = rng.choice(len(pool), size=count, replace=len(pool) < count)
    return pool[selected]


def sample_epoch_references(data: dict, rng: np.random.Generator,
                            positives_per_word: int,
                            hard_negative_fraction: float) -> Tuple[np.ndarray, dict]:
    positives = []
    for file_index, start, end, class_index, _event_index in data["positive_segments"]:
        length = end - start
        for quartile in range(positives_per_word):
            left = start + (quartile * length) // positives_per_word
            right = start + ((quartile + 1) * length) // positives_per_word
            if right > left:
                endpoint = int(rng.integers(left, right))
            else:
                endpoint = start + min(length - 1, (quartile * length) // positives_per_word)
            positives.append((file_index, endpoint, class_index))
    positive_refs = np.asarray(positives, dtype=np.int64)
    n_positive = len(positive_refs)
    requested_hard = int(round(n_positive * hard_negative_fraction))
    requested_far = n_positive - requested_hard
    hard_pool = data["hard_pool"]
    far_pool = data["far_pool"]
    if len(hard_pool) == 0:
        requested_far = n_positive
        requested_hard = 0
    elif len(far_pool) == 0:
        requested_hard = n_positive
        requested_far = 0
    hard_refs = choose_from_pool(rng, hard_pool, requested_hard)
    far_refs = choose_from_pool(rng, far_pool, requested_far)
    negatives = np.concatenate([hard_refs, far_refs], axis=0)
    negative_refs = np.column_stack([
        negatives, np.zeros(len(negatives), dtype=np.int64)
    ])
    references = np.concatenate([positive_refs, negative_refs], axis=0)
    references = references[rng.permutation(len(references))]
    return references, {
        "positive": n_positive,
        "negative": len(negative_refs),
        "hard_negative": len(hard_refs),
        "far_negative": len(far_refs),
    }


def create_grad_scaler(torch, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_context(torch, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def train_epoch(api, head, optimizer, scaler, criterion, hidden_by_file: dict,
                references: np.ndarray, batch_size: int, grad_clip: float,
                amp_enabled: bool, max_steps: Optional[int]) -> dict:
    torch = api.torch
    head.train()
    total_loss = 0.0
    total_examples = 0
    batches_seen = 0
    optimizer_steps = 0
    amp_overflow_skips = 0
    amp_scale_start = float(scaler.get_scale())
    for start in range(0, len(references), batch_size):
        if max_steps is not None and batches_seen >= max_steps:
            break
        batch = references[start:start + batch_size]
        windows = cc.gather_windows(hidden_by_file, batch[:, :2])
        inputs = torch.from_numpy(windows).to(api.DEVICE, non_blocking=True)
        targets = torch.from_numpy(batch[:, 2].copy()).long().to(api.DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(torch, amp_enabled):
            logits = head(inputs)
            loss = criterion(logits, targets)
        loss_is_finite = bool(torch.isfinite(loss.detach()))
        if not loss_is_finite:
            raise FloatingPointError(
                f"Non-finite loss before backward at batch {batches_seen + 1}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        try:
            # error_if_nonfinite raises before gradients are modified. That lets
            # the overflow path distinguish a true AMP found-inf from the much
            # rarer case where only the aggregate norm calculation overflowed.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), grad_clip, error_if_nonfinite=True
            )
            gradients_are_finite = True
        except RuntimeError as clip_error:
            individual_gradients_are_finite = all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in head.parameters()
                if parameter.grad is not None
            )
            if individual_gradients_are_finite:
                raise FloatingPointError(
                    f"Non-finite aggregate gradient norm at batch {batches_seen + 1} "
                    "although every individual gradient is finite"
                ) from clip_error
            gradients_are_finite = False
        if not gradients_are_finite:
            if not amp_enabled:
                raise FloatingPointError(
                    f"Non-finite FP32 gradient at batch {batches_seen + 1}"
                )
            scale_before = float(scaler.get_scale())
            # unscale_ has already recorded found-inf for this optimizer.
            # The standard GradScaler path therefore skips optimizer.step,
            # backs off the scale and resets its growth tracker to zero.
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            amp_overflow_skips += 1
            batches_seen += 1
            print(
                f"[amp] overflow at batch {batches_seen}: optimizer step skipped; "
                f"scale {scale_before:g} -> {scale_after:g}",
                flush=True,
            )
            if (scale_after >= scale_before
                    or amp_overflow_skips > MAX_AMP_OVERFLOW_SKIPS_PER_EPOCH):
                raise FloatingPointError(
                    "AMP overflow did not back off safely or repeated too often: "
                    f"count={amp_overflow_skips}, scale={scale_before:g}->{scale_after:g}"
                )
            continue
        scaler.step(optimizer)
        scaler.update()
        size = len(batch)
        total_loss += float(loss.detach().cpu()) * size
        total_examples += size
        batches_seen += 1
        optimizer_steps += 1
    if optimizer_steps == 0:
        raise RuntimeError("Training epoch contained no optimizer steps")
    return {
        "loss": total_loss / total_examples,
        "examples": total_examples,
        "steps": batches_seen,
        "optimizer_steps": optimizer_steps,
        "amp_overflow_skips": amp_overflow_skips,
        "amp_scale_start": amp_scale_start,
        "amp_scale_end": float(scaler.get_scale()),
    }


def crop_metadata_for_smoke(metadata: dict, endpoints: np.ndarray) -> dict:
    cropped = copy.deepcopy(metadata)
    frame_hz = float(metadata["hidden_frame_hz"])
    duration_s = float(endpoints[-1]) / frame_hz
    cropped["duration_s"] = duration_s
    cropped["events"] = [
        event for event in cropped["events"] if float(event["end_s"]) <= duration_s
    ]
    return cropped


def evaluate_validation(api, head, cache_root: Path, layer: int,
                        val_indices: Sequence[int], args: argparse.Namespace,
                        max_val_windows: Optional[int]) -> Tuple[dict, dict]:
    system = f"L{layer}"
    profiles = []
    negative_log_likelihood = 0.0
    n_labels = 0
    for file_index in val_indices:
        cached = cc.load_cached_file(
            cache_root, layer, file_index, verify_sha=args.verify_cache_sha
        )
        hidden = cc.open_hidden(cached)
        metadata = cc.read_json(cached.metadata_path)
        if max_val_windows is not None:
            stop = min(cached.n_frames, cached.first_endpoint + max_val_windows)
            hidden_for_eval = hidden[:stop]
        else:
            hidden_for_eval = hidden
        endpoints, probabilities = cc.infer_dense(
            api, head, hidden_for_eval, cached.first_endpoint,
            args.dense_batch_size, step_frames=1,
        )
        labels = cc.open_labels(cached)
        target = np.asarray(labels[endpoints], dtype=np.int64)
        selected = probabilities[np.arange(len(endpoints)), target]
        negative_log_likelihood += float(-np.log(np.maximum(selected, 1e-12)).sum())
        n_labels += len(target)
        if max_val_windows is not None:
            metadata = crop_metadata_for_smoke(metadata, endpoints)
        profiles.append(cc.file_profile(
            api, metadata, endpoints, {system: probabilities},
            smooth_ms=args.smooth_ms, smoothing=args.smoothing,
        ))
        del hidden, hidden_for_eval, labels, probabilities
    event_result = ar.event_pr_curve(
        api, profiles, system, args.threshold_points, null_permutations=0
    )
    metrics = {
        "event_f1": float(event_result["best_f1_posthoc"]["f1"]),
        "event_precision": float(event_result["best_f1_posthoc"]["precision"]),
        "event_recall": float(event_result["best_f1_posthoc"]["recall"]),
        "event_threshold": float(event_result["best_f1_posthoc"]["threshold"]),
        "event_pr_auc": float(event_result["pr_auc_envelope"]),
        "dense_cross_entropy": negative_log_likelihood / max(1, n_labels),
        "dense_windows": int(n_labels),
        "ground_truth_events": int(event_result["n_ground_truth_events"]),
    }
    return metrics, ar.strip_private(event_result)


def metric_key(metrics: dict) -> Tuple[float, float, float]:
    return (
        float(metrics["event_f1"]),
        float(metrics["event_pr_auc"]),
        -float(metrics["dense_cross_entropy"]),
    )


def restore_rng(api, rng: np.random.Generator, state: dict) -> None:
    rng.bit_generator.state = state["numpy_rng_state"]
    # last_state is loaded with map_location=DEVICE so ByteTensor RNG states
    # arrive on CUDA. PyTorch RNG setters require CPU ByteTensors even when the
    # state belongs to CUDA generators.
    api.torch.set_rng_state(state["torch_rng_state"].detach().cpu())
    if api.torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        api.torch.cuda.set_rng_state_all([
            item.detach().cpu() for item in state["cuda_rng_state_all"]
        ])
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])


def rng_payload(api, rng: np.random.Generator) -> dict:
    return {
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": api.torch.get_rng_state(),
        "cuda_rng_state_all": (
            api.torch.cuda.get_rng_state_all() if api.torch.cuda.is_available() else None
        ),
        "python_rng_state": random.getstate(),
    }


def read_completed(api, layer_dir: Path, fingerprint: str, layer: int) -> Optional[dict]:
    path = layer_dir / "completed.json"
    if not path.is_file():
        return None
    payload = cc.read_json(path)
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError(f"Completed layer has a different config: {path}")
    if int(payload.get("layer", -1)) != int(layer):
        raise RuntimeError(f"Completed record layer mismatch: {path}")
    checkpoint = Path(payload["best_checkpoint"])
    validate_selected_checkpoint(
        api.torch, checkpoint, api.DEVICE, fingerprint, layer
    )
    return payload


def train_one_layer(api, args: argparse.Namespace, layout: dict, config: dict,
                    cache_root: Path, run_dir: Path, spec: ar.ModelSpec,
                    hidden_dim: int) -> dict:
    torch = api.torch
    layer = spec.layer
    layer_dir = run_dir / f"L{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    completed = read_completed(api, layer_dir, config["fingerprint"], layer)
    if completed is not None:
        print(f"[L{layer}] completed -> reuse {Path(completed['best_checkpoint']).name}")
        return completed

    api.set_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    head = load_warm_head(api, spec, hidden_dim)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = bool(torch.cuda.is_available() and str(api.DEVICE).startswith("cuda"))
    scaler = create_grad_scaler(torch, amp_enabled)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    data = prepare_training_data(
        cache_root, layer, layout["train_files"], args.verify_cache_sha,
        args.hard_negative_ms,
    )
    print(
        f"[L{layer}] train words={len(data['positive_segments'])} "
        f"hard-pool={len(data['hard_pool'])} far-pool={len(data['far_pool'])} "
        f"skipped-events={data['skipped_events']}"
    )

    last_path = layer_dir / "last_state.pt"
    history_path = layer_dir / "history.json"
    history = cc.read_json(history_path).get("epochs", []) if history_path.is_file() else []
    start_epoch = 1
    best_metrics = None
    best_epoch = None
    best_path = None
    stale_epochs = 0
    resume_stop_reason = None
    if last_path.is_file():
        state = torch_load_checkpoint(torch, last_path, api.DEVICE)
        if state.get("config_fingerprint") != config["fingerprint"]:
            raise RuntimeError(f"Resume state config mismatch: {last_path}")
        if int(state.get("layer", -1)) != layer:
            raise RuntimeError(f"Resume state layer mismatch: {last_path}")
        head.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state.get("scaler_state_dict", {}))
        restore_rng(api, rng, state)
        start_epoch = int(state["epoch"]) + 1
        best_metrics = state.get("best_metrics")
        best_epoch = state.get("best_epoch")
        best_path = Path(state["best_checkpoint"]) if state.get("best_checkpoint") else None
        stale_epochs = int(state.get("stale_epochs", 0))
        best_fields_present = (best_metrics is not None, best_epoch is not None, best_path is not None)
        if len(set(best_fields_present)) != 1:
            raise RuntimeError(f"Resume state has inconsistent best-checkpoint fields: {last_path}")
        if best_path is not None:
            validate_selected_checkpoint(
                torch, best_path, api.DEVICE, config["fingerprint"], layer
            )
        history = [
            item for item in history if int(item.get("epoch", -1)) <= int(state["epoch"])
        ]
        if not history or int(history[-1].get("epoch", -1)) != int(state["epoch"]):
            raise RuntimeError(
                f"Resume history does not contain committed epoch {state['epoch']}: "
                f"{history_path}"
            )
        if int(state["epoch"]) >= layout["max_epochs"]:
            resume_stop_reason = "max_epochs"
        elif (
            int(state["epoch"]) >= layout["min_epochs"]
            and stale_epochs >= layout["patience"]
        ):
            resume_stop_reason = "early_stopping"
        print(f"[L{layer}] resume after epoch {start_epoch - 1}")
    elif history:
        # A process may die after history.json but before the atomic resume state.
        # Such an epoch was not committed and must be repeated from the warm start.
        print(f"[L{layer}] ignoring uncommitted history without last_state.pt")
        history = []

    stop_reason = resume_stop_reason or "max_epochs"
    epochs_to_run = (
        range(start_epoch, layout["max_epochs"] + 1)
        if resume_stop_reason is None else ()
    )
    for epoch in epochs_to_run:
        references, sample_stats = sample_epoch_references(
            data, rng, args.positives_per_word, args.hard_negative_fraction
        )
        train_stats = train_epoch(
            api, head, optimizer, scaler, criterion, data["hidden"], references,
            args.batch_size, args.grad_clip, amp_enabled,
            layout["max_steps_per_epoch"],
        )
        val_metrics, val_curve = evaluate_validation(
            api, head, cache_root, layer, layout["val_files"], args,
            layout["max_val_windows"],
        )
        improved = best_metrics is None or metric_key(val_metrics) > metric_key(best_metrics)
        if improved:
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            best_path = layer_dir / (
                f"L{layer}_best_epoch_{epoch:03d}_f1_{val_metrics['event_f1']:.4f}_"
                f"auc_{val_metrics['event_pr_auc']:.4f}_{stamp}.pth"
            )
            atomic_torch_save(torch, best_path, {
                "kind": "continuous_word_head_best",
                "script_version": SCRIPT_VERSION,
                "config_fingerprint": config["fingerprint"],
                "layer": layer,
                "epoch": epoch,
                "seed": args.seed,
                "source_seed": args.source_seed,
                "metrics": val_metrics,
                "warm_start_checkpoint": str(spec.classifier_path),
                "model_state_dict": head.state_dict(),
            })
        else:
            stale_epochs += 1

        record = {
            "epoch": epoch,
            "train": train_stats,
            "sampling": sample_stats,
            "validation": val_metrics,
            "validation_event_curve": val_curve,
            "improved": improved,
            "best_epoch_after_epoch": best_epoch,
            "best_checkpoint_after_epoch": str(best_path) if best_path else None,
            "stale_epochs": stale_epochs,
        }
        history.append(record)
        cc.atomic_json(history_path, {
            "config_fingerprint": config["fingerprint"],
            "layer": layer,
            "epochs": history,
        })
        last_state = {
            "kind": "continuous_word_head_resume",
            "script_version": SCRIPT_VERSION,
            "config_fingerprint": config["fingerprint"],
            "layer": layer,
            "epoch": epoch,
            "seed": args.seed,
            "source_seed": args.source_seed,
            "model_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metrics": best_metrics,
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_path) if best_path else None,
            "stale_epochs": stale_epochs,
            **rng_payload(api, rng),
        }
        atomic_torch_save(torch, last_path, last_state)
        print(
            f"[L{layer} epoch {epoch:02d}] trainCE={train_stats['loss']:.4f} "
            f"valF1={val_metrics['event_f1']:.4f} "
            f"valAUC={val_metrics['event_pr_auc']:.4f} "
            f"valCE={val_metrics['dense_cross_entropy']:.4f} "
            f"{'BEST' if improved else f'wait={stale_epochs}'}",
            flush=True,
        )
        if epoch >= layout["min_epochs"] and stale_epochs >= layout["patience"]:
            stop_reason = "early_stopping"
            break

    if best_path is None or best_metrics is None or best_epoch is None:
        raise RuntimeError(f"L{layer} finished without a validation-selected checkpoint")
    completed = {
        "kind": "continuous_word_head_completed",
        "script_version": SCRIPT_VERSION,
        "config_fingerprint": config["fingerprint"],
        "layer": layer,
        "seed": args.seed,
        "source_seed": args.source_seed,
        "stop_reason": stop_reason,
        "last_completed_epoch": int(history[-1]["epoch"]),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "best_checkpoint": str(best_path),
        "warm_start_checkpoint": str(spec.classifier_path),
        "history": str(history_path),
        "test_data_opened": False,
    }
    cc.atomic_json(layer_dir / "completed.json", completed)
    print(f"[L{layer}] fixed at epoch {best_epoch}: {best_path.name}")
    del head, optimizer, scaler, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return completed


def load_selected_head(api, completed: dict, hidden_dim: int):
    checkpoint = validate_selected_checkpoint(
        api.torch, Path(completed["best_checkpoint"]), api.DEVICE,
        completed["config_fingerprint"], int(completed["layer"]),
    )
    head = api.mcls.Mel2WordHidden(hidden_dim, len(api.WORDS_REMAP)).to(api.DEVICE)
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    head.eval()
    head.requires_grad_(False)
    return head


def infer_split_profiles(api, args: argparse.Namespace, cache_root: Path, run_dir: Path,
                         split_name: str, file_indices: Sequence[int], heads: dict) -> Tuple[list, list]:
    """Open and infer one split. Called for test only after the completion gate."""
    profiles = []
    timeline_paths = []
    timeline_dir = run_dir / "timelines"
    for file_index in file_indices:
        endpoints_reference = None
        metadata_reference = None
        probabilities = {}
        for layer in PRODUCTION_LAYERS:
            cached = cc.load_cached_file(
                cache_root, layer, file_index, verify_sha=args.verify_cache_sha
            )
            hidden = cc.open_hidden(cached)
            endpoints, values = cc.infer_dense(
                api, heads[layer], hidden, cached.first_endpoint,
                args.dense_batch_size, step_frames=1,
            )
            if endpoints_reference is None:
                endpoints_reference = endpoints
                metadata_reference = cc.read_json(cached.metadata_path)
            elif not np.array_equal(endpoints_reference, endpoints):
                raise RuntimeError(f"Layer endpoint mismatch in {split_name} file {file_index}")
            probabilities[f"L{layer}"] = values
            del hidden
        probabilities[ENSEMBLE_NAME] = np.mean(
            np.stack([probabilities[f"L{layer}"] for layer in PRODUCTION_LAYERS], axis=0),
            axis=0,
        ).astype(np.float32)
        assert endpoints_reference is not None and metadata_reference is not None
        profile = cc.file_profile(
            api, metadata_reference, endpoints_reference, probabilities,
            smooth_ms=args.smooth_ms, smoothing=args.smoothing,
        )
        profiles.append(profile)
        timeline_path = timeline_dir / f"{split_name}_file_{file_index:02d}.npz"
        timeline_payload = {
            "endpoints": endpoints_reference.astype(np.int64),
            "times_s": profile.times_s.astype(np.float32),
            "gt_start_s": np.asarray([item.start_s for item in profile.ground_truth], np.float32),
            "gt_end_s": np.asarray([item.end_s for item in profile.ground_truth], np.float32),
            "gt_class": np.asarray([item.class_index for item in profile.ground_truth], np.int16),
        }
        for system, values in probabilities.items():
            timeline_payload[f"prob_{system}"] = values.astype(np.float32, copy=False)
            timeline_payload[f"smooth_{system}"] = profile.smoothed[system].astype(
                np.float32, copy=False
            )
        cc.atomic_npz(timeline_path, **timeline_payload)
        timeline_paths.append(str(timeline_path))
        print(f"[{split_name} file {file_index:02d}] dense L3/L4/L5 + fixed ensemble")
    return profiles, timeline_paths


def evaluate_final_systems(api, args: argparse.Namespace, val_profiles: list,
                           test_profiles: list) -> Tuple[dict, dict]:
    systems = ["L3", "L4", "L5", ENSEMBLE_NAME]
    results = {"val": {}, "test": {}}
    operating_points = {}
    private = {"val": {}, "test": {}}
    for split_name, profiles in (("val", val_profiles), ("test", test_profiles)):
        for system in systems:
            permutations = (
                args.null_permutations
                if split_name == "test" and system == ENSEMBLE_NAME else 0
            )
            null_seed = args.seed * 1_000_003 + sum(
                (index + 1) * ord(char) for index, char in enumerate(system + split_name)
            )
            curve = ar.event_pr_curve(
                api, profiles, system, args.threshold_points,
                null_permutations=permutations, null_seed=null_seed,
            )
            private[split_name][system] = curve
            results[split_name][system] = ar.strip_private(curve)
    for system in systems:
        threshold = float(private["val"][system]["best_f1_posthoc"]["threshold"])
        point = ar.score_at_threshold(
            private["test"][system]["_candidates"], test_profiles, threshold
        )
        point["selected_on"] = "validation"
        operating_points[system] = point
        results["test"][system]["operating_point_from_val"] = point
        print(
            f"[test {system} @ val theta={threshold:.3f}] "
            f"F1={point['f1']:.3f} P={point['precision']:.3f} "
            f"R={point['recall']:.3f} FP/min={point['false_events_per_min']:.2f}"
        )
    return results, operating_points


def plot_final_pr(results: dict, operating_points: dict, run_dir: Path,
                  source_seed: int, head_seed: int) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    output = run_dir / "continuous_test_pr.png"
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for system in ("L3", "L4", "L5", ENSEMBLE_NAME):
        curve = results["test"][system]["curve"]
        ax.plot(
            curve["recall"], curve["precision"], label=system,
            linewidth=2.8 if system == ENSEMBLE_NAME else 1.4,
        )
        point = operating_points[system]
        ax.scatter([point["recall"]], [point["precision"]], s=42, zorder=5)
    chance = results["test"][ENSEMBLE_NAME].get("chance_label_null")
    if chance:
        ax.plot(
            chance["recall_mean"], chance["precision_mean"], "k--", linewidth=1.5,
            label=f"label-null ({chance['permutations']}x)",
        )
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Event recall", ylabel="Event precision",
        title="Continuous-trained Whisper decoder | "
              f"upstream seed {source_seed}, head seed {head_seed}",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def final_evaluation(api, args: argparse.Namespace, layout: dict, config: dict,
                     cache_root: Path, run_dir: Path, completions: dict,
                     hidden_dim: int) -> Path:
    marker_path = run_dir / "final_complete.json"
    if marker_path.is_file():
        marker = cc.read_json(marker_path)
        if marker.get("config_fingerprint") != config["fingerprint"]:
            raise RuntimeError("Existing final marker has a different configuration")
        result_path = Path(marker["result"])
        if not result_path.is_file():
            raise FileNotFoundError(f"Final result referenced by marker is missing: {result_path}")
        print(f"[final] already complete -> {result_path}")
        return result_path

    if tuple(sorted(completions)) != PRODUCTION_LAYERS:
        raise RuntimeError("TEST GATE CLOSED: L3, L4 and L5 must all be completed")
    for layer in PRODUCTION_LAYERS:
        if not (run_dir / f"L{layer}" / "completed.json").is_file():
            raise RuntimeError(f"TEST GATE CLOSED: L{layer} completion record is absent")
    print("[test gate] L3/L4/L5 are fixed; test caches may now be opened", flush=True)

    heads = {
        layer: load_selected_head(api, completions[layer], hidden_dim)
        for layer in PRODUCTION_LAYERS
    }
    val_profiles, val_timelines = infer_split_profiles(
        api, args, cache_root, run_dir, "val", layout["val_files"], heads
    )
    # This is the first code path in the program that receives test indices.
    test_profiles, test_timelines = infer_split_profiles(
        api, args, cache_root, run_dir, "test", layout["test_files"], heads
    )
    results, operating_points = evaluate_final_systems(
        api, args, val_profiles, test_profiles
    )
    try:
        plot_path = plot_final_pr(
            results, operating_points, run_dir, args.source_seed, args.seed
        )
    except Exception as exc:
        # A PNG is a convenience artifact. Never discard already computed,
        # validation-fixed test metrics because optional rendering failed.
        print(f"[plot warning] PR plot was not saved: {exc}", flush=True)
        plot_path = None
    created = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hotfix_records = []
    hotfix_dir = run_dir / "runtime_hotfixes"
    if hotfix_dir.is_dir():
        hotfix_records = [
            cc.read_json(path) for path in sorted(hotfix_dir.glob("*.json"))
        ]
    result = {
        "kind": "continuous_trained_word_decoder_evaluation",
        "script_version": SCRIPT_VERSION,
        "created": created,
        "config_fingerprint": config["fingerprint"],
        "run_config": str(run_dir / "run_config.json"),
        "patient": cc.PATIENT,
        "seed": args.seed,
        "seed_role": "continuous_head_training_seed",
        "source_seed": args.source_seed,
        "source_seed_role": "frozen_encoder_and_synchronous_head_seed",
        "layers": list(PRODUCTION_LAYERS),
        "primary_system": ENSEMBLE_NAME,
        "protocol": {
            "offline_continuous_inference": True,
            "word_boundaries_used_for_inference": False,
            "word_boundaries_used_for_training_labels_and_scoring": True,
            "fixed_trailing_window_frames": cc.WINDOW_FRAMES,
            "smooth_ms": args.smooth_ms,
            "smoothing": args.smoothing,
            "test_threshold_selected_on_validation": True,
            "test_opened_only_after_all_heads_fixed": True,
        },
        "completed_heads": completions,
        "splits": results,
        "primary_test_operating_point": operating_points[ENSEMBLE_NAME],
        "timelines": {"val": val_timelines, "test": test_timelines},
        "plot": str(plot_path) if plot_path else None,
        "runtime_hotfixes": hotfix_records,
        "caveats": [
            "This is offline asynchronous decoding: preprocessing and whole-file feature scaling are not strictly causal.",
            "Centered smoothing uses future probability samples when selected.",
            "Fixed phrase order and the approximately 1.48 s effective context can encode transition or phrase-position information.",
            "Post-hoc best test F1 is descriptive; the validation-selected operating point is primary.",
        ],
    }
    result_path = run_dir / f"continuous_result_{created}.json"
    cc.atomic_json(result_path, result)
    cc.atomic_json(run_dir / "final_result.json", result)
    cc.atomic_json(marker_path, {
        "kind": "continuous_final_complete",
        "config_fingerprint": config["fingerprint"],
        "result": str(result_path),
        "primary_system": ENSEMBLE_NAME,
        "primary_test_operating_point": operating_points[ENSEMBLE_NAME],
    })
    return result_path


def main() -> int:
    args = parse_args()
    validate_args(args)
    api, patient, base, archive, cache_root, _splits, layout, specs = (
        load_experiment_context(args)
    )
    required_files = layout["train_files"] + layout["val_files"]
    cache_status = inspect_cache_set(
        cache_root, layout["layers"], required_files,
        specs, verify_sha=args.verify_cache_sha, strict=False,
    )
    print(f"[runtime] {api.device_str()}")
    print(f"[mode] {layout['mode']} | layers={layout['layers']}")
    print(f"[base read-only] {base}")
    print(f"[archive read-only] {archive}")
    print(f"[cache] {cache_root}")
    print(
        f"[cache train+val] complete={len(cache_status['complete'])} "
        f"missing={len(cache_status['missing'])} errors={len(cache_status['errors'])}"
    )
    print("[test gate] test cache deliberately not inspected during preflight/training setup")
    if args.preflight:
        if cache_status["missing"]:
            print("[preflight] source/archive OK; persistent cache still needs to be built/resumed")
            for item in cache_status["missing"][:5]:
                print(f"  missing {item['cache']}: {item['reason']}")
        if cache_status["errors"]:
            for item in cache_status["errors"]:
                print(f"  ERROR {item['cache']}: {item['reason']}")
            raise RuntimeError("Preflight found invalid cache artifacts")
        print("[preflight] OK; no model trained and no test data opened")
        return 0

    inspect_cache_set(
        cache_root, layout["layers"], required_files,
        specs, verify_sha=args.verify_cache_sha, strict=True,
    )
    if not str(api.DEVICE).startswith("cuda") and not args.allow_cpu:
        raise RuntimeError("CUDA is required; use --allow-cpu only for an intentional CPU run")

    config = build_run_config(args, base, archive, cache_root, layout, specs)
    run_dir = effective_run_dir(args)
    config = ensure_run_config(run_dir, config)
    first_cached = cc.load_cached_file(
        cache_root, layout["layers"][0], layout["train_files"][0],
        verify_sha=args.verify_cache_sha,
    )
    hidden_dim = first_cached.hidden_dim
    completions = {}
    for layer in layout["layers"]:
        completions[layer] = train_one_layer(
            api, args, layout, config, cache_root, run_dir, specs[layer], hidden_dim
        )

    if args.smoke:
        print(f"[smoke done] {run_dir}")
        print("[safety] smoke used L3 train/validation only; test data was never opened")
        return 0

    if args.train_only:
        print(f"[train-only done] all production heads are fixed in {run_dir}")
        print(
            "[test gate] stopped before test cache access; rerun the same command "
            "without --train-only for the single fixed evaluation"
        )
        return 0

    result_path = final_evaluation(
        api, args, layout, config, cache_root, run_dir, completions, hidden_dim
    )
    print(f"[done] {result_path}")
    print("[safety] base project and frozen archive stayed read-only; no checkpoint deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
