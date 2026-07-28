"""Immutable out-of-fold aggregation for a completed VocalMind production run.

The aggregator is deliberately separate from model fitting.  It validates every
fold gate before reading any held-out result or prediction array, then stitches
the five predeclared test repetitions into one 100-trial OOF surface per training
seed.  It never ranks, filters, or selects models using held-out outcomes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .data.vocalmind import DEFAULT_VOCALMIND_CONTRACT
from .integrity import fingerprint_json, sha256_bytes, sha256_file
from .protocol import SplitManifest, TestGate
from .source_identity import capture_source_identity, require_clean_frozen_source
from .vocalmind_primary import (
    PRIMARY_FOLDS,
    PRIMARY_SEEDS,
    PrimaryConfig,
    closed_set_metrics,
    planned_training_units,
    validate_primary_config,
)


SCHEMA_VERSION = 1
AGGREGATOR_VERSION = "vocalmind_oof_immutable_v1"
OOF_MODELS = ("L3", "L4", "L5", "L345", "MELx3")
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy")
OUTPUT_JSON_NAME = "vocalmind_oof_summary.json"
OUTPUT_CSV_NAME = "vocalmind_oof_metrics.csv"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRIAL_ID = re.compile(
    r"^vocalized_word:(?P<word>[A-Za-z][A-Za-z0-9]*):rep(?P<rep>[0-9]{2})$"
)


class OofIntegrityError(RuntimeError):
    """A completed-run artifact does not satisfy the frozen OOF contract."""


@dataclass(frozen=True)
class GateAudit:
    fold: int
    split: SplitManifest
    authorization: Mapping[str, str]
    validation_artifacts: tuple[Mapping[str, Any], ...]


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if not _SHA256.fullmatch(normalized):
        raise OofIntegrityError(f"{label} is not a lowercase SHA256")
    return normalized


def _require_regular_file(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise OofIntegrityError(f"expected a regular immutable artifact: {path}")
    return path


def _stable_read_bytes(path: Path) -> tuple[bytes, str]:
    path = _require_regular_file(path)
    before = sha256_file(path)
    payload = path.read_bytes()
    after = sha256_file(path)
    if before != after or sha256_bytes(payload) != before:
        raise OofIntegrityError(f"artifact changed while it was being read: {path}")
    return payload, before


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    raw, content_sha256 = _stable_read_bytes(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OofIntegrityError(f"invalid UTF-8 JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise OofIntegrityError(f"JSON artifact is not an object: {path}")
    return value, content_sha256


def _read_fingerprinted_json(
    path: Path,
    *,
    expected_kind: str,
) -> tuple[dict[str, Any], str]:
    value, content_sha256 = _read_json_object(path)
    fingerprint = value.get("fingerprint")
    body = dict(value)
    body.pop("fingerprint", None)
    if fingerprint != fingerprint_json(body):
        raise OofIntegrityError(f"JSON fingerprint mismatch: {path}")
    if value.get("kind") != expected_kind:
        raise OofIntegrityError(
            f"unexpected JSON kind at {path}: {value.get('kind')!r}"
        )
    return value, content_sha256


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise OofIntegrityError(f"artifact resolves outside run root: {path}") from exc


def _validate_source_identity(run_manifest: Mapping[str, Any]) -> None:
    identity = run_manifest.get("source_identity")
    if not isinstance(identity, Mapping):
        raise OofIntegrityError("run manifest has no source-identity object")
    body = dict(identity)
    fingerprint = body.pop("fingerprint", None)
    if fingerprint != fingerprint_json(body):
        raise OofIntegrityError("source-identity fingerprint mismatch")
    if run_manifest.get("source_identity_fingerprint") != fingerprint:
        raise OofIntegrityError("run manifest points to another source identity")
    git = identity.get("git")
    if not isinstance(git, Mapping) or not git.get("commit") or git.get("dirty") is not False:
        raise OofIntegrityError("production source identity is not a clean Git commit")


def _load_run_manifest_contract(
    run_root: Path,
) -> tuple[dict[str, Any], str, PrimaryConfig]:
    manifest_path = run_root / "run_manifest.json"
    manifest, manifest_sha = _read_fingerprinted_json(
        manifest_path, expected_kind="vocalmind_primary_run"
    )
    if manifest.get("schema_version") != 1:
        raise OofIntegrityError("unsupported production-run manifest schema")
    config_payload = manifest.get("config")
    if not isinstance(config_payload, Mapping):
        raise OofIntegrityError("run manifest has no embedded frozen config")
    try:
        config = validate_primary_config(config_payload)
    except (TypeError, ValueError) as exc:
        raise OofIntegrityError("embedded production config violates the contract") from exc
    if (
        config.run_scope != "production"
        or config.status != "frozen_confirmatory"
        or config.folds != PRIMARY_FOLDS
        or config.seeds != PRIMARY_SEEDS
    ):
        raise OofIntegrityError("OOF aggregation requires the complete frozen production plan")
    if manifest.get("config_fingerprint") != config.fingerprint:
        raise OofIntegrityError("run-manifest config fingerprint mismatch")
    _require_sha256(manifest.get("dataset_index_sha256"), "dataset index fingerprint")
    _require_sha256(manifest.get("host_preflight_sha256"), "host preflight fingerprint")
    _validate_source_identity(manifest)
    return manifest, manifest_sha, config


def _require_matching_aggregation_source(
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind post-production metrics to the exact clean training checkout/runtime."""

    current = capture_source_identity()
    try:
        require_clean_frozen_source(current)
    except RuntimeError as exc:
        raise OofIntegrityError(
            "OOF aggregation requires a clean frozen source checkout"
        ) from exc
    expected = str(run_manifest.get("source_identity_fingerprint", ""))
    if current.get("fingerprint") != expected:
        raise OofIntegrityError(
            "OOF aggregation source/runtime differs from the production run; "
            "checkout the run's exact freeze commit and locked environment"
        )
    return current


def _load_run_summary(
    run_root: Path,
    *,
    config: PrimaryConfig,
) -> tuple[dict[str, Any], str]:
    summary, summary_sha = _read_fingerprinted_json(
        run_root / "summary.json", expected_kind="vocalmind_primary_run_summary"
    )
    if summary.get("schema_version") != 1:
        raise OofIntegrityError("unsupported production-run summary schema")
    if summary.get("config_fingerprint") != config.fingerprint:
        raise OofIntegrityError("run-summary config fingerprint mismatch")
    if summary.get("test_model_selection") is not False:
        raise OofIntegrityError("run summary does not prohibit test-set model selection")
    if summary.get("threshold_policy") != "not_applicable_closed_set_argmax":
        raise OofIntegrityError("run summary threshold policy differs from the frozen plan")
    return summary, summary_sha


def _summary_fold_lookup(summary: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != len(PRIMARY_FOLDS):
        raise OofIntegrityError("run summary must contain exactly five folds")
    lookup: dict[int, Mapping[str, Any]] = {}
    for item in folds:
        if not isinstance(item, Mapping):
            raise OofIntegrityError("run-summary fold entry is not an object")
        fold = int(item.get("fold", -1))
        if fold in lookup:
            raise OofIntegrityError(f"duplicate fold in run summary: {fold}")
        lookup[fold] = item
    if tuple(sorted(lookup)) != PRIMARY_FOLDS:
        raise OofIntegrityError("run summary fold set differs from [1,2,3,4,5]")
    return lookup


def _parse_trial_partition(
    ids: Sequence[str], *, fold: int, role: str
) -> dict[int, set[str]]:
    if len(ids) != len(set(ids)):
        raise OofIntegrityError(f"fold {fold} {role} partition repeats trial IDs")
    by_repetition: dict[int, set[str]] = {}
    for trial_id in ids:
        match = _TRIAL_ID.fullmatch(str(trial_id))
        if match is None:
            raise OofIntegrityError(f"invalid VocalMind trial ID: {trial_id!r}")
        repetition = int(match.group("rep"))
        words = by_repetition.setdefault(repetition, set())
        word = match.group("word")
        if word in words:
            raise OofIntegrityError(
                f"fold {fold} {role} repeats class {word} in repetition {repetition}"
            )
        words.add(word)
    return by_repetition


def _validate_fold_split(split: SplitManifest, *, fold: int) -> None:
    validation_repetition = fold % 5 + 1
    training_repetitions = set(PRIMARY_FOLDS) - {fold, validation_repetition}
    expected_words = set(DEFAULT_VOCALMIND_CONTRACT.words)
    partitions = {
        "train": (split.train_ids, training_repetitions, 60),
        "validation": (split.validation_ids, {validation_repetition}, 20),
        "held_out_test": (split.held_out_test_ids, {fold}, 20),
    }
    for role, (ids, expected_repetitions, expected_count) in partitions.items():
        if len(ids) != expected_count:
            raise OofIntegrityError(
                f"fold {fold} {role} count is {len(ids)}, expected {expected_count}"
            )
        by_repetition = _parse_trial_partition(ids, fold=fold, role=role)
        if set(by_repetition) != expected_repetitions:
            raise OofIntegrityError(
                f"fold {fold} {role} repetitions differ from the frozen cyclic split"
            )
        if any(words != expected_words for words in by_repetition.values()):
            raise OofIntegrityError(
                f"fold {fold} {role} does not contain each fixed class per repetition"
            )


def _validate_all_gates(
    run_root: Path,
    *,
    run_manifest: Mapping[str, Any],
    config: PrimaryConfig,
) -> dict[int, GateAudit]:
    """Validate all five gates before any result/NPZ artifact is opened."""

    protocol_fingerprint = _require_sha256(
        run_manifest.get("fingerprint"), "run-manifest fingerprint"
    )
    dataset_index_sha256 = _require_sha256(
        run_manifest.get("dataset_index_sha256"), "dataset index fingerprint"
    )
    audits: dict[int, GateAudit] = {}
    all_test_ids: list[str] = []
    for fold in PRIMARY_FOLDS:
        fold_root = run_root / f"fold_{fold:02d}"
        split = SplitManifest.load(_require_regular_file(fold_root / "split_manifest.json"))
        if (
            split.dataset_id != "VocalMind-v2:primary-overt"
            or split.protocol_id != f"vocalmind-primary-reps1-5-fold-{fold:02d}"
            or split.dataset_manifest_sha256 != dataset_index_sha256
        ):
            raise OofIntegrityError(f"fold {fold} split provenance differs from the run")
        _validate_fold_split(split, fold=fold)
        all_test_ids.extend(split.held_out_test_ids)

        required_units = tuple(
            f"seed{seed}_{unit.key}"
            for seed in config.seeds
            for unit in planned_training_units(config, outer_seed=seed)
        )
        gate = TestGate(
            state_directory=fold_root / "test_gate",
            split=split,
            required_units=required_units,
            protocol_fingerprint=protocol_fingerprint,
        )
        # authorization() requires an existing, valid open receipt.  It cannot
        # create one, so this remains a read-only post-completion operation.
        try:
            authorization = gate.authorization()
        except (RuntimeError, ValueError, OSError) as exc:
            raise OofIntegrityError(f"fold {fold} held-out gate is not fully valid") from exc

        validation_artifacts: list[Mapping[str, Any]] = []
        for seed in config.seeds:
            for unit in planned_training_units(config, outer_seed=seed):
                gate_unit = f"seed{seed}_{unit.key}"
                completion_path = (
                    fold_root / "test_gate" / "completed" / f"{gate_unit}.json"
                )
                completion, completion_sha = _read_fingerprinted_json(
                    completion_path,
                    expected_kind="validation_fixed_training_completion",
                )
                fixed_path = fold_root / f"seed_{seed}" / unit.key / "validation_fixed.json"
                fixed, fixed_sha = _read_fingerprinted_json(
                    fixed_path,
                    expected_kind="vocalmind_validation_fixed_unit",
                )
                expected_run_fingerprint = fingerprint_json(
                    {
                        "config_fingerprint": config.fingerprint,
                        "split_fingerprint": split.fingerprint,
                        "outer_seed": seed,
                        "initialization_seed": unit.initialization_seed,
                        "training_unit": unit.key,
                        "target_representation": unit.target_representation,
                    }
                )
                expected_fixed = {
                    "fold": fold,
                    "outer_seed": seed,
                    "initialization_seed": unit.initialization_seed,
                    "training_unit": unit.key,
                    "target_representation": unit.target_representation,
                    "config_fingerprint": config.fingerprint,
                    "split_fingerprint": split.fingerprint,
                    "model_selection": "validation_loss_only",
                    "threshold_selection": "not_applicable_closed_set_argmax",
                    "test_data_opened": False,
                }
                if any(fixed.get(key) != value for key, value in expected_fixed.items()):
                    raise OofIntegrityError(
                        f"validation-fixed receipt differs for fold {fold} {gate_unit}"
                    )
                for field in (
                    "target_reducer_sha256",
                    "regression_checkpoint_sha256",
                    "classifier_checkpoint_sha256",
                    "validation_prediction_fingerprint",
                ):
                    _require_sha256(
                        fixed.get(field), f"fold {fold} {gate_unit} {field}"
                    )
                if (
                    completion.get("unit") != gate_unit
                    or completion.get("artifact_sha256") != fixed_sha
                    or completion.get("run_fingerprint") != expected_run_fingerprint
                    or completion.get("split_fingerprint") != split.fingerprint
                    or completion.get("protocol_fingerprint") != protocol_fingerprint
                    or completion.get("test_data_opened") is not False
                ):
                    raise OofIntegrityError(
                        f"gate completion does not bind the fixed artifact: {gate_unit}"
                    )
                validation_artifacts.append(
                    {
                        "unit": gate_unit,
                        "validation_fixed_path": _relative(fixed_path, run_root),
                        "validation_fixed_sha256": fixed_sha,
                        "validation_fixed_fingerprint": fixed["fingerprint"],
                        "completion_path": _relative(completion_path, run_root),
                        "completion_sha256": completion_sha,
                        "completion_fingerprint": completion["fingerprint"],
                    }
                )
        audits[fold] = GateAudit(
            fold=fold,
            split=split,
            authorization=asdict(authorization),
            validation_artifacts=tuple(validation_artifacts),
        )
    if len(all_test_ids) != 100 or len(set(all_test_ids)) != 100:
        raise OofIntegrityError("five folds do not form exactly 100 unique OOF trial IDs")
    return audits


def _probability_matrix(value: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.dtype != np.float32 or matrix.shape != (20, 20):
        raise OofIntegrityError(f"{label} must be float32 with shape (20,20)")
    if (
        not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise OofIntegrityError(f"{label} is not a valid softmax matrix")
    return np.array(matrix, copy=True)


def _validate_prediction_receipt(
    receipt: Mapping[str, Any],
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    trial_ids: Sequence[str],
    initialization_seed: int,
    authorization: Mapping[str, str],
    label: str,
) -> None:
    body = dict(receipt)
    fingerprint = body.pop("fingerprint", None)
    if fingerprint != fingerprint_json(body):
        raise OofIntegrityError(f"prediction receipt fingerprint mismatch: {label}")
    ids_sha = fingerprint_json(list(trial_ids))
    evaluation_gate_receipt = {
        key: authorization[key]
        for key in (
            "split_fingerprint",
            "protocol_fingerprint",
            "open_receipt_fingerprint",
        )
    }
    required = {
        "schema_version": 1,
        "kind": "deterministic_ordered_prediction_receipt",
        "task": "classification",
        "split_role": "held_out_test",
        "evaluation_seed": initialization_seed,
        "sample_count": 20,
        "sample_ids_sha256": ids_sha,
        "predictions_sha256": _array_fingerprint(probabilities),
        "targets_sha256": _array_fingerprint(labels),
        "ordering": "dataset_order_no_shuffle",
        "test_gate_authorization": evaluation_gate_receipt,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise OofIntegrityError(f"prediction receipt provenance mismatch: {label}")
    _require_sha256(receipt.get("model_state_sha256"), f"{label} model fingerprint")
    _require_sha256(
        receipt.get("training_config_fingerprint"),
        f"{label} training-config fingerprint",
    )
    metrics = receipt.get("metrics")
    if not isinstance(metrics, Mapping):
        raise OofIntegrityError(f"prediction receipt metrics missing: {label}")
    expected_accuracy = float(np.mean(np.argmax(probabilities, axis=1) == labels))
    chosen = probabilities[np.arange(len(labels)), labels]
    expected_ce = float(-np.mean(np.log(np.maximum(chosen, 1e-300))))
    if not math.isclose(float(metrics.get("accuracy", math.nan)), expected_accuracy, abs_tol=1e-12):
        raise OofIntegrityError(f"prediction receipt accuracy differs: {label}")
    if not math.isclose(float(metrics.get("cross_entropy", math.nan)), expected_ce, abs_tol=1e-12):
        raise OofIntegrityError(f"prediction receipt cross-entropy differs: {label}")
    _require_sha256(metrics.get("logits_sha256"), f"{label} logits fingerprint")


def _read_prediction_npz(
    path: Path,
    *,
    expected_keys: set[str],
) -> tuple[dict[str, np.ndarray], str]:
    path = _require_regular_file(path)
    before = sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise OofIntegrityError(
                    f"prediction NPZ keys differ at {path}; got {sorted(archive.files)}"
                )
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise OofIntegrityError(f"cannot read prediction NPZ: {path}") from exc
    after = sha256_file(path)
    if before != after:
        raise OofIntegrityError(f"prediction NPZ changed while being read: {path}")
    return arrays, before


def _summary_seed_lookup(
    fold_summary: Mapping[str, Any], *, fold: int
) -> dict[int, Mapping[str, Any]]:
    if (
        fold_summary.get("test_gate_open") is not True
        or int(fold_summary.get("test_repetition", -1)) != fold
        or int(fold_summary.get("validation_repetition", -1)) != fold % 5 + 1
    ):
        raise OofIntegrityError(f"fold {fold} summary is not a completed test fold")
    entries = fold_summary.get("seeds")
    if not isinstance(entries, list) or len(entries) != len(PRIMARY_SEEDS):
        raise OofIntegrityError(f"fold {fold} summary must contain five seeds")
    lookup: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise OofIntegrityError(f"fold {fold} has a non-object seed summary")
        seed = int(entry.get("seed", -1))
        if seed in lookup:
            raise OofIntegrityError(f"fold {fold} repeats seed {seed}")
        lookup[seed] = entry
    if tuple(sorted(lookup)) != tuple(sorted(PRIMARY_SEEDS)):
        raise OofIntegrityError(f"fold {fold} seed set differs from the frozen plan")
    return lookup


def _load_all_predictions(
    run_root: Path,
    *,
    config: PrimaryConfig,
    summary: Mapping[str, Any],
    gates: Mapping[int, GateAudit],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Read held-out artifacts only after ``_validate_all_gates`` has succeeded."""

    class_order = tuple(DEFAULT_VOCALMIND_CONTRACT.words)
    summary_folds = _summary_fold_lookup(summary)
    aggregate: dict[int, dict[str, Any]] = {
        seed: {"trial_ids": [], "labels": [], **{model: [] for model in OOF_MODELS}}
        for seed in config.seeds
    }
    provenance: list[dict[str, Any]] = []
    reference_ids_by_fold: dict[int, tuple[str, ...]] = {}
    for fold in PRIMARY_FOLDS:
        gate = gates[fold]
        fold_summary = summary_folds[fold]
        if fold_summary.get("split_fingerprint") != gate.split.fingerprint:
            raise OofIntegrityError(f"fold {fold} summary points to another split")
        seed_summary = _summary_seed_lookup(fold_summary, fold=fold)
        expected_raw_units_by_seed = {
            seed: tuple(unit.key for unit in planned_training_units(config, outer_seed=seed))
            for seed in config.seeds
        }
        for seed in config.seeds:
            seed_root = run_root / f"fold_{fold:02d}" / f"seed_{seed}"
            result_path = seed_root / "result.json"
            result, result_sha = _read_fingerprinted_json(
                result_path, expected_kind="vocalmind_primary_seed_result"
            )
            expected_units = expected_raw_units_by_seed[seed]
            npz_path = seed_root / "held_out_test_predictions.npz"
            npz_keys = {"trial_ids", "labels", *expected_units, "L345", "MEL_mean"}
            arrays, npz_sha = _read_prediction_npz(npz_path, expected_keys=npz_keys)
            if result.get("prediction_npz_sha256") != npz_sha:
                raise OofIntegrityError(f"prediction NPZ hash mismatch: fold {fold} seed {seed}")

            trial_array = arrays["trial_ids"]
            label_array = arrays["labels"]
            if trial_array.shape != (20,) or trial_array.dtype.kind != "U":
                raise OofIntegrityError("trial_ids must be a 20-element Unicode array")
            if label_array.dtype != np.int64 or label_array.shape != (20,):
                raise OofIntegrityError("labels must be int64 with shape (20,)")
            trial_ids = tuple(str(value) for value in trial_array.tolist())
            labels = np.asarray(label_array, dtype=np.int64)
            if trial_ids != gate.split.held_out_test_ids:
                raise OofIntegrityError(f"NPZ trial order differs from fold {fold} split")
            if tuple(result.get("test_trial_ids", ())) != trial_ids:
                raise OofIntegrityError(f"result trial order differs: fold {fold} seed {seed}")
            if tuple(result.get("class_order", ())) != class_order:
                raise OofIntegrityError(f"fixed class order differs: fold {fold} seed {seed}")
            expected_labels = np.asarray(
                [class_order.index(_TRIAL_ID.fullmatch(item).group("word")) for item in trial_ids],
                dtype=np.int64,
            )
            if not np.array_equal(labels, expected_labels):
                raise OofIntegrityError(f"labels differ from fixed trial classes: fold {fold}")
            if fold not in reference_ids_by_fold:
                reference_ids_by_fold[fold] = trial_ids
            elif reference_ids_by_fold[fold] != trial_ids:
                raise OofIntegrityError(f"seed-specific held-out order differs in fold {fold}")

            if (
                result.get("schema_version") != 1
                or int(result.get("fold", -1)) != fold
                or int(result.get("seed", -1)) != seed
                or result.get("selection")
                != {
                    "model": "validation_loss_only",
                    "threshold": "not_applicable_closed_set_argmax",
                    "test_used_for_selection": False,
                }
                or result.get("ensemble")
                != {
                    "layers": [3, 4, 5],
                    "rule": "arithmetic_mean_of_probabilities",
                    "subset_search": False,
                }
                or result.get("primary_contrast") != "L3+L4+L5_vs_MELx3"
                or result.get("secondary_contrasts")
                != ["L3+L4+L5_vs_single_MEL", "L3+L4+L5_vs_L4"]
            ):
                raise OofIntegrityError(f"fixed analysis contract differs: fold {fold} seed {seed}")
            mel_contract = result.get("mel_compute_matched_control")
            mel_units = tuple(unit for unit in expected_units if unit.startswith("mel"))
            if not isinstance(mel_contract, Mapping) or (
                mel_contract.get("mode") != "required_three_initialization_probability_mean"
                or tuple(mel_contract.get("training_units", ())) != mel_units
                or tuple(mel_contract.get("initialization_seeds", ()))
                != tuple(seed + offset for offset in (0, 1000, 2000))
                or mel_contract.get("rule") != "arithmetic_mean_of_softmax_probabilities"
                or mel_contract.get("subset_search") is not False
            ):
                raise OofIntegrityError(f"MELx3 contract differs: fold {fold} seed {seed}")

            raw_probabilities = {
                unit: _probability_matrix(
                    arrays[unit], label=f"fold {fold} seed {seed} {unit}"
                )
                for unit in expected_units
            }
            l345_expected = np.mean(
                np.stack([raw_probabilities["L3"], raw_probabilities["L4"], raw_probabilities["L5"]]),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            mel_expected = np.mean(
                np.stack([raw_probabilities[unit] for unit in mel_units]),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            l345 = _probability_matrix(
                arrays["L345"], label=f"fold {fold} seed {seed} L345"
            )
            mel_x3 = _probability_matrix(
                arrays["MEL_mean"], label=f"fold {fold} seed {seed} MELx3"
            )
            if not np.array_equal(l345, l345_expected):
                raise OofIntegrityError(f"stored L345 is not the fixed probability mean")
            if not np.array_equal(mel_x3, mel_expected):
                raise OofIntegrityError(f"stored MELx3 is not the fixed probability mean")

            receipts = result.get("prediction_receipts")
            if not isinstance(receipts, Mapping) or set(receipts) != set(expected_units):
                raise OofIntegrityError(f"prediction receipt set differs: fold {fold} seed {seed}")
            planned = {unit.key: unit for unit in planned_training_units(config, outer_seed=seed)}
            for unit in expected_units:
                receipt = receipts[unit]
                if not isinstance(receipt, Mapping):
                    raise OofIntegrityError(f"prediction receipt is not an object: {unit}")
                _validate_prediction_receipt(
                    receipt,
                    probabilities=raw_probabilities[unit],
                    labels=labels,
                    trial_ids=trial_ids,
                    initialization_seed=planned[unit].initialization_seed,
                    authorization=gate.authorization,
                    label=f"fold {fold} seed {seed} {unit}",
                )

            system_probabilities = {
                "L3": raw_probabilities["L3"],
                "L4": raw_probabilities["L4"],
                "L5": raw_probabilities["L5"],
                "L345": l345,
                "MELx3": mel_x3,
            }
            expected_metrics = {
                unit: closed_set_metrics(raw_probabilities[unit], labels, class_order=class_order)
                for unit in expected_units
            }
            expected_metrics["L3+L4+L5"] = closed_set_metrics(
                l345, labels, class_order=class_order
            )
            expected_metrics["MELx3"] = closed_set_metrics(
                mel_x3, labels, class_order=class_order
            )
            if result.get("metrics") != expected_metrics:
                raise OofIntegrityError(f"stored metrics do not match predictions: fold {fold} seed {seed}")
            summarized = seed_summary[seed]
            if (
                summarized.get("result_fingerprint") != result.get("fingerprint")
                or summarized.get("metrics") != expected_metrics
            ):
                raise OofIntegrityError(f"run summary does not bind result: fold {fold} seed {seed}")

            aggregate[seed]["trial_ids"].extend(trial_ids)
            aggregate[seed]["labels"].append(labels)
            for model, values in system_probabilities.items():
                aggregate[seed][model].append(values)
            provenance.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "result_path": _relative(result_path, run_root),
                    "result_sha256": result_sha,
                    "result_fingerprint": result["fingerprint"],
                    "prediction_path": _relative(npz_path, run_root),
                    "prediction_sha256": npz_sha,
                }
            )
    return aggregate, provenance


def _descriptive_statistics(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(PRIMARY_SEEDS),) or not np.isfinite(array).all():
        raise OofIntegrityError("descriptive seed statistics require five finite values")
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    sem = float(sd / math.sqrt(len(array)))
    critical = float(student_t.ppf(0.975, df=len(array) - 1))
    margin = critical * sem
    return {
        "n_training_seeds": int(len(array)),
        "mean": mean,
        "sample_sd": sd,
        "sem": sem,
        "t_critical_0.975_df4": critical,
        "ci95_low": float(mean - margin),
        "ci95_high": float(mean + margin),
        "interpretation": (
            "descriptive interval across five training seeds conditional on one "
            "participant; not a population or biological-sample confidence interval"
        ),
    }


def _csv_bytes(
    *,
    run_fingerprint: str,
    per_seed: Sequence[Mapping[str, Any]],
    descriptive: Mapping[str, Mapping[str, Mapping[str, Any]]],
    primary_contrast: Mapping[str, Mapping[str, Any]],
) -> bytes:
    stream = io.StringIO(newline="")
    fieldnames = (
        "row_type",
        "run_fingerprint",
        "seed",
        "model",
        "metric",
        "value",
        "mean",
        "sample_sd",
        "sem",
        "ci95_low",
        "ci95_high",
        "n_training_seeds",
        "biological_n",
        "inference_scope",
    )
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item in per_seed:
        seed = int(item["seed"])
        for model in OOF_MODELS:
            metrics = item["metrics"][model]
            for metric in METRIC_NAMES:
                writer.writerow(
                    {
                        "row_type": "training_seed",
                        "run_fingerprint": run_fingerprint,
                        "seed": seed,
                        "model": model,
                        "metric": metric,
                        "value": format(float(metrics[metric]), ".17g"),
                        "biological_n": 1,
                        "inference_scope": "descriptive_training_seed_conditional_on_one_participant",
                    }
                )
        for metric in METRIC_NAMES:
            writer.writerow(
                {
                    "row_type": "training_seed_primary_contrast",
                    "run_fingerprint": run_fingerprint,
                    "seed": seed,
                    "model": "L345_minus_MELx3",
                    "metric": metric,
                    "value": format(
                        float(item["primary_contrast_L345_minus_MELx3"][metric]),
                        ".17g",
                    ),
                    "biological_n": 1,
                    "inference_scope": "descriptive_training_seed_conditional_on_one_participant",
                }
            )
    for model in OOF_MODELS:
        for metric in METRIC_NAMES:
            stats = descriptive[model][metric]
            writer.writerow(
                {
                    "row_type": "descriptive_seed_summary",
                    "run_fingerprint": run_fingerprint,
                    "model": model,
                    "metric": metric,
                    "mean": format(float(stats["mean"]), ".17g"),
                    "sample_sd": format(float(stats["sample_sd"]), ".17g"),
                    "sem": format(float(stats["sem"]), ".17g"),
                    "ci95_low": format(float(stats["ci95_low"]), ".17g"),
                    "ci95_high": format(float(stats["ci95_high"]), ".17g"),
                    "n_training_seeds": int(stats["n_training_seeds"]),
                    "biological_n": 1,
                    "inference_scope": "descriptive_training_seed_conditional_on_one_participant",
                }
            )
    for metric in METRIC_NAMES:
        stats = primary_contrast[metric]
        writer.writerow(
            {
                "row_type": "descriptive_seed_primary_contrast",
                "run_fingerprint": run_fingerprint,
                "model": "L345_minus_MELx3",
                "metric": metric,
                "mean": format(float(stats["mean"]), ".17g"),
                "sample_sd": format(float(stats["sample_sd"]), ".17g"),
                "sem": format(float(stats["sem"]), ".17g"),
                "ci95_low": format(float(stats["ci95_low"]), ".17g"),
                "ci95_high": format(float(stats["ci95_high"]), ".17g"),
                "n_training_seeds": int(stats["n_training_seeds"]),
                "biological_n": 1,
                "inference_scope": "descriptive_training_seed_conditional_on_one_participant",
            }
        )
    return stream.getvalue().encode("utf-8")


def _render_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable_pair(
    output_directory: Path,
    *,
    json_bytes: bytes,
    csv_bytes: bytes,
) -> bool:
    output_directory = Path(output_directory)
    expected = {
        OUTPUT_JSON_NAME: json_bytes,
        OUTPUT_CSV_NAME: csv_bytes,
    }
    if output_directory.exists():
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise OofIntegrityError(f"output path is not a regular directory: {output_directory}")
        actual_names = {path.name for path in output_directory.iterdir()}
        if actual_names != set(expected):
            raise OofIntegrityError(
                f"immutable output directory has unexpected files: {sorted(actual_names)}"
            )
        for name, content in expected.items():
            existing, _ = _stable_read_bytes(output_directory / name)
            if existing != content:
                raise OofIntegrityError(f"existing immutable OOF output differs: {name}")
        return True

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_directory.with_name(
        f".{output_directory.name}.partial-{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        for name, content in expected.items():
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        if output_directory.exists():
            raise FileExistsError(f"immutable output appeared concurrently: {output_directory}")
        os.replace(temporary, output_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return False


def aggregate_vocalmind_oof(
    run_root: Path | str,
    output_directory: Path | str,
) -> dict[str, Any]:
    """Validate and aggregate one complete five-fold/five-seed production run."""

    run_root = Path(run_root).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if not run_root.is_dir() or run_root.is_symlink():
        raise OofIntegrityError(f"production run root is not a regular directory: {run_root}")
    if output_directory == run_root:
        raise OofIntegrityError("output directory cannot equal the production run root")
    source_root = Path(__file__).resolve().parents[2]
    try:
        output_directory.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise OofIntegrityError("OOF output must stay outside the Git source tree")

    manifest, manifest_sha, config = _load_run_manifest_contract(run_root)
    aggregation_source_identity = _require_matching_aggregation_source(manifest)
    # Global gate validation intentionally precedes the first summary, result,
    # or NPZ read that contains held-out outcomes.
    gate_audits = _validate_all_gates(
        run_root,
        run_manifest=manifest,
        config=config,
    )
    summary, summary_sha = _load_run_summary(run_root, config=config)
    aggregate, prediction_provenance = _load_all_predictions(
        run_root,
        config=config,
        summary=summary,
        gates=gate_audits,
    )

    class_order = tuple(DEFAULT_VOCALMIND_CONTRACT.words)
    per_seed: list[dict[str, Any]] = []
    reference_ids: tuple[str, ...] | None = None
    reference_labels: np.ndarray | None = None
    for seed in config.seeds:
        trial_ids = tuple(str(value) for value in aggregate[seed]["trial_ids"])
        labels = np.concatenate(aggregate[seed]["labels"]).astype(np.int64, copy=False)
        if len(trial_ids) != 100 or len(set(trial_ids)) != 100 or labels.shape != (100,):
            raise OofIntegrityError(f"seed {seed} is not an exact 100-trial OOF surface")
        class_support = np.bincount(labels, minlength=len(class_order))
        if not np.array_equal(class_support, np.full(len(class_order), 5, dtype=np.int64)):
            raise OofIntegrityError(f"seed {seed} does not contain five trials per fixed class")
        if reference_ids is None:
            reference_ids = trial_ids
            reference_labels = labels.copy()
        elif trial_ids != reference_ids or not np.array_equal(labels, reference_labels):
            raise OofIntegrityError("OOF trial IDs or labels differ across training seeds")

        probabilities = {
            model: np.concatenate(aggregate[seed][model], axis=0).astype(np.float32, copy=False)
            for model in OOF_MODELS
        }
        metrics = {
            model: closed_set_metrics(values, labels, class_order=class_order)
            for model, values in probabilities.items()
        }
        per_seed.append(
            {
                "seed": seed,
                "trial_count": 100,
                "trial_ids": list(trial_ids),
                "trial_ids_sha256": fingerprint_json(list(trial_ids)),
                "labels": labels.tolist(),
                "labels_sha256": _array_fingerprint(labels),
                "class_support": {
                    class_name: int(class_support[index])
                    for index, class_name in enumerate(class_order)
                },
                "probabilities": {
                    model: values.tolist() for model, values in probabilities.items()
                },
                "probabilities_sha256": {
                    model: _array_fingerprint(values)
                    for model, values in probabilities.items()
                },
                "metrics": metrics,
                "primary_contrast_L345_minus_MELx3": {
                    metric: float(metrics["L345"][metric] - metrics["MELx3"][metric])
                    for metric in METRIC_NAMES
                },
            }
        )

    descriptive = {
        model: {
            metric: _descriptive_statistics(
                [float(item["metrics"][model][metric]) for item in per_seed]
            )
            for metric in METRIC_NAMES
        }
        for model in OOF_MODELS
    }
    descriptive_primary_contrast = {
        metric: _descriptive_statistics(
            [
                float(item["primary_contrast_L345_minus_MELx3"][metric])
                for item in per_seed
            ]
        )
        for metric in METRIC_NAMES
    }
    run_fingerprint = str(manifest["fingerprint"])
    csv_bytes = _csv_bytes(
        run_fingerprint=run_fingerprint,
        per_seed=per_seed,
        descriptive=descriptive,
        primary_contrast=descriptive_primary_contrast,
    )
    gate_provenance = [
        {
            "fold": fold,
            "split_fingerprint": gate_audits[fold].split.fingerprint,
            "test_trial_ids_sha256": fingerprint_json(
                list(gate_audits[fold].split.held_out_test_ids)
            ),
            "gate_authorization": dict(gate_audits[fold].authorization),
            "validation_fixed_artifacts": list(gate_audits[fold].validation_artifacts),
            "validation_fixed_artifacts_fingerprint": fingerprint_json(
                list(gate_audits[fold].validation_artifacts)
            ),
        }
        for fold in PRIMARY_FOLDS
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vocalmind_immutable_oof_summary",
        "aggregator_version": AGGREGATOR_VERSION,
        "source_run": {
            "run_manifest_path": "run_manifest.json",
            "run_manifest_sha256": manifest_sha,
            "run_manifest_fingerprint": run_fingerprint,
            "run_summary_path": "summary.json",
            "run_summary_sha256": summary_sha,
            "run_summary_fingerprint": summary["fingerprint"],
            "config_fingerprint": config.fingerprint,
            "source_identity_fingerprint": manifest["source_identity_fingerprint"],
            "aggregation_source_identity_fingerprint": aggregation_source_identity[
                "fingerprint"
            ],
            "aggregation_git_commit": aggregation_source_identity["git"]["commit"],
            "dataset_index_sha256": manifest["dataset_index_sha256"],
            "host_preflight_sha256": manifest["host_preflight_sha256"],
        },
        "frozen_analysis": {
            "folds": list(PRIMARY_FOLDS),
            "training_seeds": list(PRIMARY_SEEDS),
            "models": list(OOF_MODELS),
            "primary_contrast": "L345_minus_MELx3",
            "class_order": list(class_order),
            "ensemble_rule": "fixed_arithmetic_mean_of_softmax_probabilities",
            "model_or_subset_search": False,
            "threshold_selection": "not_applicable_closed_set_argmax",
            "test_used_for_selection": False,
            "test_access": "all_five_gates_validated_before_any_result_or_prediction_read",
        },
        "statistical_scope": {
            "participant_count": 1,
            "biological_inference": False,
            "folds_are_independent_biological_samples": False,
            "seeds_are_independent_biological_samples": False,
            "fold_role": "partition_stitching_into_one_100_trial_oof_surface_per_seed",
            "seed_role": "descriptive_training_stochasticity_conditional_on_one_participant",
            "ci95_label": (
                "descriptive t interval across five training seeds; not a population "
                "or biological-sample confidence interval"
            ),
        },
        "gate_provenance": gate_provenance,
        "prediction_artifacts": prediction_provenance,
        "prediction_artifacts_fingerprint": fingerprint_json(prediction_provenance),
        "per_seed_oof": per_seed,
        "descriptive_across_training_seeds": descriptive,
        "descriptive_primary_contrast_across_training_seeds": descriptive_primary_contrast,
        "metrics_csv_sha256": sha256_bytes(csv_bytes),
    }
    payload["fingerprint"] = fingerprint_json(payload)
    json_bytes = _render_json_bytes(payload)
    reused = _write_immutable_pair(
        output_directory,
        json_bytes=json_bytes,
        csv_bytes=csv_bytes,
    )
    return {
        "json_path": str(output_directory / OUTPUT_JSON_NAME),
        "csv_path": str(output_directory / OUTPUT_CSV_NAME),
        "fingerprint": payload["fingerprint"],
        "reused_identical_output": reused,
        "training_seeds": list(PRIMARY_SEEDS),
        "trials_per_seed": 100,
        "models": list(OOF_MODELS),
        "biological_n": 1,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = aggregate_vocalmind_oof(args.run_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OofIntegrityError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
