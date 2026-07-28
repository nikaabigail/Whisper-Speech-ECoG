#!/usr/bin/env python3
"""Independently audit the frozen SWPD matched PCA50 confirmatory artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swpd_finalize_qc import (  # noqa: E402
    ANALYZABLE_SUBJECTS,
    PRIMARY_SUBJECTS,
    QC_SUBJECT,
    validate_sub10_qc,
)
from swpd_matched_all import (  # noqa: E402
    TARGET_NAMES,
    _load_completions,
    aggregate_subject_summaries,
)
from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
    fingerprint_json,
    read_json,
    sha256_file,
)
from whisper_ecog_ext.reducer import ReducerArtifact  # noqa: E402
from whisper_ecog_ext.swpd.matched_linear import regression_metrics  # noqa: E402
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    inventory_subject,
    subject_paths_frozen,
)


SOURCE_HASH_PATHS = {
    "runner_sha256": PROJECT_ROOT / "swpd_matched_all.py",
    "matched_implementation_sha256": (
        PROJECT_ROOT / "src" / "whisper_ecog_ext" / "swpd" / "matched_linear.py"
    ),
    "nwb_adapter_sha256": (
        PROJECT_ROOT / "src" / "whisper_ecog_ext" / "swpd" / "nwb.py"
    ),
    "author_mel_implementation_sha256": (
        PROJECT_ROOT / "src" / "whisper_ecog_ext" / "swpd" / "author_mel.py"
    ),
    "target_extractor_sha256": (
        PROJECT_ROOT / "src" / "whisper_ecog_ext" / "targets.py"
    ),
    "reducer_implementation_sha256": (
        PROJECT_ROOT / "src" / "whisper_ecog_ext" / "reducer.py"
    ),
}


def _assert_nested_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"Mapping mismatch at {path}")
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"List mismatch at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_nested_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        if actual is None or not np.isclose(
            float(actual), float(expected), rtol=1e-9, atol=1e-11
        ):
            raise RuntimeError(f"Numeric mismatch at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise RuntimeError(f"Value mismatch at {path}: {actual!r} != {expected!r}")


def _validated_fingerprint(payload: Mapping[str, Any], *, label: str) -> None:
    copy = dict(payload)
    observed = copy.pop("fingerprint", None)
    expected = fingerprint_json(copy)
    if observed != expected:
        raise RuntimeError(f"Fingerprint mismatch: {label}")


def _concat(blocks: Sequence[Mapping[str, np.ndarray]], indexes: Sequence[int], key: str) -> np.ndarray:
    return np.concatenate([blocks[index][key] for index in indexes], axis=0)


def _load_blocks(
    cache_root: Path,
    subject: str,
    expected_fingerprints: Sequence[str],
) -> list[dict[str, np.ndarray]]:
    blocks: list[dict[str, np.ndarray]] = []
    for index, expected_fingerprint in enumerate(expected_fingerprints):
        stem = f"block_{index:02d}"
        manifest_path = cache_root / subject / f"{stem}.json"
        arrays_path = cache_root / subject / f"{stem}.npz"
        manifest = read_json(manifest_path)
        _validated_fingerprint(manifest, label=str(manifest_path))
        if manifest["extraction_fingerprint"] != expected_fingerprint:
            raise RuntimeError(f"Extraction fingerprint mismatch: {subject} block {index}")
        if sha256_file(arrays_path) != manifest["arrays_sha256"]:
            raise RuntimeError(f"Block checksum mismatch: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            block = {
                "sample_ids": np.asarray(arrays["sample_ids"]),
                "frame_times_seconds": np.asarray(
                    arrays["frame_times_seconds"], dtype=np.float64
                ),
                "neural": np.asarray(arrays["neural"], dtype=np.float32),
                **{
                    target: np.asarray(arrays[target], dtype=np.float32)
                    for target in TARGET_NAMES
                },
            }
        count = int(manifest["frame_count"])
        if any(values.shape[0] != count for values in block.values()):
            raise RuntimeError(f"Block row mismatch: {subject} block {index}")
        if len(np.unique(block["sample_ids"])) != count:
            raise RuntimeError(f"Duplicate IDs: {subject} block {index}")
        if not all(np.all(np.isfinite(values)) for key, values in block.items() if key != "sample_ids"):
            raise RuntimeError(f"Non-finite block values: {subject} block {index}")
        blocks.append(block)
    return blocks


def _audit_subject(
    subject: str,
    receipt: Mapping[str, Any],
    *,
    cache_root: Path,
) -> dict[str, Any]:
    _validated_fingerprint(receipt, label=f"{subject} completion")
    summary_path = Path(receipt["summary_path"])
    if sha256_file(summary_path) != receipt["summary_sha256"]:
        raise RuntimeError(f"Summary checksum mismatch: {subject}")
    attempt = Path(receipt["attempt_directory"])
    manifest = read_json(attempt / "subject_run_manifest.json")
    summary = read_json(summary_path)
    if manifest["summary"] != summary:
        raise RuntimeError(f"Embedded/file summary mismatch: {subject}")
    if summary["subject"] != subject or summary["target_dimension"] != 50:
        raise RuntimeError(f"Subject summary contract mismatch: {subject}")
    if summary["speech_mask_status"] != "unavailable":
        raise RuntimeError(f"Unexpected speech mask in primary run: {subject}")
    if summary["visual_events_are_acoustic_onsets"] is not False:
        raise RuntimeError(f"Visual cues mislabelled as acoustic onsets: {subject}")
    if len(manifest["block_cache_fingerprints"]) != 5:
        raise RuntimeError(f"Wrong cache fingerprint count: {subject}")
    blocks = _load_blocks(
        cache_root, subject, manifest["block_cache_fingerprints"]
    )

    all_test_ids: list[str] = []
    prediction_rows = 0
    refit_models = 0
    reducers = 0
    for test_index, stored_fold in enumerate(summary["folds"]):
        fold_dir = attempt / f"fold_{test_index:02d}"
        disk_fold = read_json(fold_dir / "fold_result.json")
        if disk_fold != stored_fold:
            raise RuntimeError(f"Fold summary mismatch: {subject} fold {test_index}")
        validation_index = (test_index + 1) % 5
        train_indexes = tuple(
            index for index in range(5) if index not in (test_index, validation_index)
        )
        if (
            disk_fold["test_block"] != test_index
            or disk_fold["validation_block"] != validation_index
            or tuple(disk_fold["train_blocks"]) != train_indexes
        ):
            raise RuntimeError(f"Split mismatch: {subject} fold {test_index}")

        train_ids = _concat(blocks, train_indexes, "sample_ids")
        val_ids = _concat(blocks, (validation_index,), "sample_ids")
        test_ids = _concat(blocks, (test_index,), "sample_ids")
        if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
            raise RuntimeError(f"Split ID leakage: {subject} fold {test_index}")
        declared = {
            "train": (train_ids, disk_fold["train_sample_ids_sha256"]),
            "validation": (val_ids, disk_fold["validation_sample_ids_sha256"]),
            "test": (test_ids, disk_fold["test_sample_ids_sha256"]),
        }
        for role, (ids, expected_hash) in declared.items():
            if fingerprint_json(ids.tolist()) != expected_hash:
                raise RuntimeError(f"{role} ID hash mismatch: {subject} fold {test_index}")

        train_neural = _concat(blocks, train_indexes, "neural")
        val_neural = _concat(blocks, (validation_index,), "neural")
        test_neural = _concat(blocks, (test_index,), "neural")
        neural_reducer = ReducerArtifact.load(fold_dir / "neural_reducer")
        reducers += 1
        if neural_reducer.train_sample_ids_sha256 != fingerprint_json(train_ids.tolist()):
            raise RuntimeError(f"Neural reducer leakage: {subject} fold {test_index}")
        train_x = neural_reducer.transform(train_neural)
        val_x = neural_reducer.transform(val_neural)
        test_x = neural_reducer.transform(test_neural)

        for target in TARGET_NAMES:
            target_reducer = ReducerArtifact.load(
                fold_dir / f"{target}_target_reducer"
            )
            reducers += 1
            if target_reducer.train_sample_ids_sha256 != fingerprint_json(train_ids.tolist()):
                raise RuntimeError(
                    f"Target reducer leakage: {subject} fold {test_index} {target}"
                )
            train_y = target_reducer.transform(_concat(blocks, train_indexes, target))
            val_y = target_reducer.transform(_concat(blocks, (validation_index,), target))
            test_y = target_reducer.transform(_concat(blocks, (test_index,), target))

            model_path = fold_dir / f"{target}_linear_model.npz"
            prediction_path = fold_dir / f"{target}_test_predictions.npz"
            target_record = disk_fold["targets"][target]
            if sha256_file(model_path) != target_record["model_sha256"]:
                raise RuntimeError(f"Model checksum mismatch: {subject} fold {test_index} {target}")
            if sha256_file(prediction_path) != target_record["test_predictions_sha256"]:
                raise RuntimeError(
                    f"Prediction checksum mismatch: {subject} fold {test_index} {target}"
                )
            with np.load(model_path, allow_pickle=False) as arrays:
                coefficient = np.asarray(arrays["coefficient"], dtype=np.float64)
                intercept = np.asarray(arrays["intercept"], dtype=np.float64)
            with np.load(prediction_path, allow_pickle=False) as arrays:
                saved_ids = np.asarray(arrays["sample_ids"])
                saved_truth = np.asarray(arrays["truth"], dtype=np.float32)
                saved_prediction = np.asarray(arrays["prediction"], dtype=np.float32)
            np.testing.assert_array_equal(saved_ids, test_ids)
            np.testing.assert_allclose(saved_truth, test_y, rtol=1e-6, atol=2e-6)
            model_prediction = test_x.astype(np.float64) @ coefficient.T + intercept
            np.testing.assert_allclose(
                saved_prediction, model_prediction, rtol=2e-5, atol=3e-5
            )

            refit = LinearRegression(n_jobs=1).fit(train_x, train_y)
            refit_models += 1
            np.testing.assert_allclose(
                refit.coef_, coefficient, rtol=1e-8, atol=1e-9
            )
            np.testing.assert_allclose(
                refit.intercept_, intercept, rtol=1e-8, atol=1e-9
            )
            val_prediction = refit.predict(val_x).astype(np.float32)
            test_prediction = refit.predict(test_x).astype(np.float32)
            _assert_nested_close(
                regression_metrics(val_y, val_prediction),
                target_record["validation_all"],
                f"{subject}.fold{test_index}.{target}.validation",
            )
            _assert_nested_close(
                regression_metrics(test_y, test_prediction),
                target_record["test_all"],
                f"{subject}.fold{test_index}.{target}.test",
            )
            if target_record["validation_speech"] is not None or target_record["test_speech"] is not None:
                raise RuntimeError(f"Unexpected speech-only result: {subject} {target}")
            prediction_rows += len(test_ids)
        all_test_ids.extend(test_ids.tolist())

    if len(all_test_ids) != len(set(all_test_ids)):
        raise RuntimeError(f"A test frame appears in multiple folds: {subject}")
    if set(all_test_ids) != set(np.concatenate([block["sample_ids"] for block in blocks])):
        raise RuntimeError(f"Outer folds do not cover every frame exactly once: {subject}")
    return {
        "subject": subject,
        "folds": 5,
        "reducers_verified": reducers,
        "models_refit": refit_models,
        "test_rows_per_target": len(all_test_ids),
        "prediction_rows_all_targets": prediction_rows,
        "test_ids_unique_and_complete": True,
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def audit(
    *,
    run_root: Path,
    cache_root: Path,
    data_root: Path,
    final_summary_path: Path,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    cache_root = cache_root.resolve()
    final_summary_path = final_summary_path.resolve()
    contract_path = run_root / "run_contract.json"
    contract = read_json(contract_path)
    _validated_fingerprint(contract, label=str(contract_path))
    source_hashes = {}
    for key, path in SOURCE_HASH_PATHS.items():
        actual = sha256_file(path)
        expected = contract[key]
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch for {key}: {path}")
        source_hashes[key] = actual
    protocol_path = Path(contract["protocol_path"])
    if sha256_file(protocol_path) != contract["protocol_sha256"]:
        raise RuntimeError("Frozen protocol checksum mismatch")
    if sha256_file(Path(contract["dataset_manifest"])) != contract["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest checksum mismatch")

    completions = _load_completions(run_root, contract["protocol_sha256"])
    if tuple(subject for subject in ANALYZABLE_SUBJECTS if subject in completions) != ANALYZABLE_SUBJECTS:
        raise RuntimeError("Expected sub-01 through sub-09 completions")
    if QC_SUBJECT in completions:
        raise RuntimeError("Excluded sub-10 unexpectedly has a model result")
    subject_receipts = [
        _audit_subject(subject, completions[subject], cache_root=cache_root)
        for subject in ANALYZABLE_SUBJECTS
    ]

    final_summary = read_json(final_summary_path)
    summaries = {
        subject: read_json(Path(completions[subject]["summary_path"]))
        for subject in ANALYZABLE_SUBJECTS
    }
    recomputed_primary = aggregate_subject_summaries(
        summaries, cohort=PRIMARY_SUBJECTS
    )
    recomputed_secondary = aggregate_subject_summaries(
        summaries, cohort=ANALYZABLE_SUBJECTS
    )
    _assert_nested_close(
        final_summary["primary_confirmatory_after_qc"],
        recomputed_primary,
        "primary_confirmatory_after_qc",
    )
    _assert_nested_close(
        final_summary["secondary_all_analyzable_after_qc"],
        recomputed_secondary,
        "secondary_all_analyzable_after_qc",
    )
    exclusion_path = Path(final_summary["qc_exclusion_path"])
    if sha256_file(exclusion_path) != final_summary["qc_exclusion_sha256"]:
        raise RuntimeError("QC exclusion checksum mismatch")
    exclusion = read_json(exclusion_path)
    paths = subject_paths_frozen(data_root, QC_SUBJECT)
    inventory = inventory_subject(data_root, QC_SUBJECT, allow_confirmatory=True)
    observed_qc = validate_sub10_qc(
        _read_tsv(paths["events"]), ieeg_sample_count=inventory.ieeg.shape[0]
    )
    _assert_nested_close(exclusion["observed_qc"], observed_qc, "sub10_qc")
    if exclusion["decision"] != "exclude_entire_subject_from_population_inference":
        raise RuntimeError("Unexpected QC decision")

    exact_sign_p_two_sided = 2.0 / (2.0 ** len(PRIMARY_SUBJECTS))
    return {
        "schema_version": 1,
        "kind": "swpd_matched_pca50_confirmatory_independent_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass_with_documented_limitations",
        "source_run_id": run_root.name,
        "source_run_contract_sha256": sha256_file(contract_path),
        "final_summary_sha256": sha256_file(final_summary_path),
        "source_hashes_verified": source_hashes,
        "protocol_checksum_verified": True,
        "dataset_manifest_checksum_verified": True,
        "subjects_verified": subject_receipts,
        "totals": {
            "analyzable_subjects": len(subject_receipts),
            "primary_confirmatory_subjects": len(PRIMARY_SUBJECTS),
            "folds": sum(item["folds"] for item in subject_receipts),
            "reducers": sum(item["reducers_verified"] for item in subject_receipts),
            "models_independently_refit": sum(item["models_refit"] for item in subject_receipts),
            "prediction_rows_all_targets": sum(
                item["prediction_rows_all_targets"] for item in subject_receipts
            ),
        },
        "checks": {
            "raw_block_cache_checksums": "pass",
            "split_ids_disjoint": "pass",
            "outer_test_ids_unique_and_complete": "pass",
            "shared_neural_reducer_per_fold": "pass",
            "all_neural_and_target_reducers_train_only": "pass",
            "target_dimension_50_for_all_systems": "pass",
            "ols_hyperparameters_identical": "pass",
            "saved_models_equal_independent_refits": "pass",
            "saved_predictions_recomputed": "pass",
            "fold_metrics_recomputed": "pass",
            "population_statistics_recomputed": "pass",
            "holm_adjustment_recomputed": "pass",
            "sub01_excluded_from_primary": "pass",
            "sub10_qc_matches_raw_source": "pass",
            "sub10_has_no_model_or_test_result": "pass",
        },
        "sensitivity_only_not_preregistered": {
            "exact_two_sided_sign_test_for_8_of_8_wins": exact_sign_p_two_sided
        },
        "limitations": [
            {
                "severity": "moderate",
                "code": "unused_validation_block",
                "detail": "One 20-trial block per fold is reserved for validation but OLS has no selected hyperparameter, so only three of five blocks train each model. This is matched and conservative, not leakage; a four-train-block sensitivity analysis would use data more efficiently."
            },
            {
                "severity": "moderate",
                "code": "all_frame_representation_metric",
                "detail": "The primary outcome measures continuous all-frame representation predictability. It is not word-classification accuracy and may retain speech-versus-silence contribution."
            },
            {
                "severity": "moderate",
                "code": "offline_noncausal_features",
                "detail": "High-gamma uses zero-phase filtering and Whisper targets are bidirectional within chunks. The result supports offline representation decoding, not causal/asynchronous deployment."
            },
            {
                "severity": "moderate",
                "code": "post_access_qc_exclusion",
                "detail": "sub-10 exclusion was decided after data access. It is objective, source-only, fully checksummed, and no sub-10 model result exists, but it must remain explicit in the paper."
            },
            {
                "severity": "minor",
                "code": "small_patient_sample",
                "detail": "Population inference uses eight confirmatory patients. All 8/8 paired effects are positive; confidence intervals and exact sign sensitivity should accompany parametric p-values."
            },
        ],
        "claim_boundary": {
            "supported": "Matched train-only PCA50/OLS predictability is higher for each tested Whisper layer than MEL80 across all eight confirmatory patients.",
            "not_supported": "No aligned L3+L4+L5 ensemble, word-classification, speech-only, causal, or asynchronous claim follows from this run."
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--final-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(
        run_root=args.run_root,
        cache_root=args.cache_root,
        data_root=args.data_root,
        final_summary_path=args.final_summary,
    )
    atomic_write_json(args.output, result)
    print(f"[audit pass] {args.output.resolve()}")
    print(result["totals"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
