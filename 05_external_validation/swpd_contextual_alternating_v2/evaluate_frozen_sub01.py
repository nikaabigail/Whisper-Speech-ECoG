#!/usr/bin/env python3
"""Evaluate already-frozen contextual alternating and PCA50 control on sub-01 folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = MODULE_ROOT.parent
sys.path[:0] = [str(EXTERNAL_ROOT), str(EXTERNAL_ROOT / "src")]

from swpd_contextual_alternating_v2.core import (  # noqa: E402
    AffineMap,
    PCATransform,
    Standardizer,
    TargetSearchSpace,
    mse,
    project_scores,
)
from swpd_contextual_alternating_v2.fit_select_sub01 import (  # noqa: E402
    EXPECTED_REFERENCE_L4,
    EXPECTED_REFERENCE_SHA256,
    _atomic_npz,
    load_block,
)
from swpd_protocol_bridge.bridge_core import component_metrics  # noqa: E402
from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
    fingerprint_json,
    read_json,
    sha256_file,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _standardizer(arrays: Mapping[str, np.ndarray], prefix: str) -> Standardizer:
    return Standardizer(arrays[f"{prefix}_mean"], arrays[f"{prefix}_scale"])


def _pca(arrays: Mapping[str, np.ndarray], prefix: str) -> PCATransform:
    return PCATransform(
        arrays[f"{prefix}_mean"],
        arrays[f"{prefix}_components"],
        arrays[f"{prefix}_explained_variance"],
        bool(int(arrays[f"{prefix}_whiten"][0])),
    )


def _affine(arrays: Mapping[str, np.ndarray], prefix: str) -> AffineMap:
    return AffineMap(arrays[f"{prefix}_coef"], arrays[f"{prefix}_intercept"])


def _metric_bundle(
    raw_l4: np.ndarray,
    mel_z: np.ndarray,
    x: np.ndarray,
    target_space: TargetSearchSpace,
    projector: np.ndarray,
    decoder: AffineMap,
    probe: AffineMap,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    truth_scores = project_scores(target_space.transform(raw_l4), projector)
    predicted_scores = decoder.predict(x)
    predicted_mel = probe.predict(predicted_scores)
    truth_l4_z = target_space.scaler.transform(raw_l4)
    predicted_l4_z = target_space.reconstruct_standardized(predicted_scores, projector)
    metrics = {
        "common_mel80": component_metrics(mel_z, predicted_mel),
        "target_score50": component_metrics(truth_scores, predicted_scores),
        "l4_full512": component_metrics(truth_l4_z, predicted_l4_z),
        "l4_full512_mse": mse(truth_l4_z, predicted_l4_z),
    }
    arrays = {
        "truth_score50": truth_scores,
        "predicted_score50": predicted_scores,
        "predicted_mel80_z": predicted_mel,
        "predicted_l4_z": predicted_l4_z,
    }
    return metrics, arrays


def _validate_selection(path: Path, contract_fp: str) -> dict[str, Any]:
    selection = read_json(path)
    stored = selection.get("fingerprint")
    payload = {key: value for key, value in selection.items() if key != "fingerprint"}
    if stored != fingerprint_json(payload):
        raise RuntimeError(f"invalid frozen selection fingerprint: {path}")
    if selection.get("run_contract_fingerprint") != contract_fp:
        raise RuntimeError("frozen selection belongs to another v2 contract")
    if selection.get("test_evaluated") is not False:
        raise RuntimeError("selection is not pre-test frozen")
    for path_key, hash_key in (
        ("artifact_path", "artifact_sha256"),
        ("validation_predictions_path", "validation_predictions_sha256"),
    ):
        artifact = Path(selection[path_key])
        if not artifact.is_file() or sha256_file(artifact) != selection[hash_key]:
            raise RuntimeError(f"frozen artifact changed: {artifact}")
    return selection


def _mean_fold_metric(folds: list[dict[str, Any]], arm: str, region: str) -> float:
    return float(np.mean([
        fold[arm]["common_mel80"][region]["mean_pearson_r"] for fold in folds
    ]))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cache = args.cache_dir.expanduser().resolve()
    reference_path = args.reference_summary.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        manifest = read_json(run_dir / "artifact_manifest.json")
        if sha256_file(summary_path) != manifest.get("summary_sha256"):
            raise RuntimeError("completed v2 summary changed")
        print(f"ALREADY COMPLETE | {summary_path}", flush=True)
        return 0
    if sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("latest contextual reference changed")
    reference = read_json(reference_path)
    contract_path = run_dir / "run_contract.json"
    contract = read_json(contract_path)
    compatibility = {
        key: value for key, value in contract.items()
        if key not in ("created_utc", "compatibility_fingerprint")
    }
    contract_fp = fingerprint_json(compatibility)
    if contract.get("compatibility_fingerprint") != contract_fp:
        raise RuntimeError("v2 run contract fingerprint is invalid")
    if contract["implementation_sha256"]["core"] != sha256_file(MODULE_ROOT / "core.py"):
        raise RuntimeError("v2 numerical core changed after fit")
    if contract["implementation_sha256"]["fit_runner"] != sha256_file(MODULE_ROOT / "fit_select_sub01.py"):
        raise RuntimeError("v2 fit runner changed after fit")
    if contract["implementation_sha256"]["evaluate_runner"] != sha256_file(Path(__file__)):
        raise RuntimeError("v2 frozen evaluator changed after fit")

    selections = []
    inventory_items = []
    for fold in range(5):
        selection_path = run_dir / "folds" / f"fold_{fold:02d}" / "selection_frozen.json"
        selection = _validate_selection(selection_path, contract_fp)
        selections.append(selection)
        inventory_items.append({
            "fold": fold,
            "selection_path": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "artifact_path": selection["artifact_path"],
            "artifact_sha256": selection["artifact_sha256"],
            "validation_predictions_path": selection["validation_predictions_path"],
            "validation_predictions_sha256": selection["validation_predictions_sha256"],
        })
    pre_test = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_covariance_alternating_v2_pre_test_inventory",
        "run_contract_path": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
        "run_contract_fingerprint": contract_fp,
        "folds": inventory_items,
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "all_folds_frozen": True,
        "test_metrics_used_for_selection": False,
        "created_utc": _now(),
    }
    pre_test["fingerprint"] = fingerprint_json(pre_test)
    pre_test_path = run_dir / "pre_test_inventory.json"
    if pre_test_path.is_file():
        existing_pre_test = read_json(pre_test_path)
        existing_payload = {
            key: value for key, value in existing_pre_test.items() if key != "fingerprint"
        }
        if existing_pre_test.get("fingerprint") != fingerprint_json(existing_payload):
            raise RuntimeError("pre-test inventory fingerprint is invalid")
        for key in (
            "run_contract_sha256", "run_contract_fingerprint",
            "evaluation_source_sha256", "folds",
        ):
            if existing_pre_test.get(key) != pre_test.get(key):
                raise RuntimeError("pre-test inventory differs from current frozen inputs")
        pre_test = existing_pre_test
    else:
        atomic_write_json(pre_test_path, pre_test, overwrite=False)
    authorization_path = run_dir / "test_gate_authorization.json"
    if authorization_path.is_file():
        authorization = read_json(authorization_path)
        authorization_payload = {
            key: value for key, value in authorization.items() if key != "fingerprint"
        }
        if authorization.get("fingerprint") != fingerprint_json(authorization_payload):
            raise RuntimeError("test authorization fingerprint is invalid")
        if authorization.get("pre_test_inventory_sha256") != sha256_file(pre_test_path):
            raise RuntimeError("test authorization belongs to another inventory")
    else:
        authorization = {
            "schema_version": 1,
            "kind": "explicit_separate_command_test_authorization",
            "pre_test_inventory_path": str(pre_test_path),
            "pre_test_inventory_sha256": sha256_file(pre_test_path),
            "authorization": "evaluate only immutable selected and fixed-control arms",
            "authorized_utc": _now(),
        }
        authorization["fingerprint"] = fingerprint_json(authorization)
        atomic_write_json(authorization_path, authorization, overwrite=False)
    print("[gate] all five selections immutable; opening fold-role test arrays", flush=True)

    reference_folds = {
        int(item["fold"]): item
        for item in reference["results"]["targets"]["L4"]["folds"]
    }
    fold_results: list[dict[str, Any]] = []
    prediction_paths = []
    for fold, selection in enumerate(selections):
        fold_root = run_dir / "folds" / f"fold_{fold:02d}"
        completion_path = fold_root / "test_complete.json"
        if completion_path.is_file():
            completion = read_json(completion_path)
            completion_payload = {
                key: value for key, value in completion.items() if key != "fingerprint"
            }
            if completion.get("fingerprint") != fingerprint_json(completion_payload):
                raise RuntimeError(f"fold {fold} test completion fingerprint is invalid")
            for path_key, hash_key in (
                ("predictions_path", "predictions_sha256"),
                ("metrics_path", "metrics_sha256"),
            ):
                path = Path(completion[path_key])
                if not path.is_file() or sha256_file(path) != completion[hash_key]:
                    raise RuntimeError(f"fold {fold} completed test artifact changed")
            result = read_json(Path(completion["metrics_path"]))
            fold_results.append(result)
            prediction_paths.append({
                "fold": fold,
                "predictions_path": completion["predictions_path"],
                "predictions_sha256": completion["predictions_sha256"],
                "metrics_path": completion["metrics_path"],
                "metrics_sha256": completion["metrics_sha256"],
                "completion_path": str(completion_path),
                "completion_sha256": sha256_file(completion_path),
            })
            print(f"[fold {fold}] validated test evaluation reused", flush=True)
            continue
        test_block = load_block(cache, fold)
        artifact_path = Path(selection["artifact_path"])
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        neural_scaler = _standardizer(arrays, "neural_scaler")
        neural_pca = _pca(arrays, "neural_pca")
        mel_scaler = _standardizer(arrays, "mel_scaler")
        target_space = TargetSearchSpace(
            _standardizer(arrays, "target_scaler"),
            _pca(arrays, "target_search_pca"),
            int(arrays["selected_projector"].shape[0]),
        )
        x = neural_pca.transform(neural_scaler.transform(test_block.neural))
        mel_z = mel_scaler.transform(test_block.mel80)
        selected_metrics, selected_arrays = _metric_bundle(
            test_block.l4, mel_z, x, target_space,
            arrays["selected_projector"],
            _affine(arrays, "selected_decoder"),
            _affine(arrays, "selected_probe"),
        )
        control_metrics, control_arrays = _metric_bundle(
            test_block.l4, mel_z, x, target_space,
            arrays["control_projector"],
            _affine(arrays, "control_decoder"),
            _affine(arrays, "control_probe"),
        )
        expected = float(
            reference_folds[fold]["common_mel80"]["all_bins"]["mean_pearson_r"]
        )
        observed = float(control_metrics["common_mel80"]["all_bins"]["mean_pearson_r"])
        if abs(observed - expected) > 1e-9:
            raise RuntimeError(
                f"fold {fold} fixed PCA50 control does not reproduce latest reference: "
                f"{observed:.12g} != {expected:.12g}"
            )
        result = {
            "fold": fold,
            "train_blocks": selection["train_blocks"],
            "validation_block": selection["validation_block"],
            "test_block": fold,
            "selected_cycle": selection["selected_cycle"],
            "fixed_control": control_metrics,
            "alternating_selected": selected_metrics,
            "delta_common_mel80_all80": (
                float(selected_metrics["common_mel80"]["all_bins"]["mean_pearson_r"])
                - observed
            ),
            "cycle_zero_reference_error": observed - expected,
        }
        fold_results.append(result)
        prediction_path = fold_root / "paired_test_predictions.npz"
        _atomic_npz(prediction_path, {
            "sample_ids": test_block.sample_ids,
            "times": test_block.times,
            "truth_mel80_z": mel_z,
            "truth_l4": test_block.l4,
            "control_truth_score50": control_arrays["truth_score50"],
            "control_predicted_score50": control_arrays["predicted_score50"],
            "control_predicted_mel80_z": control_arrays["predicted_mel80_z"],
            "selected_truth_score50": selected_arrays["truth_score50"],
            "selected_predicted_score50": selected_arrays["predicted_score50"],
            "selected_predicted_mel80_z": selected_arrays["predicted_mel80_z"],
        })
        metrics_path = fold_root / "paired_test_metrics.json"
        atomic_write_json(metrics_path, result, overwrite=False)
        completion = {
            "schema_version": 1,
            "kind": "swpd_sub01_contextual_covariance_alternating_v2_fold_test_completion",
            "fold": fold,
            "run_contract_fingerprint": contract_fp,
            "selection_sha256": sha256_file(
                run_dir / "folds" / f"fold_{fold:02d}" / "selection_frozen.json"
            ),
            "predictions_path": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "completed_utc": _now(),
        }
        completion["fingerprint"] = fingerprint_json(completion)
        atomic_write_json(completion_path, completion, overwrite=False)
        prediction_paths.append({
            "fold": fold,
            "predictions_path": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "completion_path": str(completion_path),
            "completion_sha256": sha256_file(completion_path),
        })
        print(
            f"[fold {fold}] fixed={observed:.6f} "
            f"selected={float(selected_metrics['common_mel80']['all_bins']['mean_pearson_r']):.6f} "
            f"delta={result['delta_common_mel80_all80']:+.6f}",
            flush=True,
        )

    control_all = _mean_fold_metric(fold_results, "fixed_control", "all_bins")
    selected_all = _mean_fold_metric(fold_results, "alternating_selected", "all_bins")
    control_low = _mean_fold_metric(fold_results, "fixed_control", "lower_20_bins")
    selected_low = _mean_fold_metric(fold_results, "alternating_selected", "lower_20_bins")
    if abs(control_all - EXPECTED_REFERENCE_L4) > 1e-9:
        raise RuntimeError("aggregate fixed control does not reproduce latest contextual L4")
    summary = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_covariance_alternating_v2_summary",
        "development_only": True,
        "base": "latest contextual L4 protocol sent to Ossadtchi",
        "run_contract_fingerprint": contract_fp,
        "folds": fold_results,
        "aggregate": {
            "fixed_pca50_control": {
                "all_bins_mean_r": control_all,
                "lower_20_bins_mean_r": control_low,
            },
            "alternating_selected": {
                "all_bins_mean_r": selected_all,
                "lower_20_bins_mean_r": selected_low,
            },
            "paired_delta_selected_minus_fixed": {
                "all_bins_mean_r": selected_all - control_all,
                "lower_20_bins_mean_r": selected_low - control_low,
                "fold_values": [
                    item["delta_common_mel80_all80"] for item in fold_results
                ],
                "wins": int(sum(
                    item["delta_common_mel80_all80"] > 0 for item in fold_results
                )),
            },
        },
        "cycle_zero_reference_r": EXPECTED_REFERENCE_L4,
        "cycle_zero_reproduction_error": control_all - EXPECTED_REFERENCE_L4,
        "test_evaluated_only_after_frozen_selection": True,
        "completed_utc": _now(),
    }
    atomic_write_json(summary_path, summary, overwrite=False)
    manifest = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_covariance_alternating_v2_artifact_manifest",
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "pre_test_inventory_path": str(pre_test_path),
        "pre_test_inventory_sha256": sha256_file(pre_test_path),
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "fold_artifacts": prediction_paths,
        "created_utc": _now(),
    }
    manifest["fingerprint"] = fingerprint_json(manifest)
    atomic_write_json(run_dir / "artifact_manifest.json", manifest, overwrite=False)
    print(
        f"EVALUATION COMPLETE | fixed={control_all:.6f} selected={selected_all:.6f} "
        f"delta={selected_all - control_all:+.6f} | {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
