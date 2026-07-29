#!/usr/bin/env python3
"""Open fold-role tests only after all contextual neural E2E selections are frozen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = MODULE_ROOT.parent
sys.path[:0] = [str(EXTERNAL_ROOT), str(EXTERNAL_ROOT / "src")]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from swpd_contextual_alternating_v2.core import (  # noqa: E402
    AffineMap,
    PCATransform,
    Standardizer,
    TargetSearchSpace,
    mse,
    project_scores,
)
from swpd_contextual_neural_e2e.core import ContextualResidualDecoder  # noqa: E402
from swpd_contextual_neural_e2e.fit_select_sub01 import (  # noqa: E402
    ALL_FOLDS,
    CHANNELS,
    CONTEXT_STEPS,
    DEFAULT_SEEDS,
    EXPECTED_REFERENCE_L4,
    EXPECTED_REFERENCE_SHA256,
    OUTPUT_DIM,
    PRODUCTION_BATCH_SIZE,
    PRODUCTION_EPOCHS_PER_CYCLE,
    PRODUCTION_GRAD_CLIP,
    PRODUCTION_LEARNING_RATE,
    PRODUCTION_MAX_CYCLES,
    PRODUCTION_WEIGHT_DECAY,
    _predict,
    _validated_selection,
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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise RuntimeError(f"existing immutable JSON differs: {path}")
        return
    atomic_write_json(path, dict(payload), overwrite=False)


def _standardizer(payload: Mapping[str, Any], prefix: str) -> Standardizer:
    return Standardizer(
        np.asarray(payload[f"{prefix}_mean"]),
        np.asarray(payload[f"{prefix}_scale"]),
    )


def _pca(payload: Mapping[str, Any], prefix: str) -> PCATransform:
    return PCATransform(
        np.asarray(payload[f"{prefix}_mean"]),
        np.asarray(payload[f"{prefix}_components"]),
        np.asarray(payload[f"{prefix}_explained_variance"]),
        bool(payload[f"{prefix}_whiten"]),
    )


def _affine(payload: Mapping[str, Any], prefix: str) -> AffineMap:
    return AffineMap(
        np.asarray(payload[f"{prefix}_coef"]),
        np.asarray(payload[f"{prefix}_intercept"]),
    )


def _metric_bundle(
    *,
    prediction: np.ndarray,
    raw_l4: np.ndarray,
    mel_z: np.ndarray,
    target_space: TargetSearchSpace,
    projector: np.ndarray,
    probe: AffineMap,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    whitened = target_space.transform(raw_l4)
    truth_scores = project_scores(whitened, projector)
    predicted_mel = probe.predict(prediction)
    truth_l4_z = target_space.scaler.transform(raw_l4)
    predicted_l4_z = target_space.reconstruct_standardized(prediction, projector)
    metrics = {
        "common_mel80": component_metrics(mel_z, predicted_mel),
        "target_score50": component_metrics(truth_scores, prediction),
        "l4_full512": component_metrics(truth_l4_z, predicted_l4_z),
        "l4_full512_mse": mse(truth_l4_z, predicted_l4_z),
    }
    return metrics, {
        "truth_score50": truth_scores,
        "predicted_score50": prediction,
        "predicted_mel80_z": predicted_mel,
        "predicted_l4_z": predicted_l4_z,
    }


def _primary(metrics: Mapping[str, Any]) -> float:
    if int(metrics["common_mel80"]["all_bins"].get("component_count", -1)) != 80:
        raise RuntimeError("primary MEL80 metric does not contain exactly 80 bins")
    return float(metrics["common_mel80"]["all_bins"]["mean_pearson_r"])


def _validate_completion(
    path: Path,
    contract_fp: str,
    *,
    expected_seed: int,
    expected_fold: int,
    expected_selection_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion = read_json(path)
    stored = completion.get("fingerprint")
    payload = {key: value for key, value in completion.items() if key != "fingerprint"}
    if stored != fingerprint_json(payload):
        raise RuntimeError(f"invalid test completion fingerprint: {path}")
    if completion.get("run_contract_fingerprint") != contract_fp:
        raise RuntimeError("test completion belongs to another contract")
    if (
        int(completion.get("seed", -1)) != expected_seed
        or int(completion.get("fold", -1)) != expected_fold
        or completion.get("selection_sha256") != expected_selection_sha256
    ):
        raise RuntimeError("test completion seed/fold/selection identity changed")
    for path_key, hash_key in (
        ("metrics_path", "metrics_sha256"),
        ("predictions_path", "predictions_sha256"),
    ):
        artifact = Path(completion[path_key])
        if not artifact.is_file() or sha256_file(artifact) != completion[hash_key]:
            raise RuntimeError(f"completed test artifact changed: {artifact}")
    metrics = read_json(Path(completion["metrics_path"]))
    if (
        int(metrics.get("seed", -1)) != expected_seed
        or int(metrics.get("fold", -1)) != expected_fold
        or int(metrics.get("test_block", -1)) != expected_fold
    ):
        raise RuntimeError("completed metrics seed/fold identity changed")
    return completion, metrics


def _fold_mean(rows: list[dict[str, Any]], arm: str, region: str) -> float:
    return float(np.mean([
        row[arm]["common_mel80"][region]["mean_pearson_r"] for row in rows
    ]))


def _sample_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": values,
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "n_seeds": int(len(array)),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    args = parse_args(argv)
    cache = args.cache_dir.expanduser().resolve()
    reference_path = args.reference_summary.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "artifact_manifest.json"
    existing_summary: dict[str, Any] | None = None
    completed_manifest: dict[str, Any] | None = None
    if summary_path.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest_payload = {
            key: value for key, value in manifest.items() if key != "fingerprint"
        }
        if manifest.get("fingerprint") != fingerprint_json(manifest_payload):
            raise RuntimeError("completed artifact manifest fingerprint is invalid")
        if sha256_file(summary_path) != manifest.get("summary_sha256"):
            raise RuntimeError("completed summary changed")
        existing_summary = read_json(summary_path)
        completed_manifest = manifest
    elif summary_path.is_file():
        # A crash may happen after the atomic summary write but before the
        # manifest write. Rebuild only the missing manifest after validating
        # that all deterministic summary fields still agree below.
        existing_summary = read_json(summary_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("latest contextual reference changed")
    reference = read_json(reference_path)
    contract_path = run_dir / "run_contract.json"
    fit_summary_path = run_dir / "fit_summary.json"
    contract = read_json(contract_path)
    fit_summary = read_json(fit_summary_path)
    compatibility = {
        key: value for key, value in contract.items()
        if key not in ("compatibility_fingerprint", "created_utc")
    }
    contract_fp = fingerprint_json(compatibility)
    if contract.get("compatibility_fingerprint") != contract_fp:
        raise RuntimeError("run contract fingerprint is invalid")
    if contract.get("diagnostic_smoke") is not False:
        raise RuntimeError("diagnostic-smoke runs are not authorized to open test")
    if tuple(contract.get("seeds", [])) != DEFAULT_SEEDS:
        raise RuntimeError("production test requires the frozen five-seed contract")
    if tuple(contract.get("folds", [])) != ALL_FOLDS:
        raise RuntimeError("production test requires all five folds")
    if contract.get("device") != "cuda" or args.device != "cuda":
        raise RuntimeError("production fit and frozen evaluation both require CUDA")
    expected_determinism = {
        "torch_use_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
    if contract.get("deterministic_runtime") != expected_determinism:
        raise RuntimeError("deterministic runtime contract changed")
    if os.path.normcase(str(Path(contract.get("cache_dir", "")).resolve())) != os.path.normcase(
        str(cache)
    ):
        raise RuntimeError("evaluation cache path differs from the frozen fit contract")
    if os.path.normcase(
        str(Path(contract.get("reference_summary", "")).resolve())
    ) != os.path.normcase(str(reference_path)):
        raise RuntimeError("evaluation reference path differs from the frozen fit contract")
    frozen_cache_receipts = contract.get("cache_receipts", {})
    for fold in ALL_FOLDS:
        manifest_path = cache / f"block_{fold:02d}.json"
        receipt = frozen_cache_receipts.get(f"block_{fold}")
        if not isinstance(receipt, dict):
            raise RuntimeError(f"missing frozen cache receipt for block {fold}")
        if sha256_file(manifest_path) != receipt.get("manifest_sha256"):
            raise RuntimeError(f"cache manifest differs from fit contract: block {fold}")
        cache_manifest = read_json(manifest_path)
        arrays_sha256 = cache_manifest.get("arrays_sha256")
        arrays_path = cache / str(cache_manifest.get("arrays_file", ""))
        if (
            arrays_sha256 != receipt.get("declared_arrays_sha256")
            or not arrays_path.is_file()
            or sha256_file(arrays_path) != arrays_sha256
        ):
            raise RuntimeError(f"cache arrays differ from fit contract: block {fold}")
    expected_profile = {
        "max_cycles": PRODUCTION_MAX_CYCLES,
        "epochs_per_cycle": PRODUCTION_EPOCHS_PER_CYCLE,
        "batch_size": PRODUCTION_BATCH_SIZE,
        "learning_rate": PRODUCTION_LEARNING_RATE,
        "weight_decay": PRODUCTION_WEIGHT_DECAY,
        "grad_clip": PRODUCTION_GRAD_CLIP,
        "max_train_batches": None,
        "max_eval_batches": None,
    }
    for key, expected in expected_profile.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"production hyperparameter changed: {key}")
    if fit_summary.get("all_selections_frozen") is not True:
        raise RuntimeError("fit-only stage is not complete")
    if fit_summary.get("run_contract_fingerprint") != contract_fp:
        raise RuntimeError("fit summary belongs to another run contract")
    if fit_summary.get("diagnostic_smoke") is not False:
        raise RuntimeError("diagnostic fit summary cannot authorize test")
    if fit_summary.get("test_evaluated") is not False:
        raise RuntimeError("fit summary is not a pre-test receipt")
    expected_count = len(DEFAULT_SEEDS) * len(ALL_FOLDS)
    if int(fit_summary.get("selection_count", -1)) != expected_count:
        raise RuntimeError("fit summary does not contain all fold x seed selections")
    expected_sources = {
        "fit_runner": sha256_file(MODULE_ROOT / "fit_select_sub01.py"),
        "core": sha256_file(MODULE_ROOT / "core.py"),
        "evaluator": sha256_file(Path(__file__)),
        "preflight": sha256_file(MODULE_ROOT / "preflight.py"),
        "linear_core": sha256_file(EXTERNAL_ROOT / "swpd_contextual_alternating_v2" / "core.py"),
        "bridge_core": sha256_file(EXTERNAL_ROOT / "swpd_protocol_bridge" / "bridge_core.py"),
        "run_fit_ps1": sha256_file(MODULE_ROOT / "scripts" / "run_fit.ps1"),
        "start_fit_background_ps1": sha256_file(
            MODULE_ROOT / "scripts" / "start_fit_background.ps1"
        ),
        "watch_fit_ps1": sha256_file(MODULE_ROOT / "scripts" / "watch_fit.ps1"),
        "run_evaluate_frozen_ps1": sha256_file(
            MODULE_ROOT / "scripts" / "run_evaluate_frozen.ps1"
        ),
    }
    if contract.get("implementation_sha256") != expected_sources:
        raise RuntimeError("implementation changed after fit-only selection")

    selections: dict[tuple[int, int], dict[str, Any]] = {}
    inventory_items: list[dict[str, Any]] = []
    for seed in DEFAULT_SEEDS:
        for fold in ALL_FOLDS:
            item_root = run_dir / "seeds" / f"seed_{seed}" / "folds" / f"fold_{fold:02d}"
            selection_path = item_root / "selection_frozen.json"
            selection = _validated_selection(
                selection_path,
                contract_fp,
                expected_seed=seed,
                expected_fold=fold,
            )
            if int(selection["seed"]) != seed or int(selection["fold"]) != fold:
                raise RuntimeError("selection seed/fold identity changed")
            expected_validation = (fold + 1) % len(ALL_FOLDS)
            expected_train = [
                index for index in ALL_FOLDS
                if index not in (fold, expected_validation)
            ]
            if (
                int(selection.get("test_block", -1)) != fold
                or int(selection.get("validation_block", -1)) != expected_validation
                or list(selection.get("train_blocks", [])) != expected_train
            ):
                raise RuntimeError("selection fold-role assignment changed")
            for receipt_key in ("q0_receipt", "selected_alternating_q_receipt"):
                receipt = selection[receipt_key]
                if (
                    int(receipt.get("rank", -1)) != OUTPUT_DIM
                    or float(receipt.get("orthogonality_frobenius_error", 1.0)) > 1e-8
                    or float(receipt.get("projected_train_variance_min", 0.0)) < 0.99
                    or float(receipt.get("projected_train_variance_max", 2.0)) > 1.01
                ):
                    raise RuntimeError("frozen projector integrity receipt failed")
            selections[(seed, fold)] = selection
            inventory_items.append({
                "seed": seed,
                "fold": fold,
                "selection_path": str(selection_path),
                "selection_sha256": sha256_file(selection_path),
                "artifact_path": selection["artifact_path"],
                "artifact_sha256": selection["artifact_sha256"],
            })

    # This immutable inventory is created before any fold is loaded in its test role.
    pre_test = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_neural_e2e_pre_test_inventory",
        "run_contract_path": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
        "run_contract_fingerprint": contract_fp,
        "fit_summary_path": str(fit_summary_path),
        "fit_summary_sha256": sha256_file(fit_summary_path),
        "selection_count": len(inventory_items),
        "expected_selection_count": expected_count,
        "selections": inventory_items,
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "all_fold_seed_selections_frozen": True,
        "test_metrics_used_for_selection": False,
        "created_utc": _now(),
    }
    pre_test["fingerprint"] = fingerprint_json(pre_test)
    pre_test_path = run_dir / "pre_test_inventory.json"
    if pre_test_path.is_file():
        existing = read_json(pre_test_path)
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if existing.get("fingerprint") != fingerprint_json(payload):
            raise RuntimeError("existing pre-test inventory fingerprint is invalid")
        for key in (
            "run_contract_sha256", "run_contract_fingerprint", "fit_summary_sha256",
            "selection_count", "selections", "evaluation_source_sha256",
        ):
            if existing.get(key) != pre_test.get(key):
                raise RuntimeError("existing pre-test inventory differs")
        pre_test = existing
    else:
        atomic_write_json(pre_test_path, pre_test, overwrite=False)
    authorization_path = run_dir / "test_gate_authorization.json"
    if authorization_path.is_file():
        authorization = read_json(authorization_path)
        payload = {
            key: value for key, value in authorization.items() if key != "fingerprint"
        }
        if authorization.get("fingerprint") != fingerprint_json(payload):
            raise RuntimeError("test authorization fingerprint is invalid")
        if authorization.get("pre_test_inventory_sha256") != sha256_file(pre_test_path):
            raise RuntimeError("test authorization belongs to another inventory")
    else:
        authorization = {
            "schema_version": 1,
            "kind": "explicit_separate_command_test_authorization",
            "pre_test_inventory_path": str(pre_test_path),
            "pre_test_inventory_sha256": sha256_file(pre_test_path),
            "authorization": "evaluate only immutable legacy/fixed/alternating arms",
            "authorized_utc": _now(),
        }
        authorization["fingerprint"] = fingerprint_json(authorization)
        atomic_write_json(authorization_path, authorization, overwrite=False)
    print(
        "[gate] 25/25 selections immutable; opening each fold only in its test role",
        flush=True,
    )

    reference_folds = {
        int(item["fold"]): item
        for item in reference["results"]["targets"]["L4"]["folds"]
    }
    test_blocks = {fold: load_block(cache, fold) for fold in ALL_FOLDS}
    result_rows: list[dict[str, Any]] = []
    completed_items: list[dict[str, Any]] = []
    for seed in DEFAULT_SEEDS:
        for fold in ALL_FOLDS:
            selection = selections[(seed, fold)]
            item_root = run_dir / "seeds" / f"seed_{seed}" / "folds" / f"fold_{fold:02d}"
            completion_path = item_root / "test_complete.json"
            if completion_path.is_file():
                selection_sha256 = sha256_file(item_root / "selection_frozen.json")
                completion, metrics = _validate_completion(
                    completion_path,
                    contract_fp,
                    expected_seed=seed,
                    expected_fold=fold,
                    expected_selection_sha256=selection_sha256,
                )
                result_rows.append(metrics)
                completed_items.append({
                    "seed": seed,
                    "fold": fold,
                    "completion_path": str(completion_path),
                    "completion_sha256": sha256_file(completion_path),
                    "metrics_path": completion["metrics_path"],
                    "metrics_sha256": completion["metrics_sha256"],
                    "predictions_path": completion["predictions_path"],
                    "predictions_sha256": completion["predictions_sha256"],
                })
                print(f"[seed {seed} fold {fold}] validated test reuse", flush=True)
                continue

            # Loading is deliberately pinned to CPU first, including all tensor state.
            artifact_path = Path(selection["artifact_path"])
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            if artifact.get("run_contract_fingerprint") != contract_fp:
                raise RuntimeError("frozen artifact belongs to another contract")
            if int(artifact.get("seed", -1)) != seed or int(artifact.get("fold", -1)) != fold:
                raise RuntimeError("frozen artifact seed/fold identity changed")
            block = test_blocks[fold]
            neural_scaler = _standardizer(artifact, "neural_scaler")
            neural_pca = _pca(artifact, "neural_pca")
            mel_scaler = _standardizer(artifact, "mel_scaler")
            target_space = TargetSearchSpace(
                _standardizer(artifact, "target_scaler"),
                _pca(artifact, "target_search_pca"),
                OUTPUT_DIM,
            )
            neural_z = neural_scaler.transform(block.neural)
            neural_x = neural_pca.transform(neural_z)
            inputs = np.asarray(
                neural_z.reshape(-1, CONTEXT_STEPS, CHANNELS), dtype=np.float32
            )
            mel_z = mel_scaler.transform(block.mel80)

            legacy_decoder = _affine(artifact, "legacy_decoder")
            legacy_prediction = legacy_decoder.predict(neural_x)
            model = ContextualResidualDecoder(
                context_steps=CONTEXT_STEPS,
                channels=CHANNELS,
                output_dim=OUTPUT_DIM,
            ).to(args.device)
            model.load_state_dict(artifact["fixed_model_state"])
            fixed_prediction = _predict(
                model, inputs, args.device, batch_size=int(contract["batch_size"])
            )
            model.load_state_dict(artifact["alternating_model_state"])
            alternating_prediction = _predict(
                model, inputs, args.device, batch_size=int(contract["batch_size"])
            )
            q0 = np.asarray(artifact["q0"], dtype=np.float64)
            alternating_q = np.asarray(artifact["alternating_q"], dtype=np.float64)
            legacy_metrics, legacy_arrays = _metric_bundle(
                prediction=legacy_prediction, raw_l4=block.l4, mel_z=mel_z,
                target_space=target_space, projector=q0,
                probe=_affine(artifact, "legacy_probe"),
            )
            fixed_metrics, fixed_arrays = _metric_bundle(
                prediction=fixed_prediction, raw_l4=block.l4, mel_z=mel_z,
                target_space=target_space, projector=q0,
                probe=_affine(artifact, "fixed_probe"),
            )
            alternating_metrics, alternating_arrays = _metric_bundle(
                prediction=alternating_prediction, raw_l4=block.l4, mel_z=mel_z,
                target_space=target_space, projector=alternating_q,
                probe=_affine(artifact, "alternating_probe"),
            )
            observed_legacy = _primary(legacy_metrics)
            expected_legacy = float(
                reference_folds[fold]["common_mel80"]["all_bins"]["mean_pearson_r"]
            )
            if abs(observed_legacy - expected_legacy) > 1e-9:
                raise RuntimeError(
                    f"fold {fold} legacy cycle0 does not reproduce latest L4: "
                    f"{observed_legacy:.12g} != {expected_legacy:.12g}"
                )
            fixed_r = _primary(fixed_metrics)
            alternating_r = _primary(alternating_metrics)
            metrics = {
                "seed": seed,
                "fold": fold,
                "train_blocks": selection["train_blocks"],
                "validation_block": selection["validation_block"],
                "test_block": fold,
                "fixed_selected_cycle": selection["fixed_selected_cycle"],
                "alternating_selected_cycle": selection["alternating_selected_cycle"],
                "legacy_cycle0": legacy_metrics,
                "fixed_q_neural": fixed_metrics,
                "alternating_q_neural": alternating_metrics,
                "delta_alternating_minus_fixed_all80": alternating_r - fixed_r,
                "delta_fixed_minus_legacy_all80": fixed_r - observed_legacy,
                "legacy_reference_error": observed_legacy - expected_legacy,
            }
            metrics_path = item_root / "paired_test_metrics.json"
            predictions_path = item_root / "paired_test_predictions.npz"
            _atomic_npz(predictions_path, {
                "sample_ids": block.sample_ids,
                "times": block.times,
                "truth_mel80_z": mel_z,
                "truth_l4": block.l4,
                "legacy_predicted_score50": legacy_arrays["predicted_score50"],
                "legacy_predicted_mel80_z": legacy_arrays["predicted_mel80_z"],
                "fixed_predicted_score50": fixed_arrays["predicted_score50"],
                "fixed_predicted_mel80_z": fixed_arrays["predicted_mel80_z"],
                "alternating_truth_score50": alternating_arrays["truth_score50"],
                "alternating_predicted_score50": alternating_arrays["predicted_score50"],
                "alternating_predicted_mel80_z": alternating_arrays["predicted_mel80_z"],
            })
            _write_or_validate_json(metrics_path, metrics)
            completion = {
                "schema_version": 1,
                "kind": "swpd_sub01_contextual_neural_e2e_fold_seed_test_completion",
                "run_contract_fingerprint": contract_fp,
                "seed": seed,
                "fold": fold,
                "selection_sha256": sha256_file(item_root / "selection_frozen.json"),
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_path": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "completed_utc": _now(),
            }
            completion["fingerprint"] = fingerprint_json(completion)
            atomic_write_json(completion_path, completion, overwrite=False)
            result_rows.append(metrics)
            completed_items.append({
                "seed": seed,
                "fold": fold,
                "completion_path": str(completion_path),
                "completion_sha256": sha256_file(completion_path),
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_path": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
            })
            print(
                f"[seed {seed} fold {fold}] legacy={observed_legacy:.6f} "
                f"fixed={fixed_r:.6f} alternating={alternating_r:.6f} "
                f"primary delta={alternating_r - fixed_r:+.6f}",
                flush=True,
            )

    seed_rows: list[dict[str, Any]] = []
    observed_pairs = [(int(row["seed"]), int(row["fold"])) for row in result_rows]
    expected_pairs = [(seed, fold) for seed in DEFAULT_SEEDS for fold in ALL_FOLDS]
    if sorted(observed_pairs) != sorted(expected_pairs) or len(set(observed_pairs)) != len(
        expected_pairs
    ):
        raise RuntimeError("completed result set does not contain each seed/fold exactly once")
    for seed in DEFAULT_SEEDS:
        rows = [row for row in result_rows if int(row["seed"]) == seed]
        if (
            len(rows) != len(ALL_FOLDS)
            or {int(row["fold"]) for row in rows} != set(ALL_FOLDS)
        ):
            raise RuntimeError(f"seed {seed} does not have five completed folds")
        legacy_all = _fold_mean(rows, "legacy_cycle0", "all_bins")
        fixed_all = _fold_mean(rows, "fixed_q_neural", "all_bins")
        alternating_all = _fold_mean(rows, "alternating_q_neural", "all_bins")
        legacy_low = _fold_mean(rows, "legacy_cycle0", "lower_20_bins")
        fixed_low = _fold_mean(rows, "fixed_q_neural", "lower_20_bins")
        alternating_low = _fold_mean(rows, "alternating_q_neural", "lower_20_bins")
        seed_rows.append({
            "seed": seed,
            "legacy_cycle0_all80_r": legacy_all,
            "fixed_q_neural_all80_r": fixed_all,
            "alternating_q_neural_all80_r": alternating_all,
            "primary_delta_alternating_minus_fixed_all80": alternating_all - fixed_all,
            "neural_delta_fixed_minus_legacy_all80": fixed_all - legacy_all,
            "legacy_cycle0_low20_r": legacy_low,
            "fixed_q_neural_low20_r": fixed_low,
            "alternating_q_neural_low20_r": alternating_low,
            "primary_delta_alternating_minus_fixed_low20": alternating_low - fixed_low,
        })
    legacy_values = [row["legacy_cycle0_all80_r"] for row in seed_rows]
    fixed_values = [row["fixed_q_neural_all80_r"] for row in seed_rows]
    alternating_values = [row["alternating_q_neural_all80_r"] for row in seed_rows]
    delta_values = [
        row["primary_delta_alternating_minus_fixed_all80"] for row in seed_rows
    ]
    legacy_mean = float(np.mean(legacy_values))
    if abs(legacy_mean - EXPECTED_REFERENCE_L4) > 1e-9:
        raise RuntimeError("aggregate legacy cycle0 does not reproduce latest L4")
    summary = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_neural_e2e_summary",
        "development_only": True,
        "interpretation_guard": (
            "Five seeds quantify optimizer stability on one development participant; "
            "they are not five independent patients."
        ),
        "run_contract_fingerprint": contract_fp,
        "primary_contrast": "alternating_q_neural minus fixed_q_neural",
        "fold_seed_results": result_rows,
        "per_seed_fold_means": seed_rows,
        "multiseed": {
            "legacy_cycle0_all80_r": _sample_summary(legacy_values),
            "fixed_q_neural_all80_r": _sample_summary(fixed_values),
            "alternating_q_neural_all80_r": _sample_summary(alternating_values),
            "primary_delta_alternating_minus_fixed_all80": _sample_summary(delta_values),
            "primary_wins": int(sum(value > 0 for value in delta_values)),
        },
        "legacy_reference_r": EXPECTED_REFERENCE_L4,
        "legacy_reference_error": legacy_mean - EXPECTED_REFERENCE_L4,
        "test_evaluated_only_after_all_frozen_selections": True,
        "completed_utc": _now(),
    }
    if existing_summary is not None:
        for key, value in summary.items():
            if key != "completed_utc" and existing_summary.get(key) != value:
                raise RuntimeError("existing summary differs from completed evaluations")
        summary = existing_summary
    else:
        atomic_write_json(summary_path, summary, overwrite=False)
    artifact_manifest = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_neural_e2e_artifact_manifest",
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "pre_test_inventory_path": str(pre_test_path),
        "pre_test_inventory_sha256": sha256_file(pre_test_path),
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "completed_fold_seed_items": completed_items,
        "created_utc": _now(),
    }
    artifact_manifest["fingerprint"] = fingerprint_json(artifact_manifest)
    if completed_manifest is not None:
        for key, value in artifact_manifest.items():
            if key not in ("created_utc", "fingerprint") and completed_manifest.get(key) != value:
                raise RuntimeError("completed artifact manifest inventory changed")
        print(f"ALREADY COMPLETE AND REVALIDATED | {summary_path}", flush=True)
        return 0
    atomic_write_json(manifest_path, artifact_manifest, overwrite=False)
    print("=" * 82, flush=True)
    print("CONTEXTUAL NEURAL E2E TEST COMPLETE", flush=True)
    print(
        f"legacy={np.mean(legacy_values):.6f} | "
        f"fixed={np.mean(fixed_values):.6f} | "
        f"alternating={np.mean(alternating_values):.6f}",
        flush=True,
    )
    print(
        f"primary delta={np.mean(delta_values):+.6f} +/- "
        f"{np.std(delta_values, ddof=1):.6f} SD | "
        f"wins={sum(value > 0 for value in delta_values)}/{len(delta_values)}",
        flush=True,
    )
    print(f"[done] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError, FloatingPointError, OSError, RuntimeError, ValueError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
