#!/usr/bin/env python3
"""Fit and freeze exact contextual alternating candidates on sub-01 train/validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = MODULE_ROOT.parent
sys.path[:0] = [str(EXTERNAL_ROOT), str(EXTERNAL_ROOT / "src")]

from swpd_contextual_alternating_v2.core import (  # noqa: E402
    AffineMap,
    PCATransform,
    Standardizer,
    TargetSearchSpace,
    common_mel_metrics,
    exact_projector_update,
    fit_affine,
    mse,
    project_scores,
    projected_variance_receipt,
)
from swpd_protocol_bridge.bridge_core import component_metrics  # noqa: E402
from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
    fingerprint_json,
    read_json,
    sha256_file,
)


EXPECTED_REFERENCE_SHA256 = "a6c41f7fe65605628adc575e3e02bea3f3db869caef1d89a1c11787ea6e39a2b"
EXPECTED_REFERENCE_L4 = 0.4936210266935884
OUTPUT_DIM = 50


@dataclass(frozen=True)
class Block:
    index: int
    sample_ids: np.ndarray
    times: np.ndarray
    neural: np.ndarray
    mel80: np.ndarray
    l4: np.ndarray


@dataclass(frozen=True)
class Candidate:
    cycle: int
    projector: np.ndarray
    decoder: AffineMap
    probe: AffineMap
    validation_primary_r: float
    validation_common_mel80: Mapping[str, Any]
    validation_target_score50: Mapping[str, Any]
    validation_l4_full512: Mapping[str, Any]
    validation_l4_full512_mse: float
    validation_score_prediction: np.ndarray
    validation_mel_prediction: np.ndarray


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rows(blocks: Mapping[int, Block], indexes: Sequence[int], field: str) -> np.ndarray:
    return np.concatenate([getattr(blocks[index], field) for index in indexes], axis=0)


def load_block(cache: Path, index: int) -> Block:
    manifest_path = cache / f"block_{index:02d}.json"
    manifest = read_json(manifest_path)
    stored = manifest.pop("fingerprint", None)
    if stored != fingerprint_json(manifest):
        raise RuntimeError(f"contextual cache manifest fingerprint changed: block {index}")
    if manifest.get("kind") != "swpd_sub01_contextual_whisper_block_cache":
        raise RuntimeError("contextual cache kind changed")
    if manifest.get("subject") != "sub-01" or int(manifest.get("block", -1)) != index:
        raise RuntimeError("contextual cache subject/block changed")
    if manifest.get("confirmatory_subjects_read") is not False:
        raise RuntimeError("development cache provenance changed")
    arrays_path = cache / str(manifest["arrays_file"])
    if sha256_file(arrays_path) != manifest["arrays_sha256"]:
        raise RuntimeError(f"contextual cache arrays changed: block {index}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        result = Block(
            index=index,
            sample_ids=np.asarray(archive["sample_ids"]),
            times=np.asarray(archive["frame_times_seconds"], dtype=np.float64),
            neural=np.asarray(archive["neural_context"], dtype=np.float64),
            mel80=np.asarray(archive["mel80"], dtype=np.float64),
            l4=np.asarray(archive["L4"], dtype=np.float64),
        )
    rows = len(result.sample_ids)
    if result.neural.shape != (rows, 1143) or result.mel80.shape != (rows, 80):
        raise RuntimeError("contextual neural/MEL geometry changed")
    if result.l4.shape != (rows, 512) or np.unique(result.sample_ids).size != rows:
        raise RuntimeError("contextual L4/sample-ID geometry changed")
    if any(not np.isfinite(item).all() for item in (
        result.times, result.neural, result.mel80, result.l4
    )):
        raise RuntimeError("contextual cache contains non-finite values")
    if np.any(np.diff(result.times) <= 0):
        raise RuntimeError("contextual cache timeline is not strictly increasing")
    return result


def _primary(metrics: Mapping[str, Any]) -> float:
    return float(metrics["all_bins"]["mean_pearson_r"])


def _evaluate_candidate(
    cycle: int,
    projector: np.ndarray,
    decoder: AffineMap,
    target_space: TargetSearchSpace,
    train_whitened: np.ndarray,
    val_whitened: np.ndarray,
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_mel_z: np.ndarray,
    val_mel_z: np.ndarray,
    val_l4_z: np.ndarray,
) -> Candidate:
    train_scores = project_scores(train_whitened, projector)
    true_val_scores = project_scores(val_whitened, projector)
    probe = fit_affine(train_scores, train_mel_z)
    predicted_val_scores = decoder.predict(val_x)
    predicted_val_mel = probe.predict(predicted_val_scores)
    common = component_metrics(val_mel_z, predicted_val_mel)
    target_metrics = component_metrics(true_val_scores, predicted_val_scores)
    reconstructed_l4 = target_space.reconstruct_standardized(
        predicted_val_scores, projector
    )
    full_metrics = component_metrics(val_l4_z, reconstructed_l4)
    return Candidate(
        cycle=cycle,
        projector=np.array(projector, copy=True),
        decoder=decoder,
        probe=probe,
        validation_primary_r=_primary(common),
        validation_common_mel80=common,
        validation_target_score50=target_metrics,
        validation_l4_full512=full_metrics,
        validation_l4_full512_mse=mse(val_l4_z, reconstructed_l4),
        validation_score_prediction=predicted_val_scores,
        validation_mel_prediction=predicted_val_mel,
    )


def _put_standardizer(arrays: dict[str, np.ndarray], prefix: str, value: Standardizer) -> None:
    arrays[f"{prefix}_mean"] = value.mean
    arrays[f"{prefix}_scale"] = value.scale


def _put_pca(arrays: dict[str, np.ndarray], prefix: str, value: PCATransform) -> None:
    arrays[f"{prefix}_mean"] = value.mean
    arrays[f"{prefix}_components"] = value.components
    arrays[f"{prefix}_explained_variance"] = value.explained_variance
    arrays[f"{prefix}_whiten"] = np.asarray([int(value.whiten)], dtype=np.int8)


def _put_affine(arrays: dict[str, np.ndarray], prefix: str, value: AffineMap) -> None:
    arrays[f"{prefix}_coef"] = value.coef
    arrays[f"{prefix}_intercept"] = value.intercept


def _fit_fold(
    fold: int,
    cache: Path,
    fold_root: Path,
    contract_fingerprint: str,
    search_dim: int,
    max_cycles: int,
) -> dict[str, Any]:
    selection_path = fold_root / "selection_frozen.json"
    if selection_path.is_file():
        selection = read_json(selection_path)
        stored = selection.get("fingerprint")
        payload = {key: value for key, value in selection.items() if key != "fingerprint"}
        if stored != fingerprint_json(payload):
            raise RuntimeError(f"fold {fold} selection fingerprint is invalid")
        if selection.get("run_contract_fingerprint") != contract_fingerprint:
            raise RuntimeError(f"fold {fold} selection belongs to another contract")
        for path_key, hash_key in (
            ("artifact_path", "artifact_sha256"),
            ("validation_predictions_path", "validation_predictions_sha256"),
        ):
            artifact = Path(selection[path_key])
            if not artifact.is_file() or sha256_file(artifact) != selection[hash_key]:
                raise RuntimeError(f"fold {fold} frozen selection artifact changed")
        print(f"[fold {fold}] validated frozen selection reused", flush=True)
        return selection

    validation = (fold + 1) % 5
    train = tuple(index for index in range(5) if index not in (fold, validation))
    role_indexes = train + (validation,)
    blocks = {index: load_block(cache, index) for index in role_indexes}
    print(
        f"[fold {fold}] train={list(train)} validation={validation} test={fold} remains role-excluded",
        flush=True,
    )
    train_ids = _rows(blocks, train, "sample_ids")
    val_ids = _rows(blocks, (validation,), "sample_ids")
    if np.intersect1d(train_ids, val_ids).size:
        raise RuntimeError("train/validation IDs overlap")

    train_neural = _rows(blocks, train, "neural")
    val_neural = _rows(blocks, (validation,), "neural")
    neural_scaler = Standardizer.fit(train_neural)
    neural_pca = PCATransform.fit(
        neural_scaler.transform(train_neural), OUTPUT_DIM, whiten=False
    )
    train_x = neural_pca.transform(neural_scaler.transform(train_neural))
    val_x = neural_pca.transform(neural_scaler.transform(val_neural))

    train_mel = _rows(blocks, train, "mel80")
    val_mel = _rows(blocks, (validation,), "mel80")
    mel_scaler = Standardizer.fit(train_mel)
    train_mel_z = mel_scaler.transform(train_mel)
    val_mel_z = mel_scaler.transform(val_mel)

    train_l4 = _rows(blocks, train, "l4")
    val_l4 = _rows(blocks, (validation,), "l4")
    target_space = TargetSearchSpace.fit(
        train_l4, search_dim=search_dim, output_dim=OUTPUT_DIM
    )
    train_whitened = target_space.transform(train_l4)
    val_whitened = target_space.transform(val_l4)
    val_l4_z = target_space.scaler.transform(val_l4)

    # Cycle zero must be the exact target transform used by the latest frozen protocol.
    legacy_pca50 = PCATransform.fit(
        target_space.scaler.transform(train_l4), OUTPUT_DIM, whiten=True
    )
    legacy_train = legacy_pca50.transform(target_space.scaler.transform(train_l4))
    parity_error = float(np.max(np.abs(legacy_train - train_whitened[:, :OUTPUT_DIM])))
    if parity_error > 1e-9:
        raise RuntimeError(f"cycle-zero PCA50 parity failed: {parity_error:.3e}")

    projector = target_space.initial_projector()
    control_scores = project_scores(train_whitened, projector)
    control_decoder = fit_affine(train_x, control_scores)
    control = _evaluate_candidate(
        0, projector, control_decoder, target_space,
        train_whitened, val_whitened, train_x, val_x,
        train_mel_z, val_mel_z, val_l4_z,
    )
    candidates = [control]
    history: list[dict[str, Any]] = [{
        "cycle": 0,
        "kind": "pca50_fixed_control",
        "train_mse": mse(control_scores, control_decoder.predict(train_x)),
        "validation_primary_r": control.validation_primary_r,
        "projected_train_variance": projected_variance_receipt(control_scores),
    }]
    current_decoder = control_decoder

    for cycle in range(1, max_cycles + 1):
        current_targets = project_scores(train_whitened, projector)
        phase_a_before = mse(current_targets, current_decoder.predict(train_x))
        current_decoder = fit_affine(train_x, current_targets)
        phase_a_after = mse(current_targets, current_decoder.predict(train_x))
        if phase_a_after > phase_a_before + 1e-10 * max(1.0, phase_a_before):
            raise RuntimeError("exact OLS phase increased the common train objective")
        fixed_prediction = current_decoder.predict(train_x)
        updated_projector, phase_b = exact_projector_update(
            train_whitened, fixed_prediction, projector
        )
        projector = updated_projector
        candidate = _evaluate_candidate(
            cycle, projector, current_decoder, target_space,
            train_whitened, val_whitened, train_x, val_x,
            train_mel_z, val_mel_z, val_l4_z,
        )
        candidates.append(candidate)
        history.append({
            "cycle": cycle,
            "kind": "exact_ols_then_exact_whitened_procrustes",
            "phase_a_old_mse": phase_a_before,
            "phase_a_new_mse": phase_a_after,
            "phase_b": phase_b,
            "endpoint_train_mse": mse(
                project_scores(train_whitened, projector), fixed_prediction
            ),
            "validation_primary_r": candidate.validation_primary_r,
            "projected_train_variance": projected_variance_receipt(
                project_scores(train_whitened, projector)
            ),
        })
        print(
            f"[fold {fold} cycle {cycle}] train={history[-1]['endpoint_train_mse']:.6f} "
            f"valMEL-r={candidate.validation_primary_r:.6f}",
            flush=True,
        )

    best = max(candidates, key=lambda item: (item.validation_primary_r, -item.cycle))
    arrays: dict[str, np.ndarray] = {
        "selected_projector": best.projector,
        "control_projector": control.projector,
    }
    _put_standardizer(arrays, "neural_scaler", neural_scaler)
    _put_pca(arrays, "neural_pca", neural_pca)
    _put_standardizer(arrays, "mel_scaler", mel_scaler)
    _put_standardizer(arrays, "target_scaler", target_space.scaler)
    _put_pca(arrays, "target_search_pca", target_space.pca)
    _put_affine(arrays, "selected_decoder", best.decoder)
    _put_affine(arrays, "selected_probe", best.probe)
    _put_affine(arrays, "control_decoder", control.decoder)
    _put_affine(arrays, "control_probe", control.probe)
    artifact_path = fold_root / "selected_artifact.npz"
    predictions_path = fold_root / "validation_predictions.npz"
    _atomic_npz(artifact_path, arrays)
    _atomic_npz(predictions_path, {
        "sample_ids": val_ids,
        "times": _rows(blocks, (validation,), "times"),
        "truth_mel80_z": val_mel_z,
        "selected_score_prediction": best.validation_score_prediction,
        "selected_mel80_prediction_z": best.validation_mel_prediction,
        "control_score_prediction": control.validation_score_prediction,
        "control_mel80_prediction_z": control.validation_mel_prediction,
    })
    selection: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_covariance_alternating_v2_fold_selection",
        "run_contract_fingerprint": contract_fingerprint,
        "fold": fold,
        "train_blocks": list(train),
        "validation_block": validation,
        "test_block": fold,
        "train_count": int(len(train_ids)),
        "validation_count": int(len(val_ids)),
        "train_ids_sha256": fingerprint_json(train_ids.tolist()),
        "validation_ids_sha256": fingerprint_json(val_ids.tolist()),
        "cycle_zero_pca50_max_abs_error": parity_error,
        "fixed_control_validation_primary_r": control.validation_primary_r,
        "selected_cycle": best.cycle,
        "selected_validation_primary_r": best.validation_primary_r,
        "delta_selected_minus_fixed_control": (
            best.validation_primary_r - control.validation_primary_r
        ),
        "selected_validation_common_mel80": best.validation_common_mel80,
        "selected_validation_target_score50": best.validation_target_score50,
        "selected_validation_l4_full512": best.validation_l4_full512,
        "selected_validation_l4_full512_mse": best.validation_l4_full512_mse,
        "history": history,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "validation_predictions_path": str(predictions_path),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "test_evaluated": False,
        "frozen_utc": _now(),
    }
    selection["fingerprint"] = fingerprint_json(selection)
    atomic_write_json(selection_path, selection, overwrite=False)
    print(
        f"[fold {fold}] frozen cycle={best.cycle} r={best.validation_primary_r:.6f} "
        f"delta={selection['delta_selected_minus_fixed_control']:+.6f}",
        flush=True,
    )
    return selection


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--search-dim", type=int, default=128)
    parser.add_argument("--max-cycles", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cache = args.cache_dir.expanduser().resolve()
    reference = args.reference_summary.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if args.search_dim < OUTPUT_DIM or args.max_cycles < 1:
        raise ValueError("search-dim/cycles are invalid")
    if sha256_file(reference) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("latest contextual sub-01 reference changed")
    reference_payload = read_json(reference)
    reference_l4 = float(
        reference_payload["results"]["targets"]["L4"]["aggregate_common_mel80"]["all_bins"]["mean"]
    )
    if abs(reference_l4 - EXPECTED_REFERENCE_L4) > 1e-12:
        raise RuntimeError("latest contextual L4 reference value changed")
    manifests = {}
    declared_arrays = {}
    for index in range(5):
        path = cache / f"block_{index:02d}.json"
        payload = read_json(path)
        manifests[f"block_{index}"] = sha256_file(path)
        declared_arrays[f"block_{index}"] = payload["arrays_sha256"]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_covariance_alternating_v2_contract",
        "development_only": True,
        "base": "latest contextual L4 protocol sent to Ossadtchi",
        "reference_summary": str(reference),
        "reference_summary_sha256": EXPECTED_REFERENCE_SHA256,
        "reference_l4_common_mel80_r": EXPECTED_REFERENCE_L4,
        "cache_dir": str(cache),
        "cache_manifest_sha256": manifests,
        "cache_declared_arrays_sha256": declared_arrays,
        "architecture": {
            "neural_input": "high-gamma context -200..+200 ms, 20 ms grid, 1143D",
            "neural_transform": "fold-train StandardScaler then PCA50 whiten=false",
            "decoder": "ordinary least squares",
            "target": "Whisper-base L4 raw512",
            "target_search": f"fold-train StandardScaler then whitened PCA{args.search_dim}",
            "projector": f"row-orthonormal {args.search_dim}->50",
            "common_surface": "train-only affine score50->standardized MEL80",
        },
        "split": "five folds; test i, validation i+1 cyclic, remaining three train",
        "inner_objective": "shared train MSE between OLS prediction and covariance-white projected L4",
        "outer_selection": "validation mean Pearson r over common MEL80 all 80 bins",
        "fixed_control": "cycle-zero exact L4 whitened PCA50 plus OLS",
        "search_dim": args.search_dim,
        "output_dim": OUTPUT_DIM,
        "max_cycles": args.max_cycles,
        "test_evaluation_in_this_command": False,
        "implementation_sha256": {
            "fit_runner": sha256_file(Path(__file__)),
            "core": sha256_file(MODULE_ROOT / "core.py"),
            "evaluate_runner": sha256_file(MODULE_ROOT / "evaluate_frozen_sub01.py"),
            "preflight": sha256_file(MODULE_ROOT / "preflight.py"),
            "frozen_core": sha256_file(EXTERNAL_ROOT / "swpd_contextual_frozen" / "core.py"),
            "bridge_core": sha256_file(EXTERNAL_ROOT / "swpd_protocol_bridge" / "bridge_core.py"),
            "integrity": sha256_file(EXTERNAL_ROOT / "src" / "whisper_ecog_ext" / "integrity.py"),
            "run_fit_ps1": sha256_file(MODULE_ROOT / "scripts" / "run_fit.ps1"),
            "start_fit_background_ps1": sha256_file(MODULE_ROOT / "scripts" / "start_fit_background.ps1"),
            "watch_fit_ps1": sha256_file(MODULE_ROOT / "scripts" / "watch_fit.ps1"),
            "run_evaluate_frozen_ps1": sha256_file(MODULE_ROOT / "scripts" / "run_evaluate_frozen.ps1"),
        },
        "created_utc": _now(),
    }
    # Time is provenance, not part of resume compatibility.
    compatibility = {key: value for key, value in contract.items() if key != "created_utc"}
    contract["compatibility_fingerprint"] = fingerprint_json(compatibility)
    contract_path = run_dir / "run_contract.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        existing = read_json(contract_path)
        existing_compatibility = {
            key: value for key, value in existing.items()
            if key not in ("created_utc", "compatibility_fingerprint")
        }
        if existing.get("compatibility_fingerprint") != fingerprint_json(existing_compatibility):
            raise RuntimeError("existing v2 contract is invalid")
        if existing["compatibility_fingerprint"] != contract["compatibility_fingerprint"]:
            raise RuntimeError("existing v2 contract differs from current code/protocol")
        contract = existing
    else:
        atomic_write_json(contract_path, contract, overwrite=False)
    contract_fp = str(contract["compatibility_fingerprint"])
    selections = []
    for fold in range(5):
        selections.append(_fit_fold(
            fold, cache, run_dir / "folds" / f"fold_{fold:02d}",
            contract_fp, args.search_dim, args.max_cycles,
        ))
    summary = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_covariance_alternating_v2_fit_summary",
        "run_contract_fingerprint": contract_fp,
        "folds": selections,
        "mean_fixed_control_validation_r": float(np.mean([
            item["fixed_control_validation_primary_r"] for item in selections
        ])),
        "mean_selected_validation_r": float(np.mean([
            item["selected_validation_primary_r"] for item in selections
        ])),
        "mean_validation_delta": float(np.mean([
            item["delta_selected_minus_fixed_control"] for item in selections
        ])),
        "selected_cycles": [int(item["selected_cycle"]) for item in selections],
        "test_evaluated": False,
        "completed_utc": _now(),
    }
    atomic_write_json(run_dir / "fit_summary.json", summary)
    print(
        f"FIT COMPLETE | validation fixed={summary['mean_fixed_control_validation_r']:.6f} "
        f"selected={summary['mean_selected_validation_r']:.6f} "
        f"delta={summary['mean_validation_delta']:+.6f} | TEST NOT EVALUATED",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
