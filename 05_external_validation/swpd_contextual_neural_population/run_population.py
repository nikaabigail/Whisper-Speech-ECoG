#!/usr/bin/env python3
"""Fit or evaluate the frozen fixed-Q neural decoder on SWPD sub-02..sub-09."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


MODULE_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = MODULE_ROOT.parent
sys.path[:0] = [str(EXTERNAL_ROOT), str(EXTERNAL_ROOT / "src")]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from swpd_contextual_alternating_v2.core import (  # noqa: E402
    AffineMap, PCATransform, Standardizer, TargetSearchSpace,
    fit_affine, mse, project_scores,
)
from swpd_contextual_neural_e2e.core import (  # noqa: E402
    ContextualResidualDecoder, fold_legacy_pipeline, state_dict_sha256,
)
from swpd_contextual_neural_e2e.fit_select_sub01 import (  # noqa: E402
    _atomic_torch_save, _cpu_state, _predict, _restore_rng_state, _rng_state,
    _set_seed, _train_phase,
)
from swpd_protocol_bridge.bridge_core import component_metrics  # noqa: E402
from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json, fingerprint_json, read_json, sha256_file,
)


SUBJECTS = tuple(f"sub-{number:02d}" for number in range(2, 10))
SEEDS = (1, 2, 3, 4, 42)
FOLDS = (0, 1, 2, 3, 4)
CONTEXT_STEPS = 9
SEARCH_DIM = 128
OUTPUT_DIM = 50
MAX_CYCLES = 5
EPOCHS_PER_CYCLE = 10
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SUB01_SUMMARY_SHA256 = "664b7bb788849c7c38a72fa162f76a71121d373b2d1886f4a56294e59f01f6b6"
LEGACY_POPULATION_SHA256 = "a732d822d131cf83fe3eb3a451b4a66388c4c59a11ca963096b2d6e5623659a3"


@dataclass(frozen=True)
class Block:
    subject: str
    index: int
    sample_ids: np.ndarray
    times: np.ndarray
    neural: np.ndarray
    mel80: np.ndarray
    l4: np.ndarray


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _csv(text: str, cast: Any = str) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV values must be non-empty and unique")
    return values


def _rows(blocks: Mapping[int, Block], indexes: Sequence[int], field: str) -> np.ndarray:
    return np.concatenate([getattr(blocks[index], field) for index in indexes], axis=0)


def load_block(cache_root: Path, subject: str, fold: int) -> Block:
    root = cache_root / subject
    manifest_path = root / f"block_{fold:02d}.json"
    manifest = read_json(manifest_path)
    stored = manifest.get("fingerprint")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if stored != fingerprint_json(payload):
        raise RuntimeError(f"cache manifest fingerprint changed: {subject} fold {fold}")
    if manifest.get("subject") != subject or int(manifest.get("block", -1)) != fold:
        raise RuntimeError("cache subject/fold identity changed")
    arrays_path = root / str(manifest["arrays_file"])
    if sha256_file(arrays_path) != manifest.get("arrays_sha256"):
        raise RuntimeError(f"cache arrays changed: {subject} fold {fold}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        block = Block(
            subject, fold, np.asarray(archive["sample_ids"]),
            np.asarray(archive["times"], dtype=np.float64),
            np.asarray(archive["neural"], dtype=np.float64),
            np.asarray(archive["mel80"], dtype=np.float64),
            np.asarray(archive["L4"], dtype=np.float64),
        )
    rows = len(block.sample_ids)
    if (
        block.neural.shape[0] != rows or block.neural.shape[1] % CONTEXT_STEPS
        or block.mel80.shape != (rows, 80) or block.l4.shape != (rows, 512)
        or np.unique(block.sample_ids).size != rows or np.any(np.diff(block.times) <= 0)
        or any(not value.all() for value in map(np.isfinite, (block.times, block.neural, block.mel80, block.l4)))
    ):
        raise RuntimeError(f"invalid cache geometry/timeline: {subject} fold {fold}")
    return block


def _put_standardizer(payload: dict[str, Any], prefix: str, value: Standardizer) -> None:
    payload[f"{prefix}_mean"] = np.asarray(value.mean)
    payload[f"{prefix}_scale"] = np.asarray(value.scale)


def _put_pca(payload: dict[str, Any], prefix: str, value: PCATransform) -> None:
    payload[f"{prefix}_mean"] = np.asarray(value.mean)
    payload[f"{prefix}_components"] = np.asarray(value.components)
    payload[f"{prefix}_explained_variance"] = np.asarray(value.explained_variance)
    payload[f"{prefix}_whiten"] = bool(value.whiten)


def _put_affine(payload: dict[str, Any], prefix: str, value: AffineMap) -> None:
    payload[f"{prefix}_coef"] = np.asarray(value.coef)
    payload[f"{prefix}_intercept"] = np.asarray(value.intercept)


def _standardizer(payload: Mapping[str, Any], prefix: str) -> Standardizer:
    return Standardizer(np.asarray(payload[f"{prefix}_mean"]), np.asarray(payload[f"{prefix}_scale"]))


def _pca(payload: Mapping[str, Any], prefix: str) -> PCATransform:
    return PCATransform(
        np.asarray(payload[f"{prefix}_mean"]), np.asarray(payload[f"{prefix}_components"]),
        np.asarray(payload[f"{prefix}_explained_variance"]), bool(payload[f"{prefix}_whiten"]),
    )


def _affine(payload: Mapping[str, Any], prefix: str) -> AffineMap:
    return AffineMap(np.asarray(payload[f"{prefix}_coef"]), np.asarray(payload[f"{prefix}_intercept"]))


def _primary(metrics: Mapping[str, Any]) -> float:
    if int(metrics["all_bins"].get("component_count", -1)) != 80:
        raise RuntimeError("primary metric must contain 80 MEL bins")
    return float(metrics["all_bins"]["mean_pearson_r"])


def _endpoint(
    prediction: np.ndarray, truth_l4: np.ndarray, truth_mel_z: np.ndarray,
    target_space: TargetSearchSpace, q0: np.ndarray, probe: AffineMap,
) -> dict[str, Any]:
    truth_scores = project_scores(target_space.transform(truth_l4), q0)
    predicted_mel = probe.predict(prediction)
    predicted_l4 = target_space.reconstruct_standardized(prediction, q0)
    l4_z = target_space.scaler.transform(truth_l4)
    common = component_metrics(truth_mel_z, predicted_mel)
    return {
        "common_mel80": common, "primary_r": _primary(common),
        "target_score50": component_metrics(truth_scores, prediction),
        "l4_full512": component_metrics(l4_z, predicted_l4),
        "l4_full512_mse": mse(l4_z, predicted_l4),
    }


def _selection(path: Path, contract_fp: str, subject: str, seed: int, fold: int) -> dict[str, Any]:
    item = read_json(path)
    payload = {key: value for key, value in item.items() if key != "fingerprint"}
    if item.get("fingerprint") != fingerprint_json(payload):
        raise RuntimeError("selection fingerprint changed")
    if (
        item.get("run_contract_fingerprint") != contract_fp
        or item.get("subject") != subject or int(item.get("seed", -1)) != seed
        or int(item.get("fold", -1)) != fold or item.get("test_evaluated") is not False
    ):
        raise RuntimeError("selection identity/contract changed")
    artifact = Path(item["artifact_path"])
    if not artifact.is_file() or sha256_file(artifact) != item["artifact_sha256"]:
        raise RuntimeError("selection artifact changed")
    return item


def _fit_one(
    *, subject: str, seed: int, fold: int, cache_root: Path, run_root: Path,
    contract_fp: str, device: str, max_cycles: int, epochs: int, batch_size: int,
    max_train_batches: int | None, max_eval_batches: int | None, diagnostic: bool,
) -> dict[str, Any]:
    import torch

    root = run_root / "subjects" / subject / "seeds" / f"seed_{seed}" / "folds" / f"fold_{fold:02d}"
    selection_path = root / "selection_frozen.json"
    if selection_path.is_file():
        return _selection(selection_path, contract_fp, subject, seed, fold)
    validation = (fold + 1) % 5
    train = tuple(item for item in FOLDS if item not in (fold, validation))
    blocks = {item: load_block(cache_root, subject, item) for item in train + (validation,)}
    train_ids, val_ids = _rows(blocks, train, "sample_ids"), _rows(blocks, (validation,), "sample_ids")
    if np.intersect1d(train_ids, val_ids).size:
        raise RuntimeError("train/validation IDs overlap")
    dims = {block.neural.shape[1] for block in blocks.values()}
    if len(dims) != 1:
        raise RuntimeError("channel count differs between blocks")
    channels = dims.pop() // CONTEXT_STEPS
    print(f"[{subject} seed {seed} fold {fold}] train={list(train)} val={validation} test={fold} excluded", flush=True)

    train_neural, val_neural = _rows(blocks, train, "neural"), _rows(blocks, (validation,), "neural")
    neural_scaler = Standardizer.fit(train_neural)
    train_z, val_z = neural_scaler.transform(train_neural), neural_scaler.transform(val_neural)
    neural_pca = PCATransform.fit(train_z, OUTPUT_DIM, whiten=False)
    train_x, val_x = neural_pca.transform(train_z), neural_pca.transform(val_z)
    train_inputs = np.asarray(train_z.reshape(-1, CONTEXT_STEPS, channels), dtype=np.float32)
    val_inputs = np.asarray(val_z.reshape(-1, CONTEXT_STEPS, channels), dtype=np.float32)

    train_l4, val_l4 = _rows(blocks, train, "l4"), _rows(blocks, (validation,), "l4")
    target_space = TargetSearchSpace.fit(
        train_l4, search_dim=SEARCH_DIM, output_dim=OUTPUT_DIM
    )
    q0 = target_space.initial_projector()
    train_scores = project_scores(target_space.transform(train_l4), q0)
    legacy_pca = PCATransform.fit(target_space.scaler.transform(train_l4), OUTPUT_DIM, whiten=True)
    target_parity = float(np.max(np.abs(legacy_pca.transform(target_space.scaler.transform(train_l4)) - train_scores)))
    if target_parity > 1e-9:
        raise RuntimeError("target PCA parity failed")
    train_mel, val_mel = _rows(blocks, train, "mel80"), _rows(blocks, (validation,), "mel80")
    mel_scaler = Standardizer.fit(train_mel)
    train_mel_z, val_mel_z = mel_scaler.transform(train_mel), mel_scaler.transform(val_mel)
    legacy_decoder = fit_affine(train_x, train_scores)
    legacy_probe = fit_affine(train_scores, train_mel_z)
    legacy_prediction = legacy_decoder.predict(val_x)
    legacy_metrics = _endpoint(legacy_prediction, val_l4, val_mel_z, target_space, q0, legacy_probe)

    _set_seed(seed)
    model = ContextualResidualDecoder(CONTEXT_STEPS, channels, OUTPUT_DIM).to(device)
    folded = fold_legacy_pipeline(neural_pca, legacy_decoder)
    model.initialize_legacy_skip(folded.coef, folded.intercept)
    parity = float(np.max(np.abs(_predict(model, val_inputs, device, batch_size) - legacy_prediction)))
    if parity > 5e-5:
        raise RuntimeError("legacy neural skip parity failed")
    legacy_state = _cpu_state(model)
    checkpoint_path = root / "fit_checkpoint.pt"
    current: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    next_cycle = 1
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("run_contract_fingerprint") != contract_fp
            or checkpoint.get("subject") != subject or int(checkpoint.get("seed", -1)) != seed
            or int(checkpoint.get("fold", -1)) != fold
        ):
            raise RuntimeError("resume checkpoint identity changed")
        current, history, best = checkpoint["current_state"], list(checkpoint["history"]), checkpoint["best"]
        next_cycle = int(checkpoint["next_cycle"])
        if next_cycle != len(history) + 1 or not 1 <= next_cycle <= max_cycles + 1:
            raise RuntimeError("resume checkpoint cycle history changed")
        _restore_rng_state(checkpoint["rng_state"])
        print(f"[{subject} seed {seed} fold {fold}] resume cycle={next_cycle}", flush=True)
    for cycle in range(next_cycle, max_cycles + 1):
        model.load_state_dict(legacy_state if cycle == 1 else current)
        current, epoch_rows = _train_phase(
            model, train_inputs, train_scores, device=device, epochs=epochs,
            batch_size=batch_size, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            grad_clip=GRAD_CLIP, phase_seed=seed * 100_000 + fold * 1_000 + cycle * 10,
            max_train_batches=max_train_batches, max_eval_batches=max_eval_batches,
            label=f"{subject} seed {seed} fold {fold} fixed cycle {cycle}",
        )
        model.load_state_dict(current)
        train_prediction = _predict(model, train_inputs, device, batch_size, max_eval_batches)
        probe_rows = len(train_prediction)
        probe = fit_affine(train_scores[:probe_rows], train_mel_z[:probe_rows])
        val_prediction = _predict(model, val_inputs, device, batch_size, max_eval_batches)
        metrics = _endpoint(
            val_prediction, val_l4[:len(val_prediction)], val_mel_z[:len(val_prediction)],
            target_space, q0, probe,
        )
        history.append({"cycle": cycle, "epochs": epoch_rows, "validation": metrics})
        if best is None or metrics["primary_r"] > best["score"]:
            best = {"cycle": cycle, "score": metrics["primary_r"], "state": deepcopy(current), "probe": probe, "metrics": metrics}
        print(f"[{subject} seed {seed} fold {fold} cycle {cycle}] legacy={legacy_metrics['primary_r']:.6f} fixed={metrics['primary_r']:.6f}", flush=True)
        _atomic_torch_save(checkpoint_path, {
            "schema_version": 1, "run_contract_fingerprint": contract_fp,
            "subject": subject, "seed": seed, "fold": fold, "next_cycle": cycle + 1,
            "current_state": current, "history": history, "best": best, "rng_state": _rng_state(),
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("no fixed neural cycle selected")
    artifact: dict[str, Any] = {
        "schema_version": 1, "kind": "swpd_contextual_fixed_q_neural_population_artifact",
        "run_contract_fingerprint": contract_fp, "subject": subject, "seed": seed, "fold": fold,
        "channels": channels, "train_blocks": list(train), "validation_block": validation,
        "test_block": fold, "architecture": model.architecture_receipt(), "q0": q0,
        "legacy_model_state_sha256": state_dict_sha256(legacy_state),
        "fixed_model_state": best["state"], "fixed_selected_cycle": int(best["cycle"]),
        "target_pca_parity": target_parity, "legacy_skip_parity": parity,
    }
    _put_standardizer(artifact, "neural_scaler", neural_scaler); _put_pca(artifact, "neural_pca", neural_pca)
    _put_standardizer(artifact, "mel_scaler", mel_scaler); _put_standardizer(artifact, "target_scaler", target_space.scaler)
    _put_pca(artifact, "target_search_pca", target_space.pca); _put_affine(artifact, "legacy_decoder", legacy_decoder)
    _put_affine(artifact, "legacy_probe", legacy_probe); _put_affine(artifact, "fixed_probe", best["probe"])
    artifact_path = root / "frozen_artifact.pt"
    _atomic_torch_save(artifact_path, artifact)
    selection = {
        "schema_version": 1, "kind": "swpd_contextual_fixed_q_neural_population_selection",
        "run_contract_fingerprint": contract_fp, "subject": subject, "seed": seed, "fold": fold,
        "diagnostic": diagnostic, "train_blocks": list(train), "validation_block": validation,
        "test_block": fold, "channels": channels, "train_count": len(train_ids), "validation_count": len(val_ids),
        "train_ids_sha256": fingerprint_json(train_ids.tolist()), "validation_ids_sha256": fingerprint_json(val_ids.tolist()),
        "legacy_validation": legacy_metrics, "fixed_selected_cycle": int(best["cycle"]),
        "fixed_selected_validation": best["metrics"], "fixed_history": history,
        "artifact_path": str(artifact_path), "artifact_sha256": sha256_file(artifact_path),
        "test_evaluated": False, "frozen_utc": _now(),
    }
    selection["fingerprint"] = fingerprint_json(selection)
    atomic_write_json(selection_path, selection, overwrite=False)
    print(f"[{subject} seed {seed} fold {fold}] FROZEN cycle={best['cycle']} r={best['score']:.6f}", flush=True)
    return selection


def _sources() -> dict[str, str]:
    return {
        "runner": sha256_file(Path(__file__)),
        "neural_core": sha256_file(EXTERNAL_ROOT / "swpd_contextual_neural_e2e" / "core.py"),
        "neural_fit_dependency": sha256_file(EXTERNAL_ROOT / "swpd_contextual_neural_e2e" / "fit_select_sub01.py"),
        "linear_core": sha256_file(EXTERNAL_ROOT / "swpd_contextual_alternating_v2" / "core.py"),
        "legacy_population_reference": sha256_file(
            EXTERNAL_ROOT / "swpd_contextual_frozen" / "results" / "population_summary.json"
        ),
    }


def _cache_receipts(cache_root: Path, subjects: Sequence[str], folds: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for subject in subjects:
        for fold in folds:
            path = cache_root / subject / f"block_{fold:02d}.json"
            manifest = read_json(path)
            result[f"{subject}/block_{fold}"] = {
                "manifest_sha256": sha256_file(path), "arrays_sha256": manifest["arrays_sha256"],
            }
    return result


def fit(args: argparse.Namespace) -> int:
    import torch

    subjects, seeds, folds = _csv(args.subjects), _csv(args.seeds, int), _csv(args.folds, int)
    if any(item not in SUBJECTS for item in subjects) or any(item not in FOLDS for item in folds):
        raise ValueError("subject/fold list is outside the frozen cohort")
    diagnostic = bool(
        args.diagnostic or subjects != SUBJECTS or seeds != SEEDS or folds != FOLDS
        or args.device != "cuda" or args.max_cycles != MAX_CYCLES or args.epochs != EPOCHS_PER_CYCLE
        or args.batch_size != BATCH_SIZE or args.max_train_batches is not None or args.max_eval_batches is not None
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache_root, run_root = args.cache_root.resolve(), args.run_root.resolve()
    development_summary = args.development_summary.resolve()
    if sha256_file(development_summary) != SUB01_SUMMARY_SHA256:
        raise RuntimeError("frozen sub-01 neural development summary changed")
    development = read_json(development_summary)
    if _sources()["legacy_population_reference"] != LEGACY_POPULATION_SHA256:
        raise RuntimeError("legacy contextual population reference changed")
    fixed_r = float(development["multiseed"]["fixed_q_neural_all80_r"]["mean"])
    alternating_r = float(development["multiseed"]["alternating_q_neural_all80_r"]["mean"])
    if abs(fixed_r - 0.5456440150342281) > 1e-12 or not fixed_r > alternating_r:
        raise RuntimeError("sub-01 no longer authorizes the fixed-Q population method")
    run_root.mkdir(parents=True, exist_ok=True)
    contract_payload = {
        "schema_version": 1, "kind": "swpd_contextual_fixed_q_neural_population_contract",
        "follow_up_status": "frozen secondary analysis after sub-01 neural development",
        "subjects": list(subjects), "seeds": list(seeds), "folds": list(folds), "diagnostic": diagnostic,
        "cache_root": str(cache_root), "cache_receipts": _cache_receipts(cache_root, subjects, folds),
        "development_summary": str(development_summary),
        "development_summary_sha256": SUB01_SUMMARY_SHA256,
        "frozen_method_choice": {"method": "fixed_q_neural", "fixed_r": fixed_r, "alternating_r": alternating_r},
        "device": args.device, "max_cycles": args.max_cycles, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "max_train_batches": args.max_train_batches, "max_eval_batches": args.max_eval_batches,
        "architecture": "dynamic channels -> Linear/Conv1D/BiGRU residual decoder -> fixed Whisper L4 PCA50",
        "split": "test i; validation i+1; remaining three train", "sources": _sources(),
    }
    contract_fp = fingerprint_json(contract_payload)
    contract = {**contract_payload, "fingerprint": contract_fp, "created_utc": _now()}
    contract_path = run_root / "run_contract.json"
    if contract_path.is_file():
        existing = read_json(contract_path)
        existing_payload = {key: value for key, value in existing.items() if key not in ("fingerprint", "created_utc")}
        if fingerprint_json(existing_payload) != contract_fp:
            raise RuntimeError("existing population run contract differs")
    else:
        atomic_write_json(contract_path, contract, overwrite=False)
    print(f"FROZEN FIXED-Q NEURAL POPULATION FIT | subjects={list(subjects)} seeds={list(seeds)} folds={list(folds)} diagnostic={diagnostic}", flush=True)
    selections = []
    for subject in subjects:
        for seed in seeds:
            for fold in folds:
                selections.append(_fit_one(
                    subject=subject, seed=seed, fold=fold, cache_root=cache_root, run_root=run_root,
                    contract_fp=contract_fp, device=args.device, max_cycles=args.max_cycles,
                    epochs=args.epochs, batch_size=args.batch_size,
                    max_train_batches=args.max_train_batches, max_eval_batches=args.max_eval_batches,
                    diagnostic=diagnostic,
                ))
    expected = len(subjects) * len(seeds) * len(folds)
    summary = {
        "schema_version": 1, "kind": "swpd_contextual_fixed_q_neural_population_fit_summary",
        "run_contract_fingerprint": contract_fp, "diagnostic": diagnostic,
        "selection_count": len(selections), "expected_selection_count": expected,
        "all_selections_frozen": len(selections) == expected,
        "mean_legacy_validation_r": float(np.mean([item["legacy_validation"]["primary_r"] for item in selections])),
        "mean_fixed_validation_r": float(np.mean([item["fixed_selected_validation"]["primary_r"] for item in selections])),
        "test_evaluated": False, "completed_utc": _now(),
    }
    atomic_write_json(run_root / "fit_summary.json", summary)
    print(f"FIT COMPLETE | selections={len(selections)}/{expected} | TEST NOT EVALUATED", flush=True)
    return 0


def _describe(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64); n = len(array)
    mean = float(array.mean()); sd = float(array.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n else float("nan")
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "sem": float(sem), "ci95_t": [mean - critical * sem, mean + critical * sem], "values": array.tolist()}


def evaluate(args: argparse.Namespace) -> int:
    import torch

    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    cache_root, run_root = args.cache_root.resolve(), args.run_root.resolve()
    summary_path = run_root / "population_summary.json"
    artifact_manifest_path = run_root / "artifact_manifest.json"
    existing_summary = read_json(summary_path) if summary_path.is_file() else None
    existing_manifest = read_json(artifact_manifest_path) if artifact_manifest_path.is_file() else None
    if existing_manifest is not None:
        manifest_payload = {key: value for key, value in existing_manifest.items() if key != "fingerprint"}
        if existing_manifest.get("fingerprint") != fingerprint_json(manifest_payload):
            raise RuntimeError("existing population artifact manifest fingerprint changed")
    contract = read_json(run_root / "run_contract.json")
    payload = {key: value for key, value in contract.items() if key not in ("fingerprint", "created_utc")}
    contract_fp = fingerprint_json(payload)
    fit_summary = read_json(run_root / "fit_summary.json")
    production = (
        contract.get("fingerprint") == contract_fp and tuple(contract.get("subjects", ())) == SUBJECTS
        and tuple(contract.get("seeds", ())) == SEEDS and tuple(contract.get("folds", ())) == FOLDS
        and contract.get("diagnostic") is False and contract.get("device") == "cuda" and args.device == "cuda"
        and contract.get("max_cycles") == MAX_CYCLES and contract.get("epochs") == EPOCHS_PER_CYCLE
        and contract.get("batch_size") == BATCH_SIZE and contract.get("max_train_batches") is None
        and contract.get("max_eval_batches") is None and contract.get("sources") == _sources()
        and fit_summary.get("run_contract_fingerprint") == contract_fp
        and fit_summary.get("selection_count") == len(SUBJECTS) * len(SEEDS) * len(FOLDS)
        and fit_summary.get("all_selections_frozen") is True and fit_summary.get("test_evaluated") is False
        and fit_summary.get("diagnostic") is False
    )
    if not production:
        raise RuntimeError("only the exact complete production fit may open population test")
    if os.path.normcase(str(Path(contract["cache_root"]).resolve())) != os.path.normcase(str(cache_root)):
        raise RuntimeError("evaluation cache root differs from the frozen fit contract")
    if contract.get("development_summary_sha256") != SUB01_SUMMARY_SHA256:
        raise RuntimeError("population method-choice receipt changed")
    current_cache_receipts = _cache_receipts(cache_root, SUBJECTS, FOLDS)
    if contract.get("cache_receipts") != current_cache_receipts:
        raise RuntimeError("population cache receipts changed after fit")
    selections: dict[tuple[str, int, int], dict[str, Any]] = {}
    inventory = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            for fold in FOLDS:
                root = run_root / "subjects" / subject / "seeds" / f"seed_{seed}" / "folds" / f"fold_{fold:02d}"
                path = root / "selection_frozen.json"
                item = _selection(path, contract_fp, subject, seed, fold)
                selections[(subject, seed, fold)] = item
                inventory.append({"subject": subject, "seed": seed, "fold": fold, "selection_sha256": sha256_file(path), "artifact_sha256": item["artifact_sha256"]})
    pretest = {"schema_version": 1, "kind": "swpd_fixed_q_neural_population_pretest", "run_contract_fingerprint": contract_fp, "count": len(inventory), "items": inventory, "created_utc": _now()}
    pretest["fit_summary_sha256"] = sha256_file(run_root / "fit_summary.json")
    pretest["fingerprint"] = fingerprint_json(pretest)
    pretest_path = run_root / "pre_test_inventory.json"
    if pretest_path.is_file():
        existing = read_json(pretest_path)
        existing_payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if existing.get("fingerprint") != fingerprint_json(existing_payload):
            raise RuntimeError("existing pretest inventory fingerprint changed")
        for key in ("run_contract_fingerprint", "count", "items", "fit_summary_sha256"):
            if existing.get(key) != pretest.get(key):
                raise RuntimeError("existing pretest inventory differs")
        pretest = existing
    else:
        atomic_write_json(pretest_path, pretest, overwrite=False)
    authorization_path = run_root / "test_gate_authorization.json"
    authorization = {
        "schema_version": 1, "kind": "separate_command_population_test_authorization",
        "pretest_sha256": sha256_file(pretest_path), "authorized_utc": _now(),
    }
    authorization["fingerprint"] = fingerprint_json(authorization)
    if authorization_path.is_file():
        existing_authorization = read_json(authorization_path)
        existing_payload = {key: value for key, value in existing_authorization.items() if key != "fingerprint"}
        if (
            existing_authorization.get("fingerprint") != fingerprint_json(existing_payload)
            or existing_authorization.get("pretest_sha256") != authorization["pretest_sha256"]
        ):
            raise RuntimeError("existing population test authorization changed")
        authorization = existing_authorization
    else:
        atomic_write_json(authorization_path, authorization, overwrite=False)
    print("[gate] 200/200 selections frozen; opening fold-role tests", flush=True)
    rows = []
    for subject in SUBJECTS:
        test_blocks = {fold: load_block(cache_root, subject, fold) for fold in FOLDS}
        for seed in SEEDS:
            for fold in FOLDS:
                item = selections[(subject, seed, fold)]
                root = run_root / "subjects" / subject / "seeds" / f"seed_{seed}" / "folds" / f"fold_{fold:02d}"
                completion_path = root / "test_complete.json"; metrics_path = root / "test_metrics.json"
                if completion_path.is_file():
                    completion = read_json(completion_path); metrics = read_json(metrics_path)
                    completion_payload = {key: value for key, value in completion.items() if key != "fingerprint"}
                    if (
                        completion.get("fingerprint") != fingerprint_json(completion_payload)
                        or completion.get("run_contract_fingerprint") != contract_fp
                        or completion.get("selection_sha256") != sha256_file(root / "selection_frozen.json")
                        or completion.get("metrics_path") != str(metrics_path)
                        or completion.get("metrics_sha256") != sha256_file(metrics_path)
                        or metrics.get("subject") != subject or int(metrics.get("seed", -1)) != seed
                        or int(metrics.get("fold", -1)) != fold
                    ):
                        raise RuntimeError("completed test identity/hash changed")
                    rows.append(metrics); print(f"[{subject} seed {seed} fold {fold}] reuse", flush=True); continue
                artifact = torch.load(Path(item["artifact_path"]), map_location="cpu", weights_only=False)
                block = test_blocks[fold]
                neural_scaler = _standardizer(artifact, "neural_scaler"); neural_pca = _pca(artifact, "neural_pca")
                mel_scaler = _standardizer(artifact, "mel_scaler")
                target_space = TargetSearchSpace(_standardizer(artifact, "target_scaler"), _pca(artifact, "target_search_pca"), OUTPUT_DIM)
                neural_z = neural_scaler.transform(block.neural); test_x = neural_pca.transform(neural_z)
                inputs = np.asarray(neural_z.reshape(-1, CONTEXT_STEPS, int(artifact["channels"])), dtype=np.float32)
                mel_z = mel_scaler.transform(block.mel80); q0 = np.asarray(artifact["q0"])
                legacy_prediction = _affine(artifact, "legacy_decoder").predict(test_x)
                model = ContextualResidualDecoder(CONTEXT_STEPS, int(artifact["channels"]), OUTPUT_DIM).to(args.device)
                model.load_state_dict(artifact["fixed_model_state"])
                fixed_prediction = _predict(model, inputs, args.device, BATCH_SIZE)
                legacy = _endpoint(legacy_prediction, block.l4, mel_z, target_space, q0, _affine(artifact, "legacy_probe"))
                fixed = _endpoint(fixed_prediction, block.l4, mel_z, target_space, q0, _affine(artifact, "fixed_probe"))
                metrics = {"subject": subject, "seed": seed, "fold": fold, "legacy": legacy, "fixed_neural": fixed, "delta": fixed["primary_r"] - legacy["primary_r"]}
                atomic_write_json(metrics_path, metrics, overwrite=False)
                completion = {"schema_version": 1, "run_contract_fingerprint": contract_fp, "subject": subject, "seed": seed, "fold": fold, "selection_sha256": sha256_file(root / "selection_frozen.json"), "metrics_path": str(metrics_path), "metrics_sha256": sha256_file(metrics_path), "completed_utc": _now()}
                completion["fingerprint"] = fingerprint_json(completion); atomic_write_json(completion_path, completion, overwrite=False)
                rows.append(metrics); print(f"[{subject} seed {seed} fold {fold}] legacy={legacy['primary_r']:.6f} fixed={fixed['primary_r']:.6f} delta={metrics['delta']:+.6f}", flush=True)
    expected_pairs = {(subject, seed, fold) for subject in SUBJECTS for seed in SEEDS for fold in FOLDS}
    observed_pairs = {(row["subject"], int(row["seed"]), int(row["fold"])) for row in rows}
    if observed_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise RuntimeError("population test result set is not exactly 8 x 5 x 5")
    subject_rows = []
    legacy_reference = read_json(
        EXTERNAL_ROOT / "swpd_contextual_frozen" / "results" / "population_summary.json"
    )
    legacy_by_subject = {
        row["subject"]: float(row["whisper_l4_pca50_r"])
        for row in legacy_reference["subject_rows"]
    }
    for subject in SUBJECTS:
        seed_rows = []
        for seed in SEEDS:
            group = [row for row in rows if row["subject"] == subject and row["seed"] == seed]
            if {row["fold"] for row in group} != set(FOLDS):
                raise RuntimeError("subject seed is missing folds")
            seed_rows.append({
                "seed": seed,
                "legacy": float(np.mean([row["legacy"]["primary_r"] for row in group])),
                "fixed": float(np.mean([row["fixed_neural"]["primary_r"] for row in group])),
                "legacy_low20": float(np.mean([row["legacy"]["common_mel80"]["lower_20_bins"]["mean_pearson_r"] for row in group])),
                "fixed_low20": float(np.mean([row["fixed_neural"]["common_mel80"]["lower_20_bins"]["mean_pearson_r"] for row in group])),
            })
        legacy = float(np.mean([row["legacy"] for row in seed_rows])); fixed_values = [row["fixed"] for row in seed_rows]
        legacy_low = float(np.mean([row["legacy_low20"] for row in seed_rows])); fixed_low_values = [row["fixed_low20"] for row in seed_rows]
        if abs(legacy - legacy_by_subject[subject]) > 1e-9:
            raise RuntimeError(f"{subject} legacy path does not reproduce the frozen L4 result")
        subject_rows.append({"subject": subject, "legacy_r": legacy, "fixed_neural_r": float(np.mean(fixed_values)), "delta": float(np.mean(fixed_values) - legacy), "fixed_optimizer_sd": float(np.std(fixed_values, ddof=1)), "legacy_low20_r": legacy_low, "fixed_neural_low20_r": float(np.mean(fixed_low_values)), "delta_low20": float(np.mean(fixed_low_values) - legacy_low), "seed_rows": seed_rows})
    deltas = [row["delta"] for row in subject_rows]; low_deltas = [row["delta_low20"] for row in subject_rows]
    summary = {"schema_version": 1, "kind": "swpd_contextual_fixed_q_neural_population_summary", "subjects": list(SUBJECTS), "n_subjects": len(SUBJECTS), "subject_rows": subject_rows, "population": {"legacy": _describe([row["legacy_r"] for row in subject_rows]), "fixed_neural": _describe([row["fixed_neural_r"] for row in subject_rows]), "delta_fixed_minus_legacy": _describe(deltas), "wins": int(sum(value > 0 for value in deltas)), "paired_t_p": float(stats.ttest_1samp(deltas, 0).pvalue), "exact_sign_p": float(stats.binomtest(sum(value > 0 for value in deltas), len(deltas), .5).pvalue), "delta_low20": _describe(low_deltas)}, "test_item_count": len(rows), "completed_utc": _now()}
    if existing_summary is not None:
        for key, value in summary.items():
            if key != "completed_utc" and existing_summary.get(key) != value:
                raise RuntimeError("existing population summary differs")
        summary = existing_summary
    else:
        atomic_write_json(summary_path, summary, overwrite=False)
    manifest = {"schema_version": 1, "kind": "swpd_fixed_q_neural_population_manifest", "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path), "pretest_sha256": sha256_file(pretest_path), "authorization_sha256": sha256_file(authorization_path), "test_item_count": len(rows), "created_utc": _now()}
    manifest["fingerprint"] = fingerprint_json(manifest)
    if existing_manifest is not None:
        for key, value in manifest.items():
            if key not in ("created_utc", "fingerprint") and existing_manifest.get(key) != value:
                raise RuntimeError("existing population manifest inventory differs")
        print(f"POPULATION TEST ALREADY COMPLETE AND REVALIDATED | {summary_path}", flush=True)
        return 0
    atomic_write_json(artifact_manifest_path, manifest, overwrite=False)
    print(f"POPULATION TEST COMPLETE | legacy={summary['population']['legacy']['mean']:.6f} fixed={summary['population']['fixed_neural']['mean']:.6f} delta={summary['population']['delta_fixed_minus_legacy']['mean']:+.6f} wins={summary['population']['wins']}/8", flush=True)
    print(f"[done] {summary_path}", flush=True)
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, default=Path(r"C:\WhisperECoG_Work\SWPD\runs\contextual_neural_e2e_sub01_v1\summary.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--subjects", default=",".join(SUBJECTS)); parser.add_argument("--seeds", default=",".join(map(str, SEEDS))); parser.add_argument("--folds", default=",".join(map(str, FOLDS)))
    parser.add_argument("--max-cycles", type=int, default=MAX_CYCLES); parser.add_argument("--epochs", type=int, default=EPOCHS_PER_CYCLE); parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-eval-batches", type=int); parser.add_argument("--diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return fit(args) if args.stage == "fit" else evaluate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FloatingPointError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
