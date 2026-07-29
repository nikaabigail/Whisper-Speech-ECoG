#!/usr/bin/env python3
"""Fit, select and freeze paired contextual neural E2E arms on SWPD sub-01."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

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
    exact_projector_update,
    fit_affine,
    mse,
    project_scores,
)
from swpd_contextual_neural_e2e.core import (  # noqa: E402
    ContextualResidualDecoder,
    fold_legacy_pipeline,
    state_dict_sha256,
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
DEFAULT_SEEDS = (1, 2, 3, 4, 42)
ALL_FOLDS = (0, 1, 2, 3, 4)
PRODUCTION_MAX_CYCLES = 5
PRODUCTION_EPOCHS_PER_CYCLE = 10
PRODUCTION_BATCH_SIZE = 256
PRODUCTION_LEARNING_RATE = 2e-4
PRODUCTION_WEIGHT_DECAY = 1e-4
PRODUCTION_GRAD_CLIP = 1.0
CONTEXT_STEPS = 9
CHANNELS = 127
INPUT_DIM = CONTEXT_STEPS * CHANNELS
SEARCH_DIM = 128
OUTPUT_DIM = 50


@dataclass(frozen=True)
class Block:
    index: int
    sample_ids: np.ndarray
    times: np.ndarray
    neural: np.ndarray
    mel80: np.ndarray
    l4: np.ndarray


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cpu_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [item.cpu() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("integer list must be non-empty and unique")
    return values


def _rows(blocks: Mapping[int, Block], indexes: Sequence[int], field: str) -> np.ndarray:
    return np.concatenate([getattr(blocks[index], field) for index in indexes], axis=0)


def load_block(cache: Path, index: int) -> Block:
    """Load one contextual cache block after verifying its immutable receipt."""

    manifest_path = cache / f"block_{index:02d}.json"
    manifest = read_json(manifest_path)
    stored = manifest.pop("fingerprint", None)
    if stored != fingerprint_json(manifest):
        raise RuntimeError(f"contextual cache manifest changed: block {index}")
    if manifest.get("kind") != "swpd_sub01_contextual_whisper_block_cache":
        raise RuntimeError("contextual cache kind changed")
    if manifest.get("subject") != "sub-01" or int(manifest.get("block", -1)) != index:
        raise RuntimeError("contextual cache subject/block changed")
    if manifest.get("confirmatory_subjects_read") is not False:
        raise RuntimeError("development cache provenance changed")
    if manifest.get("neural_context_ms") != [-200, -150, -100, -50, 0, 50, 100, 150, 200]:
        raise RuntimeError("contextual timing changed")
    arrays_path = cache / str(manifest["arrays_file"])
    if sha256_file(arrays_path) != manifest["arrays_sha256"]:
        raise RuntimeError(f"contextual cache arrays changed: block {index}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        block = Block(
            index=index,
            sample_ids=np.asarray(archive["sample_ids"]),
            times=np.asarray(archive["frame_times_seconds"], dtype=np.float64),
            neural=np.asarray(archive["neural_context"], dtype=np.float64),
            mel80=np.asarray(archive["mel80"], dtype=np.float64),
            l4=np.asarray(archive["L4"], dtype=np.float64),
        )
    rows = len(block.sample_ids)
    if block.neural.shape != (rows, INPUT_DIM):
        raise RuntimeError("contextual neural geometry changed")
    if block.mel80.shape != (rows, 80) or block.l4.shape != (rows, 512):
        raise RuntimeError("contextual target geometry changed")
    if np.unique(block.sample_ids).size != rows:
        raise RuntimeError("sample IDs are not unique within a block")
    if any(not value.all() for value in (
        np.isfinite(block.times), np.isfinite(block.neural),
        np.isfinite(block.mel80), np.isfinite(block.l4),
    )):
        raise RuntimeError("contextual cache contains non-finite values")
    if np.any(np.diff(block.times) <= 0):
        raise RuntimeError("contextual timeline is not strictly increasing")
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


def _projector_receipt(projector: np.ndarray, whitened_train: np.ndarray) -> dict[str, Any]:
    q = np.asarray(projector, dtype=np.float64)
    scores = np.asarray(whitened_train, dtype=np.float64) @ q.T
    variance = np.var(scores, axis=0, ddof=1)
    singular = np.linalg.svd(q, compute_uv=False)
    return {
        "shape": list(q.shape),
        "rank": int(np.linalg.matrix_rank(q)),
        "orthogonality_frobenius_error": float(
            np.linalg.norm(q @ q.T - np.eye(q.shape[0]), ord="fro")
        ),
        "minimum_singular_value": float(np.min(singular)),
        "projected_train_variance_min": float(np.min(variance)),
        "projected_train_variance_max": float(np.max(variance)),
    }


def _loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.as_tensor(inputs, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
        persistent_workers=False,
    )


def _predict(
    model: Any,
    inputs: np.ndarray,
    device: str,
    batch_size: int,
    max_batches: int | None = None,
) -> np.ndarray:
    import torch

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            if max_batches is not None and len(chunks) >= max_batches:
                break
            batch = torch.as_tensor(
                inputs[start:start + batch_size], dtype=torch.float32, device=device
            )
            output = model(batch)
            if output.ndim != 2 or output.shape[1] != OUTPUT_DIM:
                raise RuntimeError("neural decoder output geometry changed")
            if not torch.isfinite(output).all().item():
                raise FloatingPointError("non-finite neural prediction")
            chunks.append(output.detach().cpu().numpy())
    if not chunks:
        raise RuntimeError("prediction produced no batches")
    return np.concatenate(chunks, axis=0)


def _train_phase(
    model: Any,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    phase_seed: int,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train one phase; choose its epoch solely by full-train MSE."""

    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.MSELoss()
    baseline_prediction = _predict(
        model, train_inputs, device, batch_size, max_eval_batches
    )
    baseline_rows = len(baseline_prediction)
    best_mse = mse(train_targets[:baseline_rows], baseline_prediction)
    best_state: dict[str, Any] | None = _cpu_state(model)
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "batch_train_mse": None,
        "full_train_mse": best_mse,
        "full_train_rows": baseline_rows,
        "candidate": "phase_input_state",
    }]
    for epoch in range(1, epochs + 1):
        model.train()
        seen = 0
        summed = 0.0
        loader = _loader(
            train_inputs, train_targets, batch_size,
            phase_seed + epoch,
        )
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"{label}: non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(norm).item():
                raise FloatingPointError(f"{label}: non-finite gradient norm")
            optimizer.step()
            count = int(inputs.shape[0])
            summed += float(loss.detach().cpu()) * count
            seen += count
        if seen == 0:
            raise RuntimeError(f"{label}: no training batches")
        full_prediction = _predict(
            model, train_inputs, device, batch_size, max_eval_batches
        )
        compared = len(full_prediction)
        full_train_mse = mse(train_targets[:compared], full_prediction)
        row = {
            "epoch": epoch,
            "batch_train_mse": summed / seen,
            "full_train_mse": full_train_mse,
            "full_train_rows": compared,
        }
        history.append(row)
        print(
            f"[{label} epoch {epoch:02d}] batchMSE={row['batch_train_mse']:.6f} "
            f"fullTrainMSE={full_train_mse:.6f}",
            flush=True,
        )
        if full_train_mse < best_mse:
            best_mse = full_train_mse
            best_state = _cpu_state(model)
    if best_state is None:
        raise RuntimeError(f"{label}: no selected epoch")
    model.load_state_dict(best_state)
    if best_mse > history[0]["full_train_mse"] + 1e-8 * max(
        1.0, abs(history[0]["full_train_mse"])
    ):
        raise RuntimeError(f"{label}: selected model phase increased train objective")
    return best_state, history


def _primary(metrics: Mapping[str, Any]) -> float:
    if int(metrics["all_bins"].get("component_count", -1)) != 80:
        raise RuntimeError("primary MEL80 metric does not contain exactly 80 bins")
    return float(metrics["all_bins"]["mean_pearson_r"])


def _endpoint(
    model: Any,
    projector: np.ndarray,
    target_space: TargetSearchSpace,
    train_whitened: np.ndarray,
    val_whitened: np.ndarray,
    train_mel_z: np.ndarray,
    val_mel_z: np.ndarray,
    val_l4_z: np.ndarray,
    val_inputs: np.ndarray,
    *,
    device: str,
    batch_size: int,
    max_eval_batches: int | None,
) -> tuple[dict[str, Any], AffineMap]:
    train_scores = project_scores(train_whitened, projector)
    probe = fit_affine(train_scores, train_mel_z)
    prediction = _predict(model, val_inputs, device, batch_size, max_eval_batches)
    rows = len(prediction)
    truth_scores = project_scores(val_whitened[:rows], projector)
    predicted_mel = probe.predict(prediction)
    predicted_l4 = target_space.reconstruct_standardized(prediction, projector)
    common = component_metrics(val_mel_z[:rows], predicted_mel)
    result = {
        "validation_rows": rows,
        "common_mel80": common,
        "primary_r": _primary(common),
        "target_score50": component_metrics(truth_scores, prediction),
        "l4_full512": component_metrics(val_l4_z[:rows], predicted_l4),
        "l4_full512_mse": mse(val_l4_z[:rows], predicted_l4),
    }
    return result, probe


def _legacy_endpoint(
    decoder: AffineMap,
    projector: np.ndarray,
    target_space: TargetSearchSpace,
    train_scores: np.ndarray,
    val_x: np.ndarray,
    val_whitened: np.ndarray,
    train_mel_z: np.ndarray,
    val_mel_z: np.ndarray,
    val_l4_z: np.ndarray,
) -> tuple[dict[str, Any], AffineMap]:
    probe = fit_affine(train_scores, train_mel_z)
    prediction = decoder.predict(val_x)
    truth_scores = project_scores(val_whitened, projector)
    predicted_mel = probe.predict(prediction)
    predicted_l4 = target_space.reconstruct_standardized(prediction, projector)
    common = component_metrics(val_mel_z, predicted_mel)
    return {
        "validation_rows": len(prediction),
        "common_mel80": common,
        "primary_r": _primary(common),
        "target_score50": component_metrics(truth_scores, prediction),
        "l4_full512": component_metrics(val_l4_z, predicted_l4),
        "l4_full512_mse": mse(val_l4_z, predicted_l4),
    }, probe


def _validated_selection(
    path: Path,
    contract_fp: str,
    *,
    expected_seed: int,
    expected_fold: int,
) -> dict[str, Any]:
    selection = read_json(path)
    stored = selection.get("fingerprint")
    payload = {key: value for key, value in selection.items() if key != "fingerprint"}
    if stored != fingerprint_json(payload):
        raise RuntimeError(f"invalid frozen selection fingerprint: {path}")
    if selection.get("run_contract_fingerprint") != contract_fp:
        raise RuntimeError("frozen selection belongs to another contract")
    if (
        int(selection.get("seed", -1)) != expected_seed
        or int(selection.get("fold", -1)) != expected_fold
    ):
        raise RuntimeError("frozen selection seed/fold identity changed")
    artifact = Path(selection["artifact_path"])
    if not artifact.is_file() or sha256_file(artifact) != selection["artifact_sha256"]:
        raise RuntimeError(f"frozen selection artifact changed: {artifact}")
    if selection.get("test_evaluated") is not False:
        raise RuntimeError("selection is not a pre-test artifact")
    return selection


def _fit_fold_seed(
    *,
    fold: int,
    seed: int,
    cache: Path,
    output_root: Path,
    contract_fp: str,
    device: str,
    max_cycles: int,
    epochs_per_cycle: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    diagnostic: bool,
) -> dict[str, Any]:
    import torch

    item_root = output_root / "seeds" / f"seed_{seed}" / "folds" / f"fold_{fold:02d}"
    selection_path = item_root / "selection_frozen.json"
    if selection_path.is_file():
        print(f"[seed {seed} fold {fold}] validating frozen selection", flush=True)
        return _validated_selection(
            selection_path,
            contract_fp,
            expected_seed=seed,
            expected_fold=fold,
        )

    validation = (fold + 1) % 5
    train = tuple(index for index in ALL_FOLDS if index not in (fold, validation))
    blocks = {index: load_block(cache, index) for index in train + (validation,)}
    train_ids = _rows(blocks, train, "sample_ids")
    val_ids = _rows(blocks, (validation,), "sample_ids")
    if np.intersect1d(train_ids, val_ids).size:
        raise RuntimeError("train/validation IDs overlap")
    print(
        f"[seed {seed} fold {fold}] train={list(train)} validation={validation} "
        f"test={fold} role-excluded",
        flush=True,
    )

    train_neural = _rows(blocks, train, "neural")
    val_neural = _rows(blocks, (validation,), "neural")
    neural_scaler = Standardizer.fit(train_neural)
    train_neural_z = neural_scaler.transform(train_neural)
    val_neural_z = neural_scaler.transform(val_neural)
    neural_pca = PCATransform.fit(train_neural_z, OUTPUT_DIM, whiten=False)
    train_x = neural_pca.transform(train_neural_z)
    val_x = neural_pca.transform(val_neural_z)
    train_inputs = np.asarray(
        train_neural_z.reshape(-1, CONTEXT_STEPS, CHANNELS), dtype=np.float32
    )
    val_inputs = np.asarray(
        val_neural_z.reshape(-1, CONTEXT_STEPS, CHANNELS), dtype=np.float32
    )

    train_l4 = _rows(blocks, train, "l4")
    val_l4 = _rows(blocks, (validation,), "l4")
    target_space = TargetSearchSpace.fit(
        train_l4, search_dim=SEARCH_DIM, output_dim=OUTPUT_DIM
    )
    train_whitened = target_space.transform(train_l4)
    val_whitened = target_space.transform(val_l4)
    val_l4_z = target_space.scaler.transform(val_l4)
    q0 = target_space.initial_projector()
    train_scores0 = project_scores(train_whitened, q0)

    # Exact parity with the latest contextual L4 PCA50 target transform.
    legacy_pca50 = PCATransform.fit(
        target_space.scaler.transform(train_l4), OUTPUT_DIM, whiten=True
    )
    parity = float(np.max(np.abs(
        legacy_pca50.transform(target_space.scaler.transform(train_l4))
        - train_scores0
    )))
    if parity > 1e-9:
        raise RuntimeError(f"cycle-zero target parity failed: {parity:.3e}")

    train_mel = _rows(blocks, train, "mel80")
    val_mel = _rows(blocks, (validation,), "mel80")
    mel_scaler = Standardizer.fit(train_mel)
    train_mel_z = mel_scaler.transform(train_mel)
    val_mel_z = mel_scaler.transform(val_mel)

    legacy_decoder = fit_affine(train_x, train_scores0)
    legacy_metrics, legacy_probe = _legacy_endpoint(
        legacy_decoder, q0, target_space, train_scores0, val_x,
        val_whitened, train_mel_z, val_mel_z, val_l4_z,
    )

    _set_seed(seed)
    model = ContextualResidualDecoder(
        context_steps=CONTEXT_STEPS, channels=CHANNELS, output_dim=OUTPUT_DIM
    ).to(device)
    folded = fold_legacy_pipeline(neural_pca, legacy_decoder)
    model.initialize_legacy_skip(folded.coef, folded.intercept)
    neural_legacy = _predict(model, val_inputs, device, batch_size)
    model_parity = float(np.max(np.abs(neural_legacy - legacy_decoder.predict(val_x))))
    if model_parity > 5e-5:
        raise RuntimeError(f"legacy skip parity failed: {model_parity:.3e}")
    architecture = model.architecture_receipt()
    if architecture.get("dropout", 0) != 0 or architecture.get("batch_norm", False):
        raise RuntimeError("paired model unexpectedly uses dropout/BatchNorm")
    legacy_model_state = _cpu_state(model)
    legacy_model_state_sha256 = state_dict_sha256(legacy_model_state)

    checkpoint_path = item_root / "fit_checkpoint.pt"
    fixed_current_state: dict[str, Any] | None = None
    alternating_current_state: dict[str, Any] | None = None
    alternating_q = np.array(q0, copy=True)
    fixed_history: list[dict[str, Any]] = []
    alternating_history: list[dict[str, Any]] = []
    fixed_best: dict[str, Any] | None = None
    alternating_best: dict[str, Any] | None = None
    next_cycle = 1
    shared_cycle1_state_sha256: str | None = None
    if checkpoint_path.is_file():
        # CPU loading is deliberate: CPU RNG ByteTensor must never be remapped to CUDA.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_contract_fingerprint") != contract_fp:
            raise RuntimeError("resume checkpoint belongs to another contract")
        if int(checkpoint.get("fold", -1)) != fold or int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError("resume checkpoint fold/seed identity changed")
        fixed_current_state = checkpoint["fixed_current_state"]
        alternating_current_state = checkpoint["alternating_current_state"]
        alternating_q = np.asarray(checkpoint["alternating_q"], dtype=np.float64)
        fixed_history = list(checkpoint["fixed_history"])
        alternating_history = list(checkpoint["alternating_history"])
        fixed_best = checkpoint["fixed_best"]
        alternating_best = checkpoint["alternating_best"]
        shared_cycle1_state_sha256 = checkpoint.get("shared_cycle1_state_sha256")
        next_cycle = int(checkpoint["next_cycle"])
        completed_cycles = int(checkpoint.get("completed_cycles", -1))
        if not 1 <= next_cycle <= max_cycles + 1:
            raise RuntimeError("resume checkpoint next_cycle is outside the run contract")
        if completed_cycles != next_cycle - 1:
            raise RuntimeError("resume checkpoint cycle counters are inconsistent")
        if (
            len(fixed_history) != completed_cycles
            or len(alternating_history) != completed_cycles
        ):
            raise RuntimeError("resume checkpoint history length is inconsistent")
        expected_cycles = list(range(1, completed_cycles + 1))
        if (
            [int(item.get("cycle", -1)) for item in fixed_history] != expected_cycles
            or [int(item.get("cycle", -1)) for item in alternating_history]
            != expected_cycles
        ):
            raise RuntimeError("resume checkpoint history cycle order is inconsistent")
        projector_receipt = _projector_receipt(alternating_q, train_whitened)
        if (
            projector_receipt["shape"] != [OUTPUT_DIM, SEARCH_DIM]
            or projector_receipt["rank"] != OUTPUT_DIM
            or projector_receipt["orthogonality_frobenius_error"] > 1e-8
            or projector_receipt["projected_train_variance_min"] < 0.99
            or projector_receipt["projected_train_variance_max"] > 1.01
        ):
            raise RuntimeError("resume checkpoint projector integrity failed")
        _restore_rng_state(checkpoint["rng_state"])
        print(
            f"[seed {seed} fold {fold}] resume next_cycle={next_cycle}", flush=True
        )

    for cycle in range(next_cycle, max_cycles + 1):
        phase_seed = seed * 100_000 + fold * 1_000 + cycle * 10
        if cycle == 1:
            # Both arms are identical here; train once and clone the endpoint.
            model.load_state_dict(legacy_model_state)
            shared_state, shared_epochs = _train_phase(
                model, train_inputs, train_scores0,
                device=device, epochs=epochs_per_cycle, batch_size=batch_size,
                learning_rate=learning_rate, weight_decay=weight_decay,
                grad_clip=grad_clip, phase_seed=phase_seed,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                label=f"seed {seed} fold {fold} shared cycle 1",
            )
            fixed_current_state = deepcopy(shared_state)
            alternating_current_state = deepcopy(shared_state)
            shared_cycle1_state_sha256 = state_dict_sha256(shared_state)
            fixed_epoch_rows = deepcopy(shared_epochs)
            alternating_epoch_rows = deepcopy(shared_epochs)
        else:
            assert fixed_current_state is not None and alternating_current_state is not None
            model.load_state_dict(fixed_current_state)
            fixed_current_state, fixed_epoch_rows = _train_phase(
                model, train_inputs, train_scores0,
                device=device, epochs=epochs_per_cycle, batch_size=batch_size,
                learning_rate=learning_rate, weight_decay=weight_decay,
                grad_clip=grad_clip, phase_seed=phase_seed,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                label=f"seed {seed} fold {fold} fixed cycle {cycle}",
            )
            model.load_state_dict(alternating_current_state)
            alternating_targets = project_scores(train_whitened, alternating_q)
            alternating_current_state, alternating_epoch_rows = _train_phase(
                model, train_inputs, alternating_targets,
                device=device, epochs=epochs_per_cycle, batch_size=batch_size,
                learning_rate=learning_rate, weight_decay=weight_decay,
                grad_clip=grad_clip, phase_seed=phase_seed,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                label=f"seed {seed} fold {fold} alternating cycle {cycle}",
            )

        assert fixed_current_state is not None and alternating_current_state is not None
        model.load_state_dict(fixed_current_state)
        fixed_metrics, fixed_probe = _endpoint(
            model, q0, target_space, train_whitened, val_whitened,
            train_mel_z, val_mel_z, val_l4_z, val_inputs,
            device=device, batch_size=batch_size, max_eval_batches=max_eval_batches,
        )

        model.load_state_dict(alternating_current_state)
        # Procrustes relies on the train target covariance being identity.
        # Even a diagnostic run must therefore use every train row here;
        # max_eval_batches is allowed only for quick metric/model checks.
        train_prediction = _predict(
            model, train_inputs, device, batch_size, None
        )
        update_rows = len(train_prediction)
        previous_q = np.array(alternating_q, copy=True)
        alternating_q, procrustes = exact_projector_update(
            train_whitened[:update_rows], train_prediction, alternating_q
        )
        alternating_metrics, alternating_probe = _endpoint(
            model, alternating_q, target_space, train_whitened, val_whitened,
            train_mel_z, val_mel_z, val_l4_z, val_inputs,
            device=device, batch_size=batch_size, max_eval_batches=max_eval_batches,
        )
        fixed_row = {
            "cycle": cycle,
            "phase": "shared_q0" if cycle == 1 else "fixed_q0_compute_matched",
            "epochs": fixed_epoch_rows,
            "validation": fixed_metrics,
        }
        alternating_row = {
            "cycle": cycle,
            "phase": "shared_model_then_procrustes" if cycle == 1
            else "alternating_model_then_procrustes",
            "epochs": alternating_epoch_rows,
            "procrustes": procrustes,
            "projector_change_frobenius": float(
                np.linalg.norm(alternating_q - previous_q, ord="fro")
            ),
            "projector_receipt": _projector_receipt(alternating_q, train_whitened),
            "validation": alternating_metrics,
        }
        fixed_history.append(fixed_row)
        alternating_history.append(alternating_row)
        if fixed_best is None or fixed_metrics["primary_r"] > fixed_best["score"]:
            fixed_best = {
                "cycle": cycle,
                "score": fixed_metrics["primary_r"],
                "model_state": deepcopy(fixed_current_state),
                "probe": fixed_probe,
                "metrics": fixed_metrics,
            }
        if alternating_best is None or alternating_metrics["primary_r"] > alternating_best["score"]:
            alternating_best = {
                "cycle": cycle,
                "score": alternating_metrics["primary_r"],
                "model_state": deepcopy(alternating_current_state),
                "projector": np.array(alternating_q, copy=True),
                "probe": alternating_probe,
                "metrics": alternating_metrics,
            }
        print(
            f"[seed {seed} fold {fold} cycle {cycle}] "
            f"legacy={legacy_metrics['primary_r']:.6f} "
            f"fixed={fixed_metrics['primary_r']:.6f} "
            f"alternating={alternating_metrics['primary_r']:.6f}",
            flush=True,
        )
        _atomic_torch_save(checkpoint_path, {
            "schema_version": 1,
            "run_contract_fingerprint": contract_fp,
            "fold": fold,
            "seed": seed,
            "next_cycle": cycle + 1,
            "fixed_current_state": fixed_current_state,
            "alternating_current_state": alternating_current_state,
            "alternating_q": alternating_q,
            "fixed_history": fixed_history,
            "alternating_history": alternating_history,
            "fixed_best": fixed_best,
            "alternating_best": alternating_best,
            "shared_cycle1_state_sha256": shared_cycle1_state_sha256,
            "rng_state": _rng_state(),
            "completed_cycles": cycle,
        })
        del train_prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        fixed_best is None
        or alternating_best is None
        or shared_cycle1_state_sha256 is None
    ):
        raise RuntimeError("paired selections are incomplete")
    artifact_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_neural_e2e_frozen_artifact",
        "run_contract_fingerprint": contract_fp,
        "fold": fold,
        "seed": seed,
        "train_blocks": list(train),
        "validation_block": validation,
        "test_block": fold,
        "architecture": architecture,
        "legacy_model_state": legacy_model_state,
        "fixed_model_state": fixed_best["model_state"],
        "alternating_model_state": alternating_best["model_state"],
        "q0": q0,
        "alternating_q": alternating_best["projector"],
        "fixed_selected_cycle": fixed_best["cycle"],
        "alternating_selected_cycle": alternating_best["cycle"],
        "legacy_skip_max_abs_error": model_parity,
        "cycle_zero_target_max_abs_error": parity,
        "legacy_initial_model_state_sha256": legacy_model_state_sha256,
        "shared_cycle1_model_state_sha256": shared_cycle1_state_sha256,
    }
    _put_standardizer(artifact_payload, "neural_scaler", neural_scaler)
    _put_pca(artifact_payload, "neural_pca", neural_pca)
    _put_standardizer(artifact_payload, "mel_scaler", mel_scaler)
    _put_standardizer(artifact_payload, "target_scaler", target_space.scaler)
    _put_pca(artifact_payload, "target_search_pca", target_space.pca)
    _put_affine(artifact_payload, "legacy_decoder", legacy_decoder)
    _put_affine(artifact_payload, "legacy_probe", legacy_probe)
    _put_affine(artifact_payload, "fixed_probe", fixed_best["probe"])
    _put_affine(artifact_payload, "alternating_probe", alternating_best["probe"])
    artifact_path = item_root / "frozen_artifact.pt"
    _atomic_torch_save(artifact_path, artifact_payload)
    selection: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_neural_e2e_fold_seed_selection",
        "run_contract_fingerprint": contract_fp,
        "development_only": True,
        "diagnostic_smoke": diagnostic,
        "fold": fold,
        "seed": seed,
        "train_blocks": list(train),
        "validation_block": validation,
        "test_block": fold,
        "train_count": int(len(train_ids)),
        "validation_count": int(len(val_ids)),
        "train_ids_sha256": fingerprint_json(train_ids.tolist()),
        "validation_ids_sha256": fingerprint_json(val_ids.tolist()),
        "legacy_cycle0": legacy_metrics,
        "fixed_selected_cycle": int(fixed_best["cycle"]),
        "fixed_selected_validation": fixed_best["metrics"],
        "alternating_selected_cycle": int(alternating_best["cycle"]),
        "alternating_selected_validation": alternating_best["metrics"],
        "primary_validation_delta_alternating_minus_fixed": float(
            alternating_best["score"] - fixed_best["score"]
        ),
        "fixed_history": fixed_history,
        "alternating_history": alternating_history,
        "q0_receipt": _projector_receipt(q0, train_whitened),
        "selected_alternating_q_receipt": _projector_receipt(
            alternating_best["projector"], train_whitened
        ),
        "cycle_zero_target_max_abs_error": parity,
        "legacy_skip_max_abs_error": model_parity,
        "legacy_initial_model_state_sha256": legacy_model_state_sha256,
        "shared_cycle1_model_state_sha256": shared_cycle1_state_sha256,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "test_evaluated": False,
        "frozen_utc": _now(),
    }
    selection["fingerprint"] = fingerprint_json(selection)
    atomic_write_json(selection_path, selection, overwrite=False)
    print(
        f"[seed {seed} fold {fold}] FROZEN | "
        f"fixed cycle={fixed_best['cycle']} r={fixed_best['score']:.6f} | "
        f"alternating cycle={alternating_best['cycle']} "
        f"r={alternating_best['score']:.6f}",
        flush=True,
    )
    return selection


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--folds", default=",".join(map(str, ALL_FOLDS)))
    parser.add_argument("--max-cycles", type=int, default=PRODUCTION_MAX_CYCLES)
    parser.add_argument(
        "--epochs-per-cycle", type=int, default=PRODUCTION_EPOCHS_PER_CYCLE
    )
    parser.add_argument("--batch-size", type=int, default=PRODUCTION_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=PRODUCTION_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=PRODUCTION_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=PRODUCTION_GRAD_CLIP)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--diagnostic-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    seeds = _parse_csv_ints(args.seeds)
    folds = _parse_csv_ints(args.folds)
    if any(item not in ALL_FOLDS for item in folds):
        raise ValueError("fold list must be a subset of 0..4")
    for value, name in (
        (args.max_cycles, "max-cycles"),
        (args.epochs_per_cycle, "epochs-per-cycle"),
        (args.batch_size, "batch-size"),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("optimizer parameters are invalid")
    diagnostic = bool(
        args.diagnostic_smoke
        or args.max_train_batches is not None
        or args.max_eval_batches is not None
        or seeds != DEFAULT_SEEDS
        or folds != ALL_FOLDS
        or args.device != "cuda"
        or args.max_cycles != PRODUCTION_MAX_CYCLES
        or args.epochs_per_cycle != PRODUCTION_EPOCHS_PER_CYCLE
        or args.batch_size != PRODUCTION_BATCH_SIZE
        or args.learning_rate != PRODUCTION_LEARNING_RATE
        or args.weight_decay != PRODUCTION_WEIGHT_DECAY
        or args.grad_clip != PRODUCTION_GRAD_CLIP
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache = args.cache_dir.expanduser().resolve()
    reference = args.reference_summary.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if sha256_file(reference) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("latest contextual sub-01 reference changed")
    reference_payload = read_json(reference)
    reference_l4 = float(
        reference_payload["results"]["targets"]["L4"]
        ["aggregate_common_mel80"]["all_bins"]["mean"]
    )
    if abs(reference_l4 - EXPECTED_REFERENCE_L4) > 1e-12:
        raise RuntimeError("latest contextual L4 reference value changed")

    cache_receipts = {}
    for index in ALL_FOLDS:
        manifest_path = cache / f"block_{index:02d}.json"
        manifest = read_json(manifest_path)
        cache_receipts[f"block_{index}"] = {
            "manifest_sha256": sha256_file(manifest_path),
            "declared_arrays_sha256": manifest["arrays_sha256"],
        }
    compatibility: dict[str, Any] = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_neural_e2e_contract",
        "development_only": True,
        "diagnostic_smoke": diagnostic,
        "base": "latest contextual L4 protocol sent to Ossadtchi",
        "reference_summary": str(reference),
        "reference_summary_sha256": EXPECTED_REFERENCE_SHA256,
        "reference_l4_common_mel80_r": EXPECTED_REFERENCE_L4,
        "cache_dir": str(cache),
        "cache_receipts": cache_receipts,
        "device": args.device,
        "folds": list(folds),
        "seeds": list(seeds),
        "split": "test i; validation i+1 cyclic; remaining three train",
        "architecture": {
            "input": "train-standardized high-gamma context 9x127 (-200..+200 ms)",
            "model": "deterministic residual neural decoder with exact legacy affine skip",
            "dropout": False,
            "batch_norm": False,
            "target": "Whisper-base L4 raw512",
            "target_space": "train StandardScaler -> whitened PCA128 -> row-orthonormal 50",
            "legacy": "neural StandardScaler -> PCA50(non-whitened) -> OLS to Q0 scores",
        },
        "deterministic_runtime": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        },
        "arms": {
            "legacy_cycle0": "exact latest contextual L4 linear path",
            "fixed_q_neural": "Q0 fixed; full neural residual trained",
            "alternating_q_neural": "neural phase then exact row-orthogonal Procrustes",
        },
        "first_phase": "shared once because both arms begin at identical model and Q0",
        "later_phases": "fixed and alternating receive the same epoch/batch budget",
        "model_phase_selection": "lowest full train MSE; validation never inspected by epoch",
        "cycle_selection": "validation common MEL80 all-bin mean Pearson r",
        "primary_contrast": "alternating_q_neural minus fixed_q_neural",
        "search_dim": SEARCH_DIM,
        "output_dim": OUTPUT_DIM,
        "max_cycles": args.max_cycles,
        "epochs_per_cycle": args.epochs_per_cycle,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "test_evaluation_in_this_command": False,
        "implementation_sha256": {
            "fit_runner": sha256_file(Path(__file__)),
            "core": sha256_file(MODULE_ROOT / "core.py"),
            "evaluator": sha256_file(MODULE_ROOT / "evaluate_frozen_sub01.py"),
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
        },
    }
    contract_fp = fingerprint_json(compatibility)
    contract = {**compatibility, "compatibility_fingerprint": contract_fp, "created_utc": _now()}
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        existing = read_json(contract_path)
        existing_compatibility = {
            key: value for key, value in existing.items()
            if key not in ("compatibility_fingerprint", "created_utc")
        }
        if fingerprint_json(existing_compatibility) != contract_fp:
            raise RuntimeError("existing run contract differs")
    else:
        atomic_write_json(contract_path, contract, overwrite=False)
    print("=" * 82, flush=True)
    print("Contextual neural E2E | SWPD development subject sub-01 | FIT ONLY", flush=True)
    print(f"seeds={list(seeds)} folds={list(folds)} device={args.device}", flush=True)
    print(
        f"cycles={args.max_cycles} epochs/cycle={args.epochs_per_cycle} "
        f"diagnostic={diagnostic}", flush=True,
    )
    print("TEST METRICS ARE NOT EVALUATED BY THIS COMMAND", flush=True)
    print("=" * 82, flush=True)

    selections: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            selections.append(_fit_fold_seed(
                fold=fold, seed=seed, cache=cache, output_root=run_dir,
                contract_fp=contract_fp, device=args.device,
                max_cycles=args.max_cycles,
                epochs_per_cycle=args.epochs_per_cycle,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                diagnostic=diagnostic,
            ))
    expected = len(seeds) * len(folds)
    if len(selections) != expected:
        raise RuntimeError("not all fit selections completed")
    summary = {
        "schema_version": 1,
        "kind": "swpd_sub01_contextual_neural_e2e_fit_summary",
        "run_contract_fingerprint": contract_fp,
        "development_only": True,
        "diagnostic_smoke": diagnostic,
        "seeds": list(seeds),
        "folds": list(folds),
        "selection_count": len(selections),
        "expected_selection_count": expected,
        "all_selections_frozen": True,
        "mean_legacy_validation_r": float(np.mean([
            item["legacy_cycle0"]["primary_r"] for item in selections
        ])),
        "mean_fixed_neural_validation_r": float(np.mean([
            item["fixed_selected_validation"]["primary_r"] for item in selections
        ])),
        "mean_alternating_neural_validation_r": float(np.mean([
            item["alternating_selected_validation"]["primary_r"] for item in selections
        ])),
        "mean_primary_validation_delta": float(np.mean([
            item["primary_validation_delta_alternating_minus_fixed"]
            for item in selections
        ])),
        "test_evaluated": False,
        "completed_utc": _now(),
    }
    atomic_write_json(run_dir / "fit_summary.json", summary)
    print(
        f"FIT COMPLETE | selections={len(selections)}/{expected} | "
        f"validation legacy={summary['mean_legacy_validation_r']:.6f} "
        f"fixed={summary['mean_fixed_neural_validation_r']:.6f} "
        f"alternating={summary['mean_alternating_neural_validation_r']:.6f} "
        f"primary delta={summary['mean_primary_validation_delta']:+.6f}",
        flush=True,
    )
    print("TEST NOT EVALUATED", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError, FloatingPointError, OSError, RuntimeError, ValueError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
