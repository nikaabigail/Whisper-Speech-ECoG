"""Ordered deterministic predictions with content-addressed evaluation receipts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .integrity import atomic_write_json, fingerprint_json
from .protocol import TestGateAuthorization, TestGateClosed
from .reproducibility import set_deterministic_seed
from .training import model_state_fingerprint


EvaluationTask = Literal["regression", "classification"]


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationResult:
    sample_ids: tuple[str, ...]
    predictions: np.ndarray
    targets: np.ndarray
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        for field_name in ("predictions", "targets"):
            value = np.array(getattr(self, field_name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)

    def save_receipt(self, path: Path) -> Path:
        atomic_write_json(Path(path), self.receipt, overwrite=False)
        return Path(path)


def _evaluate(
    model: nn.Module,
    dataset: Dataset,
    *,
    task: EvaluationTask,
    batch_size: int,
    device: str,
    training_config_fingerprint: str,
    evaluation_seed: int,
    test_gate_authorization: TestGateAuthorization | None,
) -> EvaluationResult:
    split_role = getattr(dataset, "split_role", None)
    if split_role not in {"train", "validation", "held_out_test"}:
        raise ValueError("evaluation dataset must declare an explicit split_role")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if training_config_fingerprint and (
        len(training_config_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in training_config_fingerprint)
    ):
        raise ValueError("training_config_fingerprint must be a lowercase SHA256")

    gate_receipt = None
    if split_role == "held_out_test":
        if not isinstance(test_gate_authorization, TestGateAuthorization):
            raise TestGateClosed(
                "held-out evaluation requires a validated TestGateAuthorization"
            )
        if getattr(dataset, "split_fingerprint", None) != (
            test_gate_authorization.split_fingerprint
        ):
            raise TestGateClosed("held-out dataset belongs to a different split")
        declared_ids = tuple(str(value) for value in getattr(dataset, "sample_ids", ()))
        if not declared_ids or fingerprint_json(list(declared_ids)) != (
            test_gate_authorization.held_out_sample_ids_sha256
        ):
            raise TestGateClosed(
                "held-out dataset sample IDs/order do not match the opened test split"
            )
        gate_receipt = {
            "split_fingerprint": test_gate_authorization.split_fingerprint,
            "protocol_fingerprint": test_gate_authorization.protocol_fingerprint,
            "open_receipt_fingerprint": (
                test_gate_authorization.open_receipt_fingerprint
            ),
        }

    set_deterministic_seed(evaluation_seed, strict_torch=True)

    runtime_device = torch.device(device)
    model.to(runtime_device)
    model.eval()
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    sample_ids: list[str] = []
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            if not {"inputs", "target", "sample_id"}.issubset(batch):
                raise ValueError("evaluation batches require inputs, target and sample_id")
            inputs = torch.as_tensor(batch["inputs"], dtype=torch.float32, device=runtime_device)
            if not torch.isfinite(inputs).all().item():
                raise FloatingPointError("evaluation inputs contain NaN or Infinity")
            outputs = model(inputs)
            if not torch.isfinite(outputs).all().item():
                raise FloatingPointError("evaluation outputs contain NaN or Infinity")
            batch_ids = tuple(str(value) for value in batch["sample_id"])
            sample_ids.extend(batch_ids)
            if task == "regression":
                targets = torch.as_tensor(batch["target"], dtype=torch.float32)
                if outputs.shape != targets.shape:
                    raise ValueError("regression prediction and target shapes differ")
                if not torch.isfinite(targets).all().item():
                    raise FloatingPointError("evaluation targets contain NaN or Infinity")
                prediction_batches.append(outputs.detach().cpu().float().numpy())
                target_batches.append(targets.cpu().float().numpy())
            else:
                targets = torch.as_tensor(batch["target"], dtype=torch.long)
                if targets.ndim != 1 or outputs.ndim != 2 or outputs.shape[0] != targets.shape[0]:
                    raise ValueError("classification logits/targets have invalid shapes")
                probabilities = torch.softmax(outputs, dim=1)
                prediction_batches.append(probabilities.detach().cpu().float().numpy())
                logits_batches.append(outputs.detach().cpu().float().numpy())
                target_batches.append(targets.cpu().long().numpy())

    ids = tuple(sample_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("evaluation sample IDs must be non-empty and unique")
    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    if len(predictions) != len(ids) or len(targets) != len(ids):
        raise RuntimeError("evaluation ordering/count contract failed")

    if task == "regression":
        residual = predictions.astype(np.float64) - targets.astype(np.float64)
        metrics = {
            "mse": float(np.mean(residual * residual)),
            "mae": float(np.mean(np.abs(residual))),
        }
    else:
        logits = np.concatenate(logits_batches, axis=0).astype(np.float64)
        labels = targets.astype(np.int64)
        if np.any(labels < 0) or np.any(labels >= predictions.shape[1]):
            raise ValueError("classification target is outside the model class range")
        chosen = predictions[np.arange(len(labels)), labels]
        metrics = {
            "accuracy": float(np.mean(np.argmax(predictions, axis=1) == labels)),
            "cross_entropy": float(-np.mean(np.log(np.maximum(chosen, 1e-300)))),
            "logits_sha256": _array_fingerprint(logits),
        }
    if not all(
        math.isfinite(value) for value in metrics.values() if isinstance(value, float)
    ):
        raise FloatingPointError("evaluation metrics are non-finite")

    receipt = {
        "schema_version": 1,
        "kind": "deterministic_ordered_prediction_receipt",
        "task": task,
        "split_role": split_role,
        "batch_size": int(batch_size),
        "device": str(runtime_device),
        "evaluation_seed": int(evaluation_seed),
        "sample_count": len(ids),
        "sample_ids_sha256": fingerprint_json(list(ids)),
        "predictions_sha256": _array_fingerprint(predictions),
        "targets_sha256": _array_fingerprint(targets),
        "model_state_sha256": model_state_fingerprint(model),
        "training_config_fingerprint": training_config_fingerprint or None,
        "metrics": metrics,
        "ordering": "dataset_order_no_shuffle",
        "test_gate_authorization": gate_receipt,
    }
    receipt["fingerprint"] = fingerprint_json(receipt)
    return EvaluationResult(ids, predictions, targets, receipt)


def evaluate_regression(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    training_config_fingerprint: str = "",
    evaluation_seed: int = 0,
    test_gate_authorization: TestGateAuthorization | None = None,
) -> EvaluationResult:
    return _evaluate(
        model,
        dataset,
        task="regression",
        batch_size=batch_size,
        device=device,
        training_config_fingerprint=training_config_fingerprint,
        evaluation_seed=evaluation_seed,
        test_gate_authorization=test_gate_authorization,
    )


def evaluate_hidden_classifier(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    training_config_fingerprint: str = "",
    evaluation_seed: int = 0,
    test_gate_authorization: TestGateAuthorization | None = None,
) -> EvaluationResult:
    return _evaluate(
        model,
        dataset,
        task="classification",
        batch_size=batch_size,
        device=device,
        training_config_fingerprint=training_config_fingerprint,
        evaluation_seed=evaluation_seed,
        test_gate_authorization=test_gate_authorization,
    )
