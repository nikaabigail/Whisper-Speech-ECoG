"""Deterministic, resumable supervised training selected only on validation loss."""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .integrity import fingerprint_json, sha256_file
from .reproducibility import (
    make_torch_generator,
    seed_dataloader_worker,
    set_deterministic_seed,
)


TaskKind = Literal["regression", "classification"]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 4
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    device: str = "cpu"
    strict_determinism: bool = True
    initialize_from_seed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        for name in ("max_epochs", "batch_size", "patience"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        for name in ("learning_rate", "grad_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "min_delta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TrainingResult:
    task: TaskKind
    completed: bool
    stopped_early: bool
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    history: tuple[dict[str, Any], ...]
    config_fingerprint: str
    selected_model_fingerprint: str
    checkpoint_path: Path
    checkpoint_sha256: str

    def receipt(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "deterministic_supervised_training_result",
            "task": self.task,
            "completed": self.completed,
            "stopped_early": self.stopped_early,
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "selected_by": "validation_loss_only",
            "history": list(self.history),
            "config_fingerprint": self.config_fingerprint,
            "selected_model_fingerprint": self.selected_model_fingerprint,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def model_state_fingerprint(model: nn.Module) -> str:
    """Hash state names, dtypes, shapes and exact bytes in stable key order."""

    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_signature(model: nn.Module) -> dict:
    receipt = getattr(model, "architecture_receipt", None)
    architecture = receipt() if callable(receipt) else None
    return {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "architecture": architecture,
        "parameters": {
            name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in sorted(model.named_parameters())
        },
    }


def _reset_parameters(model: nn.Module) -> None:
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def _validate_split_roles(train_dataset: Dataset, validation_dataset: Dataset) -> None:
    if getattr(train_dataset, "split_role", None) != "train":
        raise ValueError("training dataset must declare split_role='train'")
    if getattr(validation_dataset, "split_role", None) != "validation":
        raise ValueError("validation dataset must declare split_role='validation'")
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("training and validation datasets must be non-empty")


def _loader(
    dataset: Dataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
    epoch: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=make_torch_generator(config.seed + epoch),
        worker_init_fn=seed_dataloader_worker,
        persistent_workers=False,
        drop_last=False,
    )


def _finite_tensor(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{label} contains NaN or Infinity")


def _batch_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    task: TaskKind,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[torch.Tensor, int]:
    if "inputs" not in batch or "target" not in batch:
        raise ValueError("every batch must contain 'inputs' and 'target'")
    inputs = torch.as_tensor(batch["inputs"], dtype=torch.float32, device=device)
    _finite_tensor(inputs, "model inputs")
    outputs = model(inputs)
    _finite_tensor(outputs, "model outputs")
    if task == "regression":
        targets = torch.as_tensor(batch["target"], dtype=torch.float32, device=device)
        _finite_tensor(targets, "regression targets")
        if outputs.shape != targets.shape:
            raise ValueError(
                f"regression output/target shapes differ: {tuple(outputs.shape)} vs "
                f"{tuple(targets.shape)}"
            )
    else:
        targets = torch.as_tensor(batch["target"], dtype=torch.long, device=device)
        if targets.ndim != 1 or outputs.ndim != 2 or outputs.shape[0] != targets.shape[0]:
            raise ValueError("classification requires logits (batch, classes) and labels (batch,)")
        if torch.any(targets < 0).item() or torch.any(targets >= outputs.shape[1]).item():
            raise ValueError("classification target is outside the model class range")
    loss = criterion(outputs, targets)
    _finite_tensor(loss, f"{task} loss")
    return loss, int(inputs.shape[0])


def _run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    task: TaskKind,
    device: torch.device,
    criterion: nn.Module,
    grad_clip_norm: float,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss, batch_size = _batch_loss(
            model, batch, task=task, device=device, criterion=criterion
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"non-finite gradient at step {step}: {name}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_norm, error_if_nonfinite=True
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)).item():
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        loss_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("training loader yielded no samples")
    return loss_sum / sample_count


def _run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    task: TaskKind,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            loss, batch_size = _batch_loss(
                model, batch, task=task, device=device, criterion=criterion
            )
            loss_sum += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("validation loader yielded no samples")
    result = loss_sum / sample_count
    if not math.isfinite(result):
        raise FloatingPointError("validation loss is non-finite")
    return result


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _fit(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    task: TaskKind,
    config: TrainingConfig,
    checkpoint_path: Path,
    resume: bool,
    run_context: Mapping[str, Any] | None,
    max_epochs_this_call: int | None,
) -> TrainingResult:
    _validate_split_roles(train_dataset, validation_dataset)
    checkpoint_path = Path(checkpoint_path)
    if max_epochs_this_call is not None and int(max_epochs_this_call) <= 0:
        raise ValueError("max_epochs_this_call must be positive when supplied")
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")

    set_deterministic_seed(config.seed, strict_torch=config.strict_determinism)
    device = torch.device(config.device)
    loaded_checkpoint = None
    if resume:
        # Keep RNG byte tensors on the CPU. Loading the whole checkpoint onto
        # CUDA turns torch_cpu into a CUDA ByteTensor, which torch.set_rng_state
        # rejects. Model and optimizer tensors are moved explicitly below.
        loaded_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if loaded_checkpoint.get("kind") != "resumable_supervised_training_checkpoint":
            raise RuntimeError("unexpected checkpoint kind")
    if config.initialize_from_seed:
        initialization = {"kind": "reset_parameters_after_global_seed", "seed": config.seed}
    elif loaded_checkpoint is not None:
        initialization = loaded_checkpoint.get("config_payload", {}).get("initialization")
        if not isinstance(initialization, dict) or initialization.get("kind") != "caller_supplied":
            raise RuntimeError("checkpoint has no valid caller-supplied initialization receipt")
    else:
        initialization = {
            "kind": "caller_supplied",
            "state_sha256": model_state_fingerprint(model),
        }
    config_payload = {
        "schema_version": 1,
        "task": task,
        "training": asdict(config),
        "model": _model_signature(model),
        "initialization": initialization,
        "run_context": dict(run_context or {}),
    }
    config_fingerprint = fingerprint_json(config_payload)

    if config.initialize_from_seed and not resume:
        _reset_parameters(model)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion: nn.Module = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1
    completed = False
    stopped_early = False

    if resume:
        assert loaded_checkpoint is not None
        checkpoint = loaded_checkpoint
        if checkpoint.get("kind") != "resumable_supervised_training_checkpoint":
            raise RuntimeError("unexpected checkpoint kind")
        if checkpoint.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError("checkpoint configuration fingerprint mismatch")
        if checkpoint.get("task") != task:
            raise RuntimeError("checkpoint task mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        _restore_rng_state(checkpoint["rng_state"])
        history = list(checkpoint["history"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        best_model_state = checkpoint["best_model_state"]
        start_epoch = int(checkpoint["epoch_completed"]) + 1
        completed = bool(checkpoint["completed"])
        stopped_early = bool(checkpoint["stopped_early"])

    epochs_run_this_call = 0
    if not completed:
        for epoch in range(start_epoch, config.max_epochs + 1):
            train_loss = _run_training_epoch(
                model,
                _loader(train_dataset, config, shuffle=True, epoch=epoch),
                optimizer,
                task=task,
                device=device,
                criterion=criterion,
                grad_clip_norm=config.grad_clip_norm,
            )
            validation_loss = _run_validation_epoch(
                model,
                _loader(validation_dataset, config, shuffle=False, epoch=0),
                task=task,
                device=device,
                criterion=criterion,
            )
            improved = validation_loss < best_validation_loss - config.min_delta
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                stale_epochs = 0
                best_model_state = _cpu_state_dict(model)
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_improved": improved,
                }
            )
            stopped_early = stale_epochs >= config.patience
            completed = stopped_early or epoch >= config.max_epochs
            if best_model_state is None:
                raise RuntimeError("no finite validation-selected model was produced")
            checkpoint = {
                "schema_version": 1,
                "kind": "resumable_supervised_training_checkpoint",
                "task": task,
                "config_fingerprint": config_fingerprint,
                "config_payload": config_payload,
                "epoch_completed": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "stale_epochs": stale_epochs,
                "completed": completed,
                "stopped_early": stopped_early,
                "history": history,
                "model_state": _cpu_state_dict(model),
                "best_model_state": best_model_state,
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _capture_rng_state(),
            }
            _atomic_torch_save(checkpoint, checkpoint_path)
            epochs_run_this_call += 1
            if completed:
                break
            if (
                max_epochs_this_call is not None
                and epochs_run_this_call >= int(max_epochs_this_call)
            ):
                break

    if best_model_state is None:
        raise RuntimeError("checkpoint contains no validation-selected model")
    model.load_state_dict(best_model_state, strict=True)
    model.to(device)
    epochs_completed = int(history[-1]["epoch"]) if history else 0
    return TrainingResult(
        task=task,
        completed=completed,
        stopped_early=stopped_early,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        history=tuple(history),
        config_fingerprint=config_fingerprint,
        selected_model_fingerprint=model_state_fingerprint(model),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def train_regression(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    config: TrainingConfig,
    checkpoint_path: Path,
    resume: bool = False,
    run_context: Mapping[str, Any] | None = None,
    max_epochs_this_call: int | None = None,
) -> TrainingResult:
    return _fit(
        model,
        train_dataset,
        validation_dataset,
        task="regression",
        config=config,
        checkpoint_path=checkpoint_path,
        resume=resume,
        run_context=run_context,
        max_epochs_this_call=max_epochs_this_call,
    )


def train_hidden_classifier(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    config: TrainingConfig,
    checkpoint_path: Path,
    resume: bool = False,
    run_context: Mapping[str, Any] | None = None,
    max_epochs_this_call: int | None = None,
) -> TrainingResult:
    return _fit(
        model,
        train_dataset,
        validation_dataset,
        task="classification",
        config=config,
        checkpoint_path=checkpoint_path,
        resume=resume,
        run_context=run_context,
        max_epochs_this_call=max_epochs_this_call,
    )
