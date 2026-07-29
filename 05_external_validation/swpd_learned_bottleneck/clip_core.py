"""Contrastive 50D neural/target alignment for SWPD development experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ClipConfig:
    dimension: int = 50
    batch_size: int = 64
    maximum_epochs: int = 120
    patience: int = 18
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    minimum_negative_separation_seconds: float = 0.5
    covariance_weight: float = 0.02
    variance_weight: float = 0.05
    gradient_clip_norm: float = 5.0
    seed: int = 4

    def validate(self) -> None:
        if self.dimension <= 1 or self.batch_size < 4:
            raise ValueError("CLIP dimension and batch size are too small")
        if self.maximum_epochs <= 0 or self.patience <= 0:
            raise ValueError("Epoch and patience values must be positive")
        if not 0 < self.temperature <= 1:
            raise ValueError("temperature must be in (0, 1]")
        if self.minimum_negative_separation_seconds <= 0:
            raise ValueError("minimum negative separation must be positive")


class LinearClip(nn.Module):
    def __init__(
        self,
        neural_dimension: int,
        target_dimension: int,
        output_dimension: int,
        *,
        target_initial_projection: np.ndarray,
        neural_initial_weight: np.ndarray,
        neural_initial_bias: np.ndarray,
    ) -> None:
        super().__init__()
        expected_target = (target_dimension, output_dimension)
        expected_neural = (neural_dimension, output_dimension)
        if target_initial_projection.shape != expected_target:
            raise ValueError(f"target initialization must be {expected_target}")
        if neural_initial_weight.shape != expected_neural:
            raise ValueError(f"neural initialization must be {expected_neural}")
        self.target_projection = nn.Parameter(
            torch.as_tensor(target_initial_projection, dtype=torch.float32).clone()
        )
        self.neural_weight = nn.Parameter(
            torch.as_tensor(neural_initial_weight, dtype=torch.float32).clone()
        )
        self.neural_bias = nn.Parameter(
            torch.as_tensor(neural_initial_bias, dtype=torch.float32).clone()
        )

    def neural_embedding(self, neural: torch.Tensor) -> torch.Tensor:
        return neural @ self.neural_weight + self.neural_bias

    def target_embedding(self, target: torch.Tensor) -> torch.Tensor:
        return target @ self.target_projection

    @torch.no_grad()
    def retract_target_projection(self) -> None:
        q, r = torch.linalg.qr(self.target_projection, mode="reduced")
        diagonal = torch.diagonal(r)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        self.target_projection.copy_(q * signs.unsqueeze(0))


def parse_block_ids(sample_ids: Sequence[str]) -> np.ndarray:
    result = []
    for sample_id in sample_ids:
        fields = str(sample_id).split(":")
        block = next((field for field in fields if field.startswith("block-")), None)
        if block is None:
            raise ValueError(f"Sample ID has no block field: {sample_id}")
        result.append(int(block.split("-", 1)[1]))
    return np.asarray(result, dtype=np.int64)


def temporally_separated_indices(
    sample_ids: Sequence[str],
    frame_times_seconds: np.ndarray,
    *,
    minimum_separation_seconds: float,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Select an epoch subset whose same-block examples are temporally separated."""

    times = np.asarray(frame_times_seconds, dtype=np.float64)
    if times.shape != (len(sample_ids),):
        raise ValueError("One frame time is required per sample ID")
    blocks = parse_block_ids(sample_ids)
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(epoch))
    selected: list[int] = []
    for block in sorted(set(blocks.tolist())):
        indexes = np.flatnonzero(blocks == block)
        order = indexes[np.argsort(times[indexes])]
        if len(order) < 2:
            selected.extend(order.tolist())
            continue
        deltas = np.diff(times[order])
        positive = deltas[deltas > 0]
        if not positive.size:
            raise ValueError(f"Block {block} has no increasing frame times")
        stride = max(1, int(math.ceil(minimum_separation_seconds / np.median(positive))))
        offset = int(rng.integers(0, min(stride, len(order))))
        selected.extend(order[offset::stride].tolist())
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    # Independent validation of the scientific sampling contract.
    for block in sorted(set(blocks[result].tolist())):
        chosen_times = np.sort(times[result[blocks[result] == block]])
        if len(chosen_times) > 1 and np.min(np.diff(chosen_times)) + 1e-9 < minimum_separation_seconds:
            raise RuntimeError("Temporal-negative separation contract failed")
    return result


def batches(indexes: np.ndarray, batch_size: int) -> list[np.ndarray]:
    values = np.asarray(indexes, dtype=np.int64)
    result = [values[start : start + batch_size] for start in range(0, len(values), batch_size)]
    return [batch for batch in result if len(batch) >= 4]


def _off_diagonal(values: torch.Tensor) -> torch.Tensor:
    count = values.shape[0]
    return values.flatten()[:-1].view(count - 1, count + 1)[:, 1:].flatten()


def clip_loss(
    neural_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    config: ClipConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    neural = F.normalize(neural_embedding, dim=1, eps=1e-8)
    target = F.normalize(target_embedding, dim=1, eps=1e-8)
    logits = neural @ target.T / config.temperature
    labels = torch.arange(len(neural), device=neural.device)
    info_nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    desired_std = 0.5
    neural_std = torch.sqrt(neural_embedding.var(dim=0, unbiased=False) + 1e-4)
    target_std = torch.sqrt(target_embedding.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(desired_std - neural_std).mean() + F.relu(desired_std - target_std).mean()
    )
    neural_centered = neural - neural.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    neural_cov = neural_centered.T @ neural_centered / max(len(neural) - 1, 1)
    target_cov = target_centered.T @ target_centered / max(len(target) - 1, 1)
    covariance = (
        _off_diagonal(neural_cov).pow(2).mean()
        + _off_diagonal(target_cov).pow(2).mean()
    )
    total = info_nce + config.variance_weight * variance + config.covariance_weight * covariance
    return total, {
        "total": float(total.detach().cpu()),
        "info_nce": float(info_nce.detach().cpu()),
        "variance": float(variance.detach().cpu()),
        "covariance": float(covariance.detach().cpu()),
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def evaluate_loss(
    model: LinearClip,
    neural: np.ndarray,
    target: np.ndarray,
    evaluation_batches: Sequence[np.ndarray],
    config: ClipConfig,
    device: torch.device,
) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for index in evaluation_batches:
            x = torch.as_tensor(neural[index], dtype=torch.float32, device=device)
            y = torch.as_tensor(target[index], dtype=torch.float32, device=device)
            loss, _ = clip_loss(model.neural_embedding(x), model.target_embedding(y), config)
            values.append(float(loss.detach().cpu()))
    if not values:
        raise RuntimeError("No validation contrastive batches were available")
    return float(np.mean(values))


def train_clip(
    model: LinearClip,
    train_neural: np.ndarray,
    train_target: np.ndarray,
    train_ids: Sequence[str],
    train_times: np.ndarray,
    validation_neural: np.ndarray,
    validation_target: np.ndarray,
    validation_ids: Sequence[str],
    validation_times: np.ndarray,
    config: ClipConfig,
    device: torch.device,
    checkpoint_path: Path,
    fingerprint: str,
) -> tuple[LinearClip, dict[str, Any]]:
    config.validate()
    model.to(device)
    model.retract_target_projection()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_epoch = 1
    best_epoch = 0
    best_validation = float("inf")
    best_state = _cpu_state_dict(model)
    wait = 0
    history: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(f"CLIP checkpoint fingerprint mismatch: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        start_epoch = int(checkpoint["epoch_completed"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation = float(checkpoint["best_validation"])
        best_state = checkpoint["best_model_state"]
        wait = int(checkpoint["wait"])
        history = list(checkpoint["history"])
        print(f"    [resume] epoch={start_epoch} best={best_epoch}", flush=True)

    validation_indexes = temporally_separated_indices(
        validation_ids,
        validation_times,
        minimum_separation_seconds=config.minimum_negative_separation_seconds,
        seed=config.seed + 17,
        epoch=0,
    )
    validation_batches = batches(validation_indexes, config.batch_size)
    for epoch in range(start_epoch, config.maximum_epochs + 1):
        model.train()
        epoch_indexes = temporally_separated_indices(
            train_ids,
            train_times,
            minimum_separation_seconds=config.minimum_negative_separation_seconds,
            seed=config.seed,
            epoch=epoch,
        )
        train_batches = batches(epoch_indexes, config.batch_size)
        if not train_batches:
            raise RuntimeError("No training contrastive batches were available")
        train_losses = []
        for index in train_batches:
            x = torch.as_tensor(train_neural[index], dtype=torch.float32, device=device)
            y = torch.as_tensor(train_target[index], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = clip_loss(model.neural_embedding(x), model.target_embedding(y), config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite CLIP loss at epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            model.retract_target_projection()
            train_losses.append(details["total"])
        validation_loss = evaluate_loss(
            model,
            validation_neural,
            validation_target,
            validation_batches,
            config,
            device,
        )
        improved = validation_loss < best_validation - 1e-6
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            wait = 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "gradient_norm_last_batch": float(gradient_norm.detach().cpu()),
            "train_examples": int(sum(len(item) for item in train_batches)),
            "validation_examples": int(sum(len(item) for item in validation_batches)),
            "best": improved,
        }
        history.append(row)
        atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "epoch_completed": epoch,
                "best_epoch": best_epoch,
                "best_validation": best_validation,
                "wait": wait,
                "history": history,
                "model_state": _cpu_state_dict(model),
                "best_model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(config),
            },
        )
        flag = " BEST" if improved else f" wait={wait}"
        print(
            f"    [epoch {epoch:03d}] train={row['train_loss']:.4f} "
            f"val={validation_loss:.4f}{flag}",
            flush=True,
        )
        if wait >= config.patience:
            break
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "early_stopped": bool(history and history[-1]["epoch"] < config.maximum_epochs),
        "history": history,
    }


def embed(
    model: LinearClip,
    neural: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    neural_values = []
    target_values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(neural), batch_size):
            stop = start + batch_size
            x = torch.as_tensor(neural[start:stop], dtype=torch.float32, device=device)
            y = torch.as_tensor(target[start:stop], dtype=torch.float32, device=device)
            neural_values.append(F.normalize(model.neural_embedding(x), dim=1).cpu().numpy())
            target_values.append(F.normalize(model.target_embedding(y), dim=1).cpu().numpy())
    return np.concatenate(neural_values), np.concatenate(target_values)


def embedding_diagnostics(neural: np.ndarray, target: np.ndarray) -> dict[str, float]:
    def effective_rank(values: np.ndarray) -> float:
        singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
        probabilities = singular**2
        probabilities /= probabilities.sum()
        return float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))))

    positive = np.sum(neural * target, axis=1)
    shifted = np.sum(neural * np.roll(target, 1, axis=0), axis=1)
    return {
        "neural_effective_rank": effective_rank(neural),
        "target_effective_rank": effective_rank(target),
        "neural_min_component_sd": float(np.min(np.std(neural, axis=0))),
        "target_min_component_sd": float(np.min(np.std(target, axis=0))),
        "mean_positive_cosine": float(np.mean(positive)),
        "mean_shifted_negative_cosine": float(np.mean(shifted)),
        "cosine_margin": float(np.mean(positive) - np.mean(shifted)),
    }
