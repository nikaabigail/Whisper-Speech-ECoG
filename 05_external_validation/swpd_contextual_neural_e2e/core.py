"""Neural and numerical core for contextual end-to-end alternating training.

The signal preprocessing and Whisper encoder are intentionally frozen.  The
trainable path starts at the already extracted 9 x channels high-gamma context
and ends at a constrained 50-dimensional Whisper-L4 target space.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from swpd_contextual_alternating_v2.core import (
    AffineMap,
    PCATransform,
    Standardizer,
    TargetSearchSpace,
    common_mel_metrics,
    exact_projector_update,
    fit_affine,
    mse,
    project_scores,
)
from swpd_protocol_bridge.bridge_core import component_metrics


def fold_legacy_pipeline(
    neural_pca: PCATransform,
    decoder: AffineMap,
) -> AffineMap:
    """Fold ``standardized neural -> PCA50 -> OLS`` into one affine map."""

    if neural_pca.whiten:
        raise ValueError("legacy neural PCA must use whiten=False")
    if decoder.coef.shape[1] != neural_pca.components.shape[0]:
        raise ValueError("legacy PCA and OLS geometries differ")
    weight = np.asarray(decoder.coef @ neural_pca.components, dtype=np.float64)
    bias = np.asarray(decoder.intercept - neural_pca.mean @ weight.T, dtype=np.float64)
    return AffineMap(weight, bias)


class ContextualResidualDecoder(nn.Module):
    """Trainable decoder from a 9-step high-gamma context to 50 target scores.

    ``skip`` is initialized with the exact fold-specific PCA50/OLS solution.
    The residual output starts at zero, so before training the whole network is
    numerically identical to that solution.  Every parameter remains trainable.
    """

    def __init__(
        self,
        context_steps: int = 9,
        channels: int = 127,
        output_dim: int = 50,
        spatial_dim: int = 64,
        recurrent_dim: int = 64,
    ) -> None:
        super().__init__()
        if context_steps < 3 or context_steps % 2 != 1:
            raise ValueError("context_steps must be an odd integer >= 3")
        if min(channels, output_dim, spatial_dim, recurrent_dim) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.context_steps = int(context_steps)
        self.channels = int(channels)
        self.output_dim = int(output_dim)
        self.spatial_dim = int(spatial_dim)
        self.recurrent_dim = int(recurrent_dim)
        self.skip = nn.Linear(self.context_steps * self.channels, self.output_dim)
        self.spatial = nn.Linear(self.channels, self.spatial_dim)
        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.temporal_conv = nn.Conv1d(
            self.spatial_dim, self.spatial_dim, kernel_size=3, padding=1
        )
        self.temporal_norm = nn.LayerNorm(self.spatial_dim)
        self.temporal = nn.GRU(
            input_size=self.spatial_dim,
            hidden_size=self.recurrent_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.residual_hidden = nn.Linear(2 * self.recurrent_dim, self.recurrent_dim)
        self.residual_output = nn.Linear(self.recurrent_dim, self.output_dim)

    def initialize_legacy_skip(
        self,
        weight: np.ndarray | torch.Tensor,
        bias: np.ndarray | torch.Tensor,
    ) -> None:
        expected_weight = (self.output_dim, self.context_steps * self.channels)
        expected_bias = (self.output_dim,)
        weight_tensor = torch.as_tensor(weight, dtype=self.skip.weight.dtype)
        bias_tensor = torch.as_tensor(bias, dtype=self.skip.bias.dtype)
        if tuple(weight_tensor.shape) != expected_weight or tuple(bias_tensor.shape) != expected_bias:
            raise ValueError("legacy skip geometry changed")
        with torch.no_grad():
            self.skip.weight.copy_(weight_tensor)
            self.skip.bias.copy_(bias_tensor)
            self.residual_output.weight.zero_()
            self.residual_output.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            if tuple(inputs.shape[1:]) != (self.context_steps, self.channels):
                raise ValueError("contextual decoder sequence geometry changed")
            sequence = inputs
            flattened = inputs.reshape(inputs.shape[0], -1)
        elif inputs.ndim == 2:
            if inputs.shape[1] != self.context_steps * self.channels:
                raise ValueError("contextual decoder flat geometry changed")
            flattened = inputs
            sequence = inputs.reshape(-1, self.context_steps, self.channels)
        else:
            raise ValueError("contextual decoder input must be a matrix or sequence batch")
        shortcut = self.skip(flattened)
        sequence = F.gelu(self.spatial_norm(self.spatial(sequence)))
        convolved = self.temporal_conv(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = F.gelu(self.temporal_norm(convolved))
        sequence, _ = self.temporal(sequence)
        center = sequence[:, self.context_steps // 2]
        residual = F.gelu(self.residual_hidden(center))
        return shortcut + self.residual_output(residual)

    def architecture_receipt(self) -> dict[str, Any]:
        return {
            "kind": "contextual_residual_ecog_decoder",
            "context_steps": self.context_steps,
            "channels": self.channels,
            "input_dim": self.context_steps * self.channels,
            "output_dim": self.output_dim,
            "spatial_dim": self.spatial_dim,
            "recurrent": "one-layer bidirectional GRU",
            "recurrent_dim_per_direction": self.recurrent_dim,
            "pooling": "central context state",
            "normalization": "LayerNorm only; no BatchNorm",
            "dropout": 0.0,
            "batch_norm": False,
            "legacy_skip_trainable": True,
            "residual_output_zero_initialized": True,
        }


def clone_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value, device="cpu", dtype=torch.uint8) for value in state["torch_cuda"]]
        )


def projector_receipt(whitened_train: np.ndarray, projector: np.ndarray) -> dict[str, Any]:
    targets = project_scores(whitened_train, projector)
    q = np.asarray(projector, dtype=np.float64)
    variance = np.var(targets, axis=0, ddof=1)
    return {
        "shape": list(q.shape),
        "rank": int(np.linalg.matrix_rank(q, tol=1e-10)),
        "fro_norm": float(np.linalg.norm(q, ord="fro")),
        "orthogonality_fro_error": float(
            np.linalg.norm(q @ q.T - np.eye(q.shape[0]), ord="fro")
        ),
        "projected_train_variance_min": float(variance.min()),
        "projected_train_variance_max": float(variance.max()),
        "projected_train_variance_mean": float(variance.mean()),
        "zero_variance_collapse": bool(np.any(variance < 1e-6)),
    }


def score_and_full_target_metrics(
    target_space: TargetSearchSpace,
    projector: np.ndarray,
    raw_truth: np.ndarray,
    predicted_scores: np.ndarray,
) -> dict[str, Any]:
    true_scores = target_space.scores(raw_truth, projector)
    reconstructed = target_space.reconstruct_standardized(predicted_scores, projector)
    standardized_truth = target_space.scaler.transform(raw_truth)
    return {
        "score50": component_metrics(true_scores, predicted_scores),
        "score50_mse": mse(true_scores, predicted_scores),
        "full_l4_512": component_metrics(standardized_truth, reconstructed),
        "full_l4_512_mse": mse(standardized_truth, reconstructed),
    }


__all__ = [
    "AffineMap",
    "ContextualResidualDecoder",
    "PCATransform",
    "Standardizer",
    "TargetSearchSpace",
    "capture_rng_state",
    "clone_state_dict_cpu",
    "common_mel_metrics",
    "exact_projector_update",
    "fit_affine",
    "fold_legacy_pipeline",
    "mse",
    "project_scores",
    "projector_receipt",
    "restore_rng_state",
    "score_and_full_target_metrics",
    "state_dict_sha256",
]
