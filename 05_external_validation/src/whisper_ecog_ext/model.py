"""Dataset-independent one-second ECoG encoder used in external validation."""

from __future__ import annotations

import torch
from torch import nn


class OneSecondEcogEncoder(nn.Module):
    """Historical-compatible encoder with an explicit physical input contract.

    At the fixed defaults, an inclusive one-second 1000 Hz window has 1001
    samples. Temporal decimation produces 101 frames, each with 30 channels,
    hence the pre-projection hidden representation is exactly 3030D.
    """

    def __init__(
        self,
        input_channels: int,
        target_dim: int = 50,
        *,
        window_samples: int = 1001,
        hidden_channels: int = 30,
        temporal_stride: int = 10,
        filtering_kernel: int = 25,
        envelope_kernel: int = 15,
        use_lstm: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_channels", input_channels),
            ("target_dim", target_dim),
            ("window_samples", window_samples),
            ("hidden_channels", hidden_channels),
            ("temporal_stride", temporal_stride),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if filtering_kernel % 2 != 1 or envelope_kernel % 2 != 1:
            raise ValueError("temporal convolution kernels must be odd")
        if hidden_channels % 2 != 0 and use_lstm:
            raise ValueError("hidden_channels must be even for the bidirectional LSTM")

        self.input_channels = int(input_channels)
        self.target_dim = int(target_dim)
        self.window_samples = int(window_samples)
        self.hidden_channels = int(hidden_channels)
        self.temporal_stride = int(temporal_stride)
        self.filtering_kernel = int(filtering_kernel)
        self.envelope_kernel = int(envelope_kernel)
        self.use_lstm = bool(use_lstm)
        self.temporal_frames = (self.window_samples - 1) // self.temporal_stride + 1
        self.hidden_dim = self.temporal_frames * self.hidden_channels

        self.unmix = nn.Conv1d(self.input_channels, self.hidden_channels, kernel_size=1)
        self.unmix_norm = nn.BatchNorm1d(self.hidden_channels, affine=False)
        self.band_filter = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=self.filtering_kernel,
            padding=self.filtering_kernel // 2,
            groups=self.hidden_channels,
            bias=False,
        )
        self.band_norm = nn.BatchNorm1d(self.hidden_channels, affine=False)
        self.envelope_smoother = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=self.envelope_kernel,
            padding=self.envelope_kernel // 2,
            groups=self.hidden_channels,
        )
        self.temporal_model = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.hidden_norm = nn.BatchNorm1d(self.hidden_dim, affine=False)
        self.projection = nn.Linear(self.hidden_dim, self.target_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, channels, samples)")
        if int(inputs.shape[1]) != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} channels, got {int(inputs.shape[1])}"
            )
        if int(inputs.shape[2]) != self.window_samples:
            raise ValueError(
                f"expected exactly {self.window_samples} samples, got {int(inputs.shape[2])}"
            )

        values = self.unmix_norm(self.unmix(inputs))
        values = self.band_filter(values)
        values = torch.abs(self.band_norm(values))
        values = self.envelope_smoother(values)
        values = values[:, :, :: self.temporal_stride].contiguous()
        if self.use_lstm:
            values, _ = self.temporal_model(values.transpose(1, 2))
            values = values.transpose(1, 2)
        hidden = values.reshape(values.shape[0], -1)
        if int(hidden.shape[1]) != self.hidden_dim:
            raise RuntimeError(
                f"internal hidden dimension {int(hidden.shape[1])} != {self.hidden_dim}"
            )
        return self.hidden_norm(hidden)

    def forward(self, inputs: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor:
        hidden = self.encode(inputs)
        return hidden if return_hidden else self.projection(hidden)

    def architecture_receipt(self) -> dict:
        return {
            "kind": "one_second_ecog_encoder",
            "input_channels": self.input_channels,
            "target_dim": self.target_dim,
            "window_samples": self.window_samples,
            "canonical_sample_rate_hz": 1000,
            "inclusive_window": True,
            "hidden_channels": self.hidden_channels,
            "temporal_stride": self.temporal_stride,
            "filtering_kernel": self.filtering_kernel,
            "envelope_kernel": self.envelope_kernel,
            "temporal_frames": self.temporal_frames,
            "hidden_dim": self.hidden_dim,
            "use_lstm": self.use_lstm,
        }
