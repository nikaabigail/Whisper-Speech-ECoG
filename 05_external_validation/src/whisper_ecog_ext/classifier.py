"""Historical hidden-trajectory word classifier with an explicit shape contract."""

from __future__ import annotations

import torch
from torch import nn


class HiddenSequenceClassifier(nn.Module):
    """Hidden features -> temporal convolution -> pool -> BiLSTM -> classes.

    Defaults reproduce the functional ``Mel2WordHidden`` topology used by the
    synchronous project: hidden features are already downsampled, a width-10
    convolution collapses the complete feature axis into 100 temporal channels,
    max-pooling uses width 10, and the two final 100D LSTM directions feed the
    classifier. The ignored one-layer LSTM dropout argument is deliberately 0.
    """

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        *,
        convolution_channels: int = 100,
        convolution_kernel: int = 10,
        pool_kernel: int = 10,
        lstm_hidden: int = 100,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_features", input_features),
            ("num_classes", num_classes),
            ("convolution_channels", convolution_channels),
            ("convolution_kernel", convolution_kernel),
            ("pool_kernel", pool_kernel),
            ("lstm_hidden", lstm_hidden),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least two")

        self.input_features = int(input_features)
        self.num_classes = int(num_classes)
        self.convolution_channels = int(convolution_channels)
        self.convolution_kernel = int(convolution_kernel)
        self.pool_kernel = int(pool_kernel)
        self.lstm_hidden = int(lstm_hidden)

        self.temporal_convolution = nn.Conv2d(
            1,
            self.convolution_channels,
            kernel_size=(self.input_features, self.convolution_kernel),
        )
        self.temporal_pool = nn.MaxPool1d(self.pool_kernel)
        self.temporal_lstm = nn.LSTM(
            input_size=self.convolution_channels,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.classifier = nn.Linear(2 * self.lstm_hidden, self.num_classes)

    @property
    def minimum_sequence_frames(self) -> int:
        return self.convolution_kernel + self.pool_kernel - 1

    def forward(self, hidden_sequence: torch.Tensor) -> torch.Tensor:
        if hidden_sequence.ndim != 3:
            raise ValueError(
                "hidden_sequence must have shape (batch, hidden_features, time)"
            )
        if int(hidden_sequence.shape[1]) != self.input_features:
            raise ValueError(
                f"expected {self.input_features} hidden features, "
                f"got {int(hidden_sequence.shape[1])}"
            )
        if int(hidden_sequence.shape[2]) < self.minimum_sequence_frames:
            raise ValueError(
                f"at least {self.minimum_sequence_frames} temporal frames are required"
            )

        convolved = self.temporal_convolution(hidden_sequence.unsqueeze(1)).squeeze(2)
        pooled = self.temporal_pool(convolved)
        _, (hidden, _) = self.temporal_lstm(pooled.transpose(1, 2))
        final_hidden = hidden.transpose(0, 1).reshape(hidden_sequence.shape[0], -1)
        return self.classifier(final_hidden)

    def architecture_receipt(self) -> dict:
        return {
            "kind": "historical_hidden_sequence_classifier",
            "input_layout": "batch_hidden_features_time",
            "input_features": self.input_features,
            "num_classes": self.num_classes,
            "pre_downsampling": 1,
            "convolution_channels": self.convolution_channels,
            "convolution_kernel": self.convolution_kernel,
            "pool": "max",
            "pool_kernel": self.pool_kernel,
            "lstm_hidden": self.lstm_hidden,
            "lstm_bidirectional": True,
            "lstm_layers": 1,
            "lstm_effective_dropout": 0.0,
        }
