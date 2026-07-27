"""Fixed representation-independent neural preprocessing for VocalMind.

The transform has no fitted state and therefore cannot learn from validation or
test data.  Train-only channel centering/scaling remains a separate downstream
artifact in :mod:`whisper_ecog_ext.neural_data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class VocalMindNeuralPreprocessor:
    """CAR -> zero-phase high-pass/notches/low-pass on whole 3 s trials."""

    sample_rate_hz: int = 1000
    high_pass_hz: float = 10.0
    low_pass_hz: float = 200.0
    butterworth_order: int = 5
    notch_frequencies_hz: tuple[float, ...] = (50.0, 100.0, 150.0)
    notch_q: tuple[float, ...] = (25.0, 100.0, 150.0)
    expected_channel_count: int = 110

    def __post_init__(self) -> None:
        if int(self.sample_rate_hz) != 1000:
            raise ValueError("VocalMind primary neural rate is fixed to exactly 1000 Hz")
        if int(self.expected_channel_count) <= 1:
            raise ValueError("CAR requires at least two channels")
        if int(self.butterworth_order) <= 0:
            raise ValueError("butterworth_order must be positive")
        nyquist = float(self.sample_rate_hz) / 2.0
        if not 0 < float(self.high_pass_hz) < float(self.low_pass_hz) < nyquist:
            raise ValueError("band-pass limits must be strictly inside Nyquist")
        if len(self.notch_frequencies_hz) != len(self.notch_q) or not self.notch_frequencies_hz:
            raise ValueError("each notch frequency requires one Q value")
        if any(
            not np.isfinite(frequency)
            or not 0 < float(frequency) < nyquist
            or not np.isfinite(q)
            or float(q) <= 0
            for frequency, q in zip(self.notch_frequencies_hz, self.notch_q)
        ):
            raise ValueError("notch frequencies/Q values must be finite and admissible")

    @classmethod
    def from_mapping(
        cls,
        value: dict,
        *,
        expected_channel_count: int,
    ) -> "VocalMindNeuralPreprocessor":
        return cls(
            sample_rate_hz=int(value["sample_rate_hz"]),
            high_pass_hz=float(value["high_pass_hz"]),
            low_pass_hz=float(value["low_pass_hz"]),
            butterworth_order=int(value["butterworth_order"]),
            notch_frequencies_hz=tuple(float(item) for item in value["notch_frequencies_hz"]),
            notch_q=tuple(float(item) for item in value["notch_q"]),
            expected_channel_count=int(expected_channel_count),
        )

    def transform(self, trial: np.ndarray) -> np.ndarray:
        values = np.asarray(trial, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("neural trial must have shape (time, channels)")
        if int(values.shape[1]) != int(self.expected_channel_count):
            raise ValueError(
                f"expected {self.expected_channel_count} pinned clean channels, "
                f"got {values.shape[1]}"
            )
        if not np.isfinite(values).all():
            raise ValueError("neural trial contains NaN or Infinity")

        # Common-average reference is spatial and deterministic; no split statistic
        # or per-trial variance normalization is fitted here.
        filtered = values - values.mean(axis=1, keepdims=True, dtype=np.float64)
        high_pass = signal.butter(
            int(self.butterworth_order),
            float(self.high_pass_hz),
            btype="highpass",
            fs=int(self.sample_rate_hz),
            output="sos",
        )
        filtered = signal.sosfiltfilt(high_pass, filtered, axis=0)
        for frequency, q_value in zip(self.notch_frequencies_hz, self.notch_q):
            numerator, denominator = signal.iirnotch(
                float(frequency),
                Q=float(q_value),
                fs=int(self.sample_rate_hz),
            )
            filtered = signal.filtfilt(numerator, denominator, filtered, axis=0)
        low_pass = signal.butter(
            int(self.butterworth_order),
            float(self.low_pass_hz),
            btype="lowpass",
            fs=int(self.sample_rate_hz),
            output="sos",
        )
        filtered = signal.sosfiltfilt(low_pass, filtered, axis=0)
        if not np.isfinite(filtered).all():
            raise FloatingPointError("neural preprocessing produced NaN or Infinity")
        result = np.ascontiguousarray(filtered, dtype=np.float32)
        result.setflags(write=False)
        return result

    def transform_many(
        self,
        trials: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        return tuple(self.transform(trial) for trial in trials)

    def provenance(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "vocalmind_fixed_classic_neural_preprocessing",
            "input_rate_hz": int(self.sample_rate_hz),
            "output_rate_hz": int(self.sample_rate_hz),
            "resampling": "none_release_is_exactly_1000_hz",
            "input_channels": int(self.expected_channel_count),
            "channel_selection": "release_provided_110_clean_channels_pinned_order",
            "steps": [
                "common_average_reference_across_pinned_channels_per_sample",
                "zero_phase_butterworth_high_pass",
                "zero_phase_iir_notches",
                "zero_phase_butterworth_low_pass",
            ],
            "high_pass_hz": float(self.high_pass_hz),
            "low_pass_hz": float(self.low_pass_hz),
            "butterworth_order": int(self.butterworth_order),
            "notch_frequencies_hz": [float(value) for value in self.notch_frequencies_hz],
            "notch_q": [float(value) for value in self.notch_q],
            "filter_direction": "forward_backward_zero_phase",
            "per_trial_z_score": False,
            "fitted_parameters": False,
            "downstream_scaling": "train_only_channel_standardizer",
            "representation_independent": True,
            "official_vocalmind_reference_revision": (
                "e1202bab23cc8a2c944e5e13264b2ce0a37b2d03"
            ),
            "official_vocalmind_eeg_pipeline": (
                "CAR then HGA 70-150 Hz plus low-frequency <100 Hz, resampled to "
                "200 Hz, with per-trial z-score"
            ),
            "deliberate_primary_difference": (
                "this matched Whisper/MEL comparison keeps the historical project "
                "10-200 Hz 1000 Hz ECoG topology and replaces per-trial z-score with "
                "one train-only channel standardizer"
            ),
        }
