"""Modernized exact reproduction of the SWPD authors' MEL validation.

Scientific protocol follows the official MIT-licensed implementation pinned at
``cfb563696a8d44207532e3777ba6c5aabaf68805``:

* 70--170 Hz Hilbert envelope with 98--102 and 148--152 Hz band stops;
* 50 ms windows, 10 ms frame shift;
* nine neural contexts from -200 to +200 ms in 50 ms steps;
* non-shuffled 10-fold CV;
* fold-train z-normalization and PCA, first 50 components;
* ordinary least squares to the 23-bin author log-MEL target.

Only compatibility and numerical guards are modernized (``np.float`` and
``scipy.hanning`` replacements, deterministic null seed, safe constant-column
handling).  This module is development-only and hard-locks every subject other
than sub-01.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft as scipy_fft
from scipy import signal
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .nwb import PILOT_SUBJECT, SWPDRecording, assert_pilot_subject


WINDOW_SECONDS = 0.05
FRAME_SHIFT_SECONDS = 0.01
MODEL_ORDER = 4
CONTEXT_STEP_FRAMES = 5
MEL_BINS = 23
PCA_COMPONENTS = 50
CV_FOLDS = 10
AUTHOR_AUDIO_RATE_HZ = 48_000.0
TARGET_AUDIO_RATE_HZ = 16_000
MAX_AUDIO_RATE_RELATIVE_DEVIATION = 1e-3


@dataclass(frozen=True)
class AuthorFeatures:
    neural: np.ndarray
    mel: np.ndarray
    ieeg_rate_hz: float
    measured_audio_rate_hz: float
    author_processing_audio_rate_hz: float
    target_audio_rate_hz: float
    constant_audio: bool


class AuthorMelFilterBank:
    """The authors' triangular filter bank with removed NumPy deprecations."""

    def __init__(self, spectrum_size: int, coefficients: int, sample_rate: int):
        num_bands = int(coefficients)
        max_mel = self.freq_to_mel(sample_rate / 2.0)
        mel_step = max_mel / (num_bands + 1)
        edges = np.arange(num_bands + 2) * mel_step
        centers = [
            self.freq_to_bin(
                math.floor(self.mel_to_freq(value)), sample_rate, spectrum_size
            )
            for value in edges
        ]
        matrix = np.zeros((num_bands, spectrum_size), dtype=np.float64)
        for index in range(num_bands):
            start, center, end = centers[index : index + 3]
            up_width = float(center - start)
            down_width = float(end - center)
            if up_width > 0:
                matrix[index, start:center] = (
                    np.arange(start, center, dtype=np.float64) - start
                ) / up_width
            if down_width > 0:
                matrix[index, center:end] = (
                    end - np.arange(center, end, dtype=np.float64)
                ) / down_width
        mel_matrix = matrix.T
        normalizer = np.sum(mel_matrix, axis=0)
        normalizer[normalizer == 0] = 1.0
        self.mel_matrix = np.nan_to_num(mel_matrix / normalizer)

    @staticmethod
    def freq_to_bin(frequency: float, sample_rate: int, spectrum_size: int) -> int:
        return int(math.floor((frequency / (sample_rate / 2.0)) * spectrum_size))

    @staticmethod
    def freq_to_mel(frequency: float) -> float:
        return 2595.0 * math.log10(1.0 + frequency / 700.0)

    @staticmethod
    def mel_to_freq(mel: float) -> float:
        return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)

    def to_log_mels(self, spectrogram: np.ndarray) -> np.ndarray:
        mel = np.dot(spectrogram, self.mel_matrix)
        result = np.log(mel + 1e-7)
        result[~np.isfinite(result)] = 0.0
        return result


def _window_geometry(
    sample_count: int,
    sample_rate: float,
    window_seconds: float = WINDOW_SECONDS,
    frame_shift_seconds: float = FRAME_SHIFT_SECONDS,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    window = int(math.floor(window_seconds * sample_rate))
    shift = int(math.floor(frame_shift_seconds * sample_rate))
    if window <= 0 or shift <= 0 or sample_count <= window:
        raise ValueError("Signal is too short for the author window geometry")
    count = int(math.floor((sample_count - window_seconds * sample_rate) / (frame_shift_seconds * sample_rate)))
    starts = np.floor(np.arange(count) * frame_shift_seconds * sample_rate).astype(np.int64)
    stops = np.floor(starts + window_seconds * sample_rate).astype(np.int64)
    if stops[-1] > sample_count:
        raise RuntimeError("Author window geometry exceeds the signal")
    return window, shift, count, starts, stops


def extract_high_gamma(
    data: np.ndarray,
    sample_rate: float,
    *,
    window_seconds: float = WINDOW_SECONDS,
    frame_shift_seconds: float = FRAME_SHIFT_SECONDS,
) -> np.ndarray:
    """Extract author high-gamma features for one channel batch."""

    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("iEEG batch must be a finite samples x channels array")
    _, _, count, starts, stops = _window_geometry(
        array.shape[0], sample_rate, window_seconds, frame_shift_seconds
    )
    array = signal.detrend(array, axis=0)
    filters = (
        signal.iirfilter(
            4,
            [70 / (sample_rate / 2), 170 / (sample_rate / 2)],
            btype="bandpass",
            output="sos",
        ),
        signal.iirfilter(
            4,
            [98 / (sample_rate / 2), 102 / (sample_rate / 2)],
            btype="bandstop",
            output="sos",
        ),
        signal.iirfilter(
            4,
            [148 / (sample_rate / 2), 152 / (sample_rate / 2)],
            btype="bandstop",
            output="sos",
        ),
    )
    for second_order_sections in filters:
        array = signal.sosfiltfilt(second_order_sections, array, axis=0)
    analytic_length = scipy_fft.next_fast_len(array.shape[0])
    envelope = np.abs(signal.hilbert(array, analytic_length, axis=0)[: array.shape[0]])

    # Preserve the official per-window np.mean operation instead of replacing
    # it with a cumulative-sum approximation whose rounding differs slightly.
    features = np.empty((count, envelope.shape[1]), dtype=np.float64)
    for frame, (start, stop) in enumerate(zip(starts, stops)):
        features[frame] = np.mean(envelope[start:stop], axis=0)
    if features.shape[0] != count or not np.all(np.isfinite(features)):
        raise RuntimeError("Non-finite or misaligned author high-gamma features")
    return features


def stack_author_context(
    features: np.ndarray,
    *,
    model_order: int = MODEL_ORDER,
    step_frames: int = CONTEXT_STEP_FRAMES,
) -> np.ndarray:
    array = np.asarray(features)
    edge = model_order * step_frames
    if array.ndim != 2 or array.shape[0] <= 2 * edge:
        raise ValueError("Too few feature frames for the author temporal context")
    offsets = np.arange(-edge, edge + 1, step_frames)
    centers = np.arange(edge, array.shape[0] - edge)
    stacked = np.stack([array[centers + offset] for offset in offsets], axis=1)
    return stacked.reshape(len(centers), -1)


def extract_author_log_mel(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = WINDOW_SECONDS,
    frame_shift_seconds: float = FRAME_SHIFT_SECONDS,
    chunk_frames: int = 2048,
) -> np.ndarray:
    array = np.asarray(audio).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Audio contains NaN or Inf")
    window, _, count, starts, stops = _window_geometry(
        len(array), sample_rate, window_seconds, frame_shift_seconds
    )
    # Equivalent to the removed scipy.hanning(window + 1)[:-1] call.
    taper = signal.windows.hann(window + 1, sym=True)[:-1]
    spectrum_size = window // 2 + 1
    filter_bank = AuthorMelFilterBank(spectrum_size, MEL_BINS, sample_rate)
    result = np.empty((count, MEL_BINS), dtype=np.float64)
    for first in range(0, count, chunk_frames):
        last = min(first + chunk_frames, count)
        frames = np.stack(
            [array[starts[index] : stops[index]] for index in range(first, last)], axis=0
        )
        magnitude = np.abs(np.fft.rfft(frames * taper[None, :], axis=1))
        result[first:last] = filter_bank.to_log_mels(magnitude)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Author log-MEL extraction produced NaN or Inf")
    return result


def _scale_author_audio(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros(audio.shape, dtype=np.int16), True
    return np.asarray(audio / peak * 32767, dtype=np.int16), False


def extract_features_from_pilot(
    data_root: Path,
    *,
    channel_batch_size: int = 16,
) -> AuthorFeatures:
    if channel_batch_size <= 0:
        raise ValueError("channel_batch_size must be positive")
    with SWPDRecording(data_root, PILOT_SUBJECT) as recording:
        inventory = recording.inventory()
        ieeg_samples, channels = inventory.ieeg.shape
        ieeg_rate = inventory.ieeg.rate_hz
        if ieeg_rate <= 2 * 170:
            raise ValueError("iEEG sampling rate is too low for 170 Hz high gamma")

        batch_features: list[np.ndarray] = []
        for first in range(0, channels, channel_batch_size):
            last = min(first + channel_batch_size, channels)
            raw_batch = recording.read_ieeg(0, ieeg_samples, slice(first, last))
            batch_features.append(extract_high_gamma(raw_batch, ieeg_rate))
        neural_unstacked = np.concatenate(batch_features, axis=1)
        neural = stack_author_context(neural_unstacked)

        audio_samples = inventory.audio.shape[0]
        raw_audio = recording.read_audio(0, audio_samples).astype(np.float64, copy=False)
        measured_audio_rate = inventory.audio.rate_hz
        relative_rate_error = abs(measured_audio_rate - AUTHOR_AUDIO_RATE_HZ) / AUTHOR_AUDIO_RATE_HZ
        if relative_rate_error > MAX_AUDIO_RATE_RELATIVE_DEVIATION:
            raise ValueError(
                "Measured NWB audio rate differs too far from the authors' pinned "
                f"48 kHz processing assumption: {measured_audio_rate:.6f} Hz"
            )
        # The released NWB timestamp clock is approximately 47,999.19 Hz for
        # sub-01, while the official extract_features.py explicitly sets
        # audio_sr=48000 and decimates by 3.  Exact reproduction follows that
        # processing assumption and records both values in provenance.
        ratio = int(AUTHOR_AUDIO_RATE_HZ // TARGET_AUDIO_RATE_HZ)
        downsampled = signal.decimate(raw_audio, ratio)
        scaled, constant_audio = _scale_author_audio(downsampled)
        mel = extract_author_log_mel(scaled, TARGET_AUDIO_RATE_HZ)

    edge = MODEL_ORDER * CONTEXT_STEP_FRAMES
    mel = mel[edge : mel.shape[0] - edge]
    length = min(neural.shape[0], mel.shape[0])
    neural = neural[:length]
    mel = mel[:length]
    if length < CV_FOLDS or not np.all(np.isfinite(neural)) or not np.all(np.isfinite(mel)):
        raise RuntimeError("Author neural/MEL features are invalid after alignment")
    return AuthorFeatures(
        neural=neural,
        mel=mel,
        ieeg_rate_hz=ieeg_rate,
        measured_audio_rate_hz=measured_audio_rate,
        author_processing_audio_rate_hz=AUTHOR_AUDIO_RATE_HZ,
        target_audio_rate_hz=float(TARGET_AUDIO_RATE_HZ),
        constant_audio=constant_audio,
    )


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= np.finfo(np.float64).eps or np.std(right) <= np.finfo(np.float64).eps:
        return float("nan")
    return float(pearsonr(left, right).statistic)


def run_nonshuffled_cv(
    neural: np.ndarray,
    target: np.ndarray,
    *,
    folds: int = CV_FOLDS,
    pca_components: int = PCA_COMPONENTS,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(neural, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("neural and target must be aligned 2D arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("CV inputs contain NaN or Inf")
    splitter = KFold(n_splits=folds, shuffle=False)
    predictions = np.empty_like(y, dtype=np.float64)
    correlations = np.full((folds, y.shape[1]), np.nan, dtype=np.float64)
    explained = np.empty(folds, dtype=np.float64)
    constants_per_fold: list[int] = []

    for fold, (train, test) in enumerate(splitter.split(x)):
        mean = np.mean(x[train], axis=0)
        std = np.std(x[train], axis=0)
        constant = ~np.isfinite(std) | (std <= np.finfo(np.float64).eps)
        constants_per_fold.append(int(np.sum(constant)))
        safe_std = std.copy()
        safe_std[constant] = 1.0
        train_x = (x[train] - mean) / safe_std
        test_x = (x[test] - mean) / safe_std
        max_components = min(train_x.shape[0], train_x.shape[1])
        if pca_components > max_components:
            raise ValueError(
                f"pca_components={pca_components} exceeds fold limit {max_components}"
            )
        pca = PCA(n_components=pca_components, svd_solver="full")
        pca.fit(train_x)
        # The official code multiplies by the selected component matrix instead
        # of calling PCA.transform.  train_x has already been centered from the
        # fold-train mean, so this preserves those exact semantics.
        train_reduced = np.dot(train_x, pca.components_.T)
        test_reduced = np.dot(test_x, pca.components_.T)
        explained[fold] = float(np.sum(pca.explained_variance_ratio_))
        estimator = LinearRegression(n_jobs=1)
        estimator.fit(train_reduced, y[train])
        predictions[test] = estimator.predict(test_reduced)
        for component in range(y.shape[1]):
            correlations[fold, component] = _safe_pearson(
                y[test, component], predictions[test, component]
            )

    details = {
        "fold_component_correlations": correlations.tolist(),
        "fold_mean_correlations": np.nanmean(correlations, axis=1).tolist(),
        "fold_explained_variance": explained.tolist(),
        "constant_neural_columns_per_fold": constants_per_fold,
        "mean_correlation": float(np.nanmean(correlations)),
        "mean_explained_variance": float(np.mean(explained)),
    }
    return predictions, details


def circular_shift_null(
    target: np.ndarray,
    *,
    rounds: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    y = np.asarray(target, dtype=np.float64)
    lower = int(y.shape[0] * 0.1)
    upper = int(y.shape[0] * 0.9)
    if upper <= lower:
        raise ValueError("Target is too short for the author circular-shift null")
    rng = np.random.default_rng(seed)
    correlations = np.full((rounds, y.shape[1]), np.nan, dtype=np.float64)
    split_points = rng.integers(lower, upper, size=rounds)
    for round_index, split in enumerate(split_points):
        shifted = np.concatenate((y[split:], y[:split]), axis=0)
        for component in range(y.shape[1]):
            correlations[round_index, component] = _safe_pearson(
                y[:, component], shifted[:, component]
            )
    means = np.nanmean(correlations, axis=1)
    return {
        "rounds": rounds,
        "seed": seed,
        "mean_correlations": means.tolist(),
        "maximum_mean_correlation": float(np.nanmax(means)),
        "mean_of_mean_correlations": float(np.nanmean(means)),
    }
