"""Local-time MEL and pinned Whisper-base L3/L4/L5 acoustic targets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.signal


WHISPER_SAMPLE_RATE = 16_000
WHISPER_CHUNK_SECONDS = 30
WHISPER_CHUNK_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SECONDS
DEFAULT_LAYERS = (3, 4, 5)
_PINNED_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


def _mono_finite_audio(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("audio must be a non-empty mono waveform")
    if not np.isfinite(values).all():
        raise ValueError("audio contains NaN or Infinity")
    return values


def _peak_normalize_audio(audio: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    values = _mono_finite_audio(audio)
    peak = float(np.max(np.abs(values)))
    if peak <= float(epsilon):
        return np.zeros_like(values)
    return np.asarray(values / peak, dtype=np.float32)


def resample_audio_polyphase(
    audio: np.ndarray, original_rate: int, target_rate: int
) -> np.ndarray:
    values = _mono_finite_audio(audio)
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if int(original_rate) == int(target_rate):
        return values.copy()
    common = math.gcd(int(original_rate), int(target_rate))
    resampled = scipy.signal.resample_poly(
        values,
        up=int(target_rate) // common,
        down=int(original_rate) // common,
    )
    return np.asarray(resampled, dtype=np.float32)


def align_features_local(
    source_times_s: np.ndarray,
    source_features: np.ndarray,
    target_times_s: np.ndarray,
    *,
    max_edge_hold_s: float = 0.0,
) -> np.ndarray:
    """Align with only the two neighboring source frames, never a global FFT."""

    source_times = np.asarray(source_times_s, dtype=np.float64)
    features = np.asarray(source_features, dtype=np.float32)
    target_times = np.asarray(target_times_s, dtype=np.float64)
    if source_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError("source and target times must be one-dimensional")
    if not np.isfinite(max_edge_hold_s) or float(max_edge_hold_s) < 0:
        raise ValueError("max_edge_hold_s must be finite and non-negative")
    if features.ndim != 2 or features.shape[0] != source_times.shape[0]:
        raise ValueError("source_features must have shape (source_times, features)")
    if source_times.size == 0:
        raise ValueError("at least one source frame is required")
    if not (np.isfinite(source_times).all() and np.isfinite(target_times).all()):
        raise ValueError("frame times must be finite")
    if source_times.size > 1 and np.any(np.diff(source_times) <= 0):
        raise ValueError("source frame times must be strictly increasing")
    if target_times.size > 1 and np.any(np.diff(target_times) < 0):
        raise ValueError("target frame times must be sorted")
    if target_times.size == 0:
        return np.empty((0, features.shape[1]), dtype=np.float32)
    tolerance = float(max_edge_hold_s) + 1e-9
    if (
        target_times[0] < source_times[0] - tolerance
        or target_times[-1] > source_times[-1] + tolerance
    ):
        raise ValueError(
            "target frame lies outside the source timeline; edge padding is forbidden"
        )
    if source_times.size == 1:
        return np.repeat(features, target_times.size, axis=0)

    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 1, source_times.size - 1)
    left = right - 1
    denominator = source_times[right] - source_times[left]
    weight = np.clip((target_times - source_times[left]) / denominator, 0.0, 1.0)
    aligned = features[left] * (1.0 - weight[:, None]) + features[right] * weight[:, None]
    return np.asarray(aligned, dtype=np.float32)


@dataclass(frozen=True)
class LayerFeatureTimeline:
    frame_times_s: np.ndarray
    by_layer: Mapping[int, np.ndarray]


class WhisperLayerTargetExtractor:
    """Extract Whisper-base L3/L4/L5 together from each 30-second chunk."""

    def __init__(
        self,
        *,
        revision: str,
        model_name: str = "openai/whisper-base",
        layers: Sequence[int] = DEFAULT_LAYERS,
        device: str | None = None,
        encoder=None,
        feature_extractor=None,
        peak_normalize: bool = True,
    ) -> None:
        if not _PINNED_REVISION.fullmatch(str(revision)):
            raise ValueError("revision must be an exact 40-character Hugging Face commit SHA")
        normalized_layers = tuple(sorted(set(int(item) for item in layers)))
        if normalized_layers != DEFAULT_LAYERS:
            raise ValueError("the external-validation target is fixed to layers (3, 4, 5)")
        if (encoder is None) != (feature_extractor is None):
            raise ValueError("encoder and feature_extractor must be supplied together")

        import torch

        self.revision = str(revision).lower()
        self.model_name = str(model_name)
        self.layers = normalized_layers
        self.peak_normalize = bool(peak_normalize)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if encoder is None:
            from transformers import WhisperFeatureExtractor, WhisperModel

            feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_name, revision=self.revision
            )
            encoder = WhisperModel.from_pretrained(
                self.model_name, revision=self.revision
            ).encoder
        self.feature_extractor = feature_extractor
        self.encoder = encoder.eval().to(self.device)
        if hasattr(self.encoder, "requires_grad_"):
            self.encoder.requires_grad_(False)

    def provenance(self) -> dict:
        return {
            "kind": "whisper_encoder_hidden_states",
            "model_name": self.model_name,
            "revision": self.revision,
            "layers": list(self.layers),
            "sample_rate": WHISPER_SAMPLE_RATE,
            "chunk_seconds": WHISPER_CHUNK_SECONDS,
            "peak_normalize": self.peak_normalize,
            "alignment": "neighboring-frame linear interpolation",
        }

    def extract_native(self, audio: np.ndarray, sample_rate: int) -> LayerFeatureTimeline:
        import torch

        source_audio = (
            _peak_normalize_audio(audio) if self.peak_normalize else _mono_finite_audio(audio)
        )
        audio16 = resample_audio_polyphase(
            source_audio, int(sample_rate), WHISPER_SAMPLE_RATE
        )
        layer_chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in self.layers}
        time_chunks: list[np.ndarray] = []
        for start in range(0, audio16.size, WHISPER_CHUNK_SAMPLES):
            chunk = audio16[start : start + WHISPER_CHUNK_SAMPLES]
            batch = self.feature_extractor(
                chunk,
                sampling_rate=WHISPER_SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
                max_length=WHISPER_CHUNK_SAMPLES,
                truncation=True,
            )
            input_features = batch.input_features.to(self.device)
            with torch.inference_mode():
                result = self.encoder(
                    input_features, output_hidden_states=True, return_dict=True
                )
            hidden_states = result.hidden_states
            if len(hidden_states) <= max(self.layers):
                raise RuntimeError(
                    f"Whisper returned {len(hidden_states)} hidden-state tensors; "
                    f"layer {max(self.layers)} is unavailable"
                )
            output_frames = int(hidden_states[self.layers[0]].shape[1])
            frames_per_second = output_frames / WHISPER_CHUNK_SECONDS
            real_frames = min(
                output_frames,
                max(1, int(round(chunk.size / WHISPER_SAMPLE_RATE * frames_per_second))),
            )
            chunk_start_s = start / WHISPER_SAMPLE_RATE
            times = chunk_start_s + (np.arange(real_frames, dtype=np.float64) + 0.5) / frames_per_second
            time_chunks.append(times)
            for layer in self.layers:
                tensor = hidden_states[layer]
                if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                    raise RuntimeError(f"unexpected Whisper L{layer} shape: {tuple(tensor.shape)}")
                if int(tensor.shape[1]) != output_frames:
                    raise RuntimeError("Whisper layer timelines are not aligned")
                layer_chunks[layer].append(
                    tensor[0, :real_frames].detach().float().cpu().numpy().astype(np.float32)
                )

        frame_times = np.concatenate(time_chunks)
        return LayerFeatureTimeline(
            frame_times_s=frame_times,
            by_layer={layer: np.concatenate(layer_chunks[layer], axis=0) for layer in self.layers},
        )

    def extract_aligned(
        self, audio: np.ndarray, sample_rate: int, target_times_s: np.ndarray
    ) -> dict[int, np.ndarray]:
        native = self.extract_native(audio, sample_rate)
        return {
            layer: align_features_local(
                native.frame_times_s,
                values,
                target_times_s,
                max_edge_hold_s=0.5 / 50.0,
            )
            for layer, values in native.by_layer.items()
        }


class MelTargetExtractor:
    """Local log-MEL target on an explicit frame-time grid."""

    def __init__(
        self,
        *,
        n_mels: int = 80,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        frame_hz: float = 50.0,
        n_fft: int = 400,
        win_length: int = 400,
        fmin: float = 0.0,
        fmax: float = 8_000.0,
        epsilon: float = 1e-10,
        peak_normalize: bool = True,
    ) -> None:
        self.n_mels = int(n_mels)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(round(self.sample_rate / float(frame_hz)))
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.epsilon = float(epsilon)
        self.peak_normalize = bool(peak_normalize)
        if not (self.n_mels > 0 and self.hop_length > 0 and self.n_fft >= self.win_length > 0):
            raise ValueError("invalid MEL dimensions")
        if not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("invalid MEL frequency range")

    def extract_native(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        source_audio = (
            _peak_normalize_audio(audio) if self.peak_normalize else _mono_finite_audio(audio)
        )
        values = resample_audio_polyphase(
            source_audio, int(sample_rate), self.sample_rate
        )
        if values.size < self.n_fft:
            values = np.pad(values, (0, self.n_fft - values.size))
        power = librosa.feature.melspectrogram(
            y=values,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            power=2.0,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        features = np.log(np.maximum(power.T, self.epsilon)).astype(np.float32)
        times = (
            np.arange(features.shape[0], dtype=np.float64) * self.hop_length
            + self.win_length / 2.0
        ) / self.sample_rate
        return times, features

    def extract_aligned(
        self, audio: np.ndarray, sample_rate: int, target_times_s: np.ndarray
    ) -> np.ndarray:
        times, features = self.extract_native(audio, sample_rate)
        return align_features_local(
            times,
            features,
            target_times_s,
            max_edge_hold_s=0.5 * self.hop_length / self.sample_rate,
        )

    def provenance(self) -> dict:
        return {
            "kind": "log_mel_power",
            "n_mels": self.n_mels,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "center": False,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "peak_normalize": self.peak_normalize,
            "alignment": "neighboring-frame linear interpolation",
        }
