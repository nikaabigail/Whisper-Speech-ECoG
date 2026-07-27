"""VocalMind author-compatible 80-bin log-amplitude MEL target."""

from __future__ import annotations

import numpy as np

from .targets import _peak_normalize_audio, align_features_local, resample_audio_polyphase


VOCALMIND_CODE_REPOSITORY = "https://github.com/tianyu-h42/sEEG-Processing-SpeechDecoding"
VOCALMIND_CODE_REVISION = "e1202bab23cc8a2c944e5e13264b2ce0a37b2d03"


class VocalMindAuthorMelTargetExtractor:
    """Port the author's spectral parameters with an explicit amplitude policy.

    The original code computes an STFT magnitude (not power), projects it with
    the librosa MEL basis, and applies base-10 log clipping at ``1e-6``.
    ``librosa.stft`` uses centered constant padding in the pinned implementation.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        window: str = "hann",
        n_mels: int = 80,
        fmin_hz: float = 80.0,
        fmax_hz: float = 7600.0,
        epsilon: float = 1e-6,
        peak_normalize: bool = False,
    ) -> None:
        if (
            int(sample_rate_hz) != 16_000
            or int(n_fft) != 1024
            or int(hop_length) != 320
            or int(win_length) != 1024
            or str(window) != "hann"
            or int(n_mels) != 80
            or float(fmin_hz) != 80.0
            or float(fmax_hz) != 7600.0
            or float(epsilon) != 1e-6
        ):
            raise ValueError("VocalMind author-MEL parameters are pinned")
        self.sample_rate_hz = int(sample_rate_hz)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.window = str(window)
        self.n_mels = int(n_mels)
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self.epsilon = float(epsilon)
        self.peak_normalize = bool(peak_normalize)

    def extract_native(
        self,
        audio_mono: np.ndarray,
        sample_rate_hz: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        source = (
            _peak_normalize_audio(audio_mono)
            if self.peak_normalize
            else np.asarray(audio_mono, dtype=np.float32)
        )
        waveform = resample_audio_polyphase(
            source,
            int(sample_rate_hz),
            self.sample_rate_hz,
        )
        spectrum = librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="constant",
        )
        magnitude = np.abs(spectrum)
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate_hz,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin_hz,
            fmax=self.fmax_hz,
        )
        features = np.log10(np.maximum(self.epsilon, mel_basis @ magnitude)).T
        times = (
            np.arange(features.shape[0], dtype=np.float64)
            * self.hop_length
            / self.sample_rate_hz
        )
        result = np.ascontiguousarray(features, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != self.n_mels:
            raise RuntimeError(f"unexpected author-MEL shape: {result.shape}")
        if not np.isfinite(result).all():
            raise FloatingPointError("author-MEL contains NaN or Infinity")
        return times, result

    def extract_aligned(
        self,
        audio_mono: np.ndarray,
        sample_rate_hz: int,
        target_times_s: np.ndarray,
    ) -> np.ndarray:
        times, features = self.extract_native(audio_mono, sample_rate_hz)
        return align_features_local(times, features, target_times_s)

    def provenance(self) -> dict:
        return {
            "kind": (
                "vocalmind_author_spectral_parameters_shared_peak_normalization"
                if self.peak_normalize
                else "vocalmind_author_exact_no_peak_normalization"
            ),
            "source_repository": VOCALMIND_CODE_REPOSITORY,
            "source_revision": VOCALMIND_CODE_REVISION,
            "source_function": "src/audio_preproc.py:librosa_wav2spec",
            "sample_rate_hz": self.sample_rate_hz,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "frame_step_ms": self.hop_length / self.sample_rate_hz * 1000.0,
            "win_length": self.win_length,
            "window_ms": self.win_length / self.sample_rate_hz * 1000.0,
            "window": self.window,
            "center": True,
            "pad_mode": "constant",
            "spectrum": "magnitude_not_power",
            "n_mels": self.n_mels,
            "fmin_hz": self.fmin_hz,
            "fmax_hz": self.fmax_hz,
            "mel_basis": "librosa_default_slaney",
            "log": "log10",
            "epsilon": self.epsilon,
            "waveform_normalization": (
                "per_trial_peak_abs_epsilon_1e-8_shared_with_whisper"
                if self.peak_normalize
                else "none_official_fidelity"
            ),
            "official_fidelity_difference": bool(self.peak_normalize),
            "rep6_resampling_sensitivity_audit": {
                "comparison": "polyphase_vs_official_librosa_resample",
                "pearson_correlation": 0.999978,
                "mean_absolute_difference": 0.00121,
                "maximum_absolute_difference": 0.0849,
                "classification_metric": False,
            },
            "alignment_to_regression_grid": "neighboring_frame_linear_interpolation",
        }
