"""Leakage-controlled VocalMind primary overt-word orchestration.

This module is deliberately dataset-specific and thin: model fitting, immutable
split gates, target reducers, neural standardization, evaluation, and the fixed
probability ensemble are delegated to the public common-core interfaces.

The held-out repetition is never numerically loaded until every configured seed
has validation-fixed MEL, L3, L4, and L5 regression/classification artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .classifier import HiddenSequenceClassifier
from .data.vocalmind import (
    DEFAULT_VOCALMIND_CONTRACT,
    VocalMindAdapter,
    VocalMindContract,
    build_primary_split_manifest,
)
from .ensemble import LayerProbabilities, fixed_l345_probability_ensemble
from .evaluation import EvaluationResult, evaluate_hidden_classifier
from .integrity import atomic_write_json, fingerprint_json, read_json, sha256_file
from .model import OneSecondEcogEncoder
from .neural_data import (
    ChannelStandardizerArtifact,
    FrameWindowDataset,
    fit_train_only_channel_standardizer,
)
from .protocol import SplitManifest, TestGate
from .reducer import ReducerArtifact, fit_train_only_reducer
from .source_identity import capture_source_identity, require_clean_frozen_source
from .targets import WhisperLayerTargetExtractor
from .training import TrainingConfig, TrainingResult, train_hidden_classifier, train_regression
from .vocalmind_neural import VocalMindNeuralPreprocessor
from .vocalmind_targets import (
    VOCALMIND_CODE_REVISION,
    VocalMindAuthorMelTargetExtractor,
)


CONFIG_SCHEMA_VERSION = 2
RUNNER_VERSION = "vocalmind_primary_overt_v3_frozen_confirmatory"
REPRESENTATIONS = ("mel", "L3", "L4", "L5")
WHISPER_LAYERS = (3, 4, 5)
PINNED_WHISPER_MODEL = "openai/whisper-base"
PINNED_WHISPER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
PRIMARY_SEEDS = (1, 2, 3, 4, 42)
PRIMARY_FOLDS = (1, 2, 3, 4, 5)
REP6_DEVELOPMENT_AUDIT = {
    "development_only_rep6": True,
    "allowed_use": "adapter_audio_timing_memory_shape_smoke_only",
    "fit_allowed": False,
    "reported_model_metric_allowed": False,
    "audited_trial_count": 19,
    "crude_20ms_rms_onset_s_min_median_max": [0.66, 1.04, 1.30],
    "crude_20ms_rms_offset_s_min_median_max": [1.08, 1.90, 2.24],
    "boundary_limitation": (
        "valid lag-window endpoints start at 1.000 s; onset context is retained, but the "
        "earliest audited word ends near 1.080 s, leaving few post-onset endpoints; "
        "no padding is permitted"
    ),
}


class PrimaryConfigError(ValueError):
    """A primary-experiment config violates the frozen comparison contract."""


class RunIncomplete(RuntimeError):
    """A resumable bounded training call stopped before the fold was fixed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrimaryConfigError(f"{label} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    missing = sorted(required_set - set(value))
    unexpected = sorted(set(value) - required_set - optional_set)
    if missing or unexpected:
        raise PrimaryConfigError(
            f"{label} keys differ from the schema; missing={missing}, unexpected={unexpected}"
        )


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise PrimaryConfigError(f"{label} must be {expected!r}, got {value!r}")


@dataclass(frozen=True)
class PrimaryConfig:
    """Validated immutable view of a JSON experiment configuration."""

    payload: Mapping[str, Any]
    fingerprint: str

    @property
    def run_scope(self) -> str:
        return str(self.payload["run_scope"])

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["folds"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["seeds"])

    @property
    def target_dim(self) -> int:
        return int(self.payload["targets"]["target_dim"])

    @property
    def frame_step_samples(self) -> int:
        grid = self.payload["frame_grid"]
        return int(grid["neural_sample_rate_hz"] // grid["regression_frame_hz"])

    def training_config(self, section: str, *, seed: int, device: str) -> TrainingConfig:
        values = dict(_mapping(self.payload[section], section))
        return TrainingConfig(seed=int(seed), device=str(device), **values)

    def architecture_receipt(self, input_channels: int) -> dict[str, Any]:
        encoder = dict(_mapping(self.payload["ecog_encoder"], "ecog_encoder"))
        classifier = dict(_mapping(self.payload["classifier"], "classifier"))
        return {
            "runner_version": RUNNER_VERSION,
            "input_channels": int(input_channels),
            "frame_grid": dict(self.payload["frame_grid"]),
            "neural_preprocessing": dict(self.payload["neural_preprocessing"]),
            "audio": dict(self.payload["audio"]),
            "targets": dict(self.payload["targets"]),
            "ecog_encoder": encoder,
            "classifier": classifier,
            "hidden_extraction": dict(self.payload["hidden_extraction"]),
            "selection": dict(self.payload["selection"]),
            "ensemble": dict(self.payload["ensemble"]),
            "mel_compute_matched": dict(self.payload["mel_compute_matched"]),
        }


def validate_primary_config(value: Mapping[str, Any]) -> PrimaryConfig:
    """Reject protocol drift instead of silently accepting a near-match."""

    value = _mapping(value, "config")
    _exact_keys(
        value,
        required={
            "schema_version",
            "status",
            "dataset",
            "analysis",
            "run_scope",
            "folds",
            "seeds",
            "frame_grid",
            "neural_preprocessing",
            "audio",
            "targets",
            "ecog_encoder",
            "classifier",
            "hidden_extraction",
            "regression_training",
            "classifier_training",
            "selection",
            "ensemble",
            "mel_compute_matched",
        },
        optional={"notes"},
        label="config",
    )
    _exact(value["schema_version"], CONFIG_SCHEMA_VERSION, "schema_version")
    _exact(value["dataset"], "VocalMind-v2", "dataset")
    _exact(value["analysis"], "primary_overt_word_20class", "analysis")
    scope = str(value["run_scope"])
    if scope not in {"fast_smoke", "pilot", "production"}:
        raise PrimaryConfigError(
            "run_scope must be 'fast_smoke', 'pilot', or 'production'"
        )
    status = str(value["status"])
    if scope in {"fast_smoke", "pilot"}:
        _exact(status, "development_only", f"{scope} status")
    elif status not in {"protocol_freeze_required", "frozen_confirmatory"}:
        raise PrimaryConfigError(
            "production status must be protocol_freeze_required or frozen_confirmatory"
        )

    folds = tuple(int(item) for item in value["folds"])
    seeds = tuple(int(item) for item in value["seeds"])
    if not folds or len(folds) != len(set(folds)) or not set(folds).issubset(PRIMARY_FOLDS):
        raise PrimaryConfigError("folds must be a non-empty unique subset of [1,2,3,4,5]")
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise PrimaryConfigError("seeds must be unique non-negative integers")
    if scope in {"fast_smoke", "pilot"}:
        _exact(folds, (1,), f"{scope} folds")
        _exact(seeds, (4,), f"{scope} seeds")
    else:
        _exact(folds, PRIMARY_FOLDS, "production folds")
        _exact(seeds, PRIMARY_SEEDS, "production seeds")

    grid = _mapping(value["frame_grid"], "frame_grid")
    _exact_keys(
        grid,
        required={
            "neural_sample_rate_hz",
            "regression_frame_hz",
            "window_samples",
            "window_end_offset_samples",
            "first_frame_end_sample",
            "hidden_stride",
            "classifier_frame_hz",
            "strict_historical_parity",
            "zero_padding",
        },
        label="frame_grid",
    )
    invariant_grid = {
        "neural_sample_rate_hz": 1000,
        "window_samples": 1001,
        "window_end_offset_samples": 0,
        "first_frame_end_sample": 1000,
        "zero_padding": False,
    }
    for key, expected in invariant_grid.items():
        _exact(grid[key], expected, f"frame_grid.{key}")
    parity_grid = (
        {
            "regression_frame_hz": 50,
            "hidden_stride": 1,
            "classifier_frame_hz": 50,
            "strict_historical_parity": False,
        }
        if scope == "fast_smoke"
        else {
            "regression_frame_hz": 1000,
            "hidden_stride": 10,
            "classifier_frame_hz": 100,
            "strict_historical_parity": True,
        }
    )
    for key, expected in parity_grid.items():
        _exact(grid[key], expected, f"frame_grid.{key}")
    if int(grid["regression_frame_hz"]) // int(grid["hidden_stride"]) != int(
        grid["classifier_frame_hz"]
    ):
        raise PrimaryConfigError("hidden stride does not produce classifier_frame_hz")

    neural_preprocessing = _mapping(value["neural_preprocessing"], "neural_preprocessing")
    _exact_keys(
        neural_preprocessing,
        required={
            "sample_rate_hz",
            "channel_policy",
            "high_pass_hz",
            "low_pass_hz",
            "butterworth_order",
            "notch_frequencies_hz",
            "notch_q",
            "filter_direction",
            "per_trial_z_score",
            "downstream_standardization",
        },
        label="neural_preprocessing",
    )
    for key, expected in {
        "sample_rate_hz": 1000,
        "channel_policy": "release_110_clean_channels_then_car",
        "high_pass_hz": 10.0,
        "low_pass_hz": 200.0,
        "butterworth_order": 5,
        "notch_frequencies_hz": [50.0, 100.0, 150.0],
        "notch_q": [25.0, 100.0, 150.0],
        "filter_direction": "forward_backward_zero_phase",
        "per_trial_z_score": False,
        "downstream_standardization": "train_only_channel_standardizer",
    }.items():
        _exact(
            neural_preprocessing[key],
            expected,
            f"neural_preprocessing.{key}",
        )

    audio = _mapping(value["audio"], "audio")
    _exact_keys(
        audio,
        required={"source_sample_rate_hz", "source_channels", "channel_policy"},
        label="audio",
    )
    _exact(audio["source_sample_rate_hz"], 44_100, "audio.source_sample_rate_hz")
    _exact(audio["source_channels"], 2, "audio.source_channels")
    _exact(audio["channel_policy"], "arithmetic_mean_to_mono", "audio.channel_policy")

    targets = _mapping(value["targets"], "targets")
    _exact_keys(
        targets,
        required={"target_dim", "reducer_seed", "mel", "whisper"},
        label="targets",
    )
    _exact(targets["target_dim"], 50, "targets.target_dim")
    if int(targets["reducer_seed"]) < 0:
        raise PrimaryConfigError("targets.reducer_seed must be non-negative")
    mel = _mapping(targets["mel"], "targets.mel")
    _exact_keys(
        mel,
        required={
            "kind",
            "waveform_normalization",
            "source_revision",
            "n_mels",
            "sample_rate_hz",
            "n_fft",
            "hop_length",
            "win_length",
            "window",
            "center",
            "pad_mode",
            "spectrum",
            "fmin_hz",
            "fmax_hz",
            "log",
            "epsilon",
            "pca_components",
            "pca_whiten",
        },
        label="targets.mel",
    )
    for key, expected in {
        "kind": "vocalmind_author_spectral_parameters_shared_peak_normalization",
        "waveform_normalization": "per_trial_peak_abs_epsilon_1e-8_shared_with_whisper",
        "source_revision": VOCALMIND_CODE_REVISION,
        "n_mels": 80,
        "sample_rate_hz": 16_000,
        "n_fft": 1024,
        "hop_length": 320,
        "win_length": 1024,
        "window": "hann",
        "center": True,
        "pad_mode": "constant",
        "spectrum": "magnitude_not_power",
        "fmin_hz": 80.0,
        "fmax_hz": 7600.0,
        "log": "log10",
        "epsilon": 1e-6,
        "pca_components": 50,
        "pca_whiten": True,
    }.items():
        _exact(mel[key], expected, f"targets.mel.{key}")
    whisper = _mapping(targets["whisper"], "targets.whisper")
    _exact_keys(
        whisper,
        required={
            "model",
            "revision",
            "layers",
            "raw_dim",
            "peak_normalize",
            "pca_components",
            "pca_whiten",
        },
        label="targets.whisper",
    )
    for key, expected in {
        "model": PINNED_WHISPER_MODEL,
        "revision": PINNED_WHISPER_REVISION,
        "layers": list(WHISPER_LAYERS),
        "raw_dim": 512,
        "peak_normalize": True,
        "pca_components": 50,
        "pca_whiten": True,
    }.items():
        _exact(whisper[key], expected, f"targets.whisper.{key}")

    encoder = _mapping(value["ecog_encoder"], "ecog_encoder")
    _exact_keys(
        encoder,
        required={
            "target_dim",
            "window_samples",
            "hidden_channels",
            "temporal_stride",
            "filtering_kernel",
            "envelope_kernel",
            "use_lstm",
            "hidden_dim",
        },
        label="ecog_encoder",
    )
    for key, expected in {
        "target_dim": 50,
        "window_samples": 1001,
        "hidden_channels": 30,
        "temporal_stride": 10,
        "filtering_kernel": 25,
        "envelope_kernel": 15,
        "use_lstm": True,
        "hidden_dim": 3030,
    }.items():
        _exact(encoder[key], expected, f"ecog_encoder.{key}")

    classifier = _mapping(value["classifier"], "classifier")
    _exact_keys(
        classifier,
        required={
            "input_features",
            "num_classes",
            "convolution_channels",
            "convolution_kernel",
            "pool_kernel",
            "lstm_hidden",
        },
        label="classifier",
    )
    for key, expected in {
        "input_features": 3030,
        "num_classes": 20,
        "convolution_channels": 100,
        "convolution_kernel": 10,
        "pool_kernel": 10,
        "lstm_hidden": 100,
    }.items():
        _exact(classifier[key], expected, f"classifier.{key}")

    hidden_extraction = _mapping(value["hidden_extraction"], "hidden_extraction")
    _exact_keys(hidden_extraction, required={"batch_size"}, label="hidden_extraction")
    if int(hidden_extraction["batch_size"]) < 32:
        raise PrimaryConfigError("hidden_extraction.batch_size must be at least 32")

    training_keys = {
        "max_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "min_delta",
        "grad_clip_norm",
        "num_workers",
        "strict_determinism",
        "initialize_from_seed",
    }
    for section in ("regression_training", "classifier_training"):
        training = _mapping(value[section], section)
        _exact_keys(training, required=training_keys, label=section)
        try:
            TrainingConfig(seed=4, device="cpu", **dict(training))
        except (TypeError, ValueError) as exc:
            raise PrimaryConfigError(f"invalid {section}: {exc}") from exc

    selection = _mapping(value["selection"], "selection")
    _exact_keys(
        selection,
        required={"regression_model", "word_model", "threshold", "test_gate"},
        label="selection",
    )
    _exact(
        selection["regression_model"],
        "minimum_validation_loss_with_early_stopping",
        "selection.regression_model",
    )
    _exact(
        selection["word_model"],
        "minimum_validation_loss_with_early_stopping",
        "selection.word_model",
    )
    _exact(
        selection["threshold"],
        "not_applicable_closed_set_argmax",
        "selection.threshold",
    )
    _exact(
        selection["test_gate"],
        "all_configured_seeds_x_all_predeclared_units_fixed",
        "selection.test_gate",
    )
    ensemble = _mapping(value["ensemble"], "ensemble")
    _exact_keys(ensemble, required={"layers", "rule", "subset_search"}, label="ensemble")
    _exact(ensemble["layers"], list(WHISPER_LAYERS), "ensemble.layers")
    _exact(
        ensemble["rule"],
        "arithmetic_mean_of_softmax_probabilities",
        "ensemble.rule",
    )
    _exact(ensemble["subset_search"], False, "ensemble.subset_search")

    mel_control = _mapping(value["mel_compute_matched"], "mel_compute_matched")
    _exact_keys(
        mel_control,
        required={
            "production_required",
            "production_seed_offsets",
            "pilot_initializations",
            "rule",
            "subset_search",
        },
        label="mel_compute_matched",
    )
    for key, expected in {
        "production_required": True,
        "production_seed_offsets": [0, 1000, 2000],
        "pilot_initializations": 1,
        "rule": "arithmetic_mean_of_softmax_probabilities",
        "subset_search": False,
    }.items():
        _exact(mel_control[key], expected, f"mel_compute_matched.{key}")

    canonical = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return PrimaryConfig(payload=canonical, fingerprint=fingerprint_json(canonical))


def load_primary_config(path: Path | str) -> PrimaryConfig:
    try:
        value = read_json(Path(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimaryConfigError(f"cannot read config {path}: {exc}") from exc
    return validate_primary_config(_mapping(value, "config"))


def valid_frame_end_samples(
    trial_samples: int,
    *,
    sample_rate_hz: int = 1000,
    frame_hz: int = 1000,
    window_samples: int = 1001,
    window_end_offset_samples: int = 0,
) -> np.ndarray:
    """Return the configured grid whose inclusive one-second windows are in-bounds."""

    for name, value in (
        ("trial_samples", trial_samples),
        ("sample_rate_hz", sample_rate_hz),
        ("frame_hz", frame_hz),
        ("window_samples", window_samples),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(sample_rate_hz) % int(frame_hz) != 0:
        raise ValueError("sample rate must be exactly divisible by frame rate")
    step = int(sample_rate_hz) // int(frame_hz)
    first_end = int(window_samples) - 1 - int(window_end_offset_samples)
    last_end_exclusive = int(trial_samples) - int(window_end_offset_samples)
    result = np.arange(first_end, last_end_exclusive, step, dtype=np.int64)
    if result.size == 0:
        raise ValueError("trial has no valid unpadded one-second frame")
    starts = result + int(window_end_offset_samples) - int(window_samples) + 1
    ends = result + int(window_end_offset_samples)
    if np.any(starts < 0) or np.any(ends >= int(trial_samples)):
        raise RuntimeError("valid-frame implementation admitted a padded window")
    result.setflags(write=False)
    return result


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("VocalMind audio must have shape (frames, 2)")
    if not np.isfinite(values).all():
        raise ValueError("audio contains NaN or Infinity")
    mono = np.mean(values, axis=1, dtype=np.float32)
    mono.setflags(write=False)
    return mono


@dataclass(frozen=True)
class FrameSpec:
    sample_ids: tuple[str, ...]
    trial_ids: tuple[str, ...]
    frame_times_s: np.ndarray
    frame_end_samples: np.ndarray
    frames_per_trial: int


def build_frame_spec(
    trial_ids: Sequence[str],
    *,
    trial_samples: int,
    sample_rate_hz: int = 1000,
    frame_hz: int = 1000,
    window_samples: int = 1001,
) -> FrameSpec:
    ids = tuple(str(value) for value in trial_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("trial_ids must be non-empty and unique")
    ends = valid_frame_end_samples(
        trial_samples,
        sample_rate_hz=sample_rate_hz,
        frame_hz=frame_hz,
        window_samples=window_samples,
    )
    sample_ids: list[str] = []
    frame_trial_ids: list[str] = []
    times: list[float] = []
    repeated_ends: list[int] = []
    for trial_id in ids:
        for end in ends:
            sample_ids.append(f"{trial_id}:end{int(end):06d}")
            frame_trial_ids.append(trial_id)
            times.append(int(end) / int(sample_rate_hz))
            repeated_ends.append(int(end))
    time_array = np.asarray(times, dtype=np.float64)
    end_array = np.asarray(repeated_ends, dtype=np.int64)
    time_array.setflags(write=False)
    end_array.setflags(write=False)
    return FrameSpec(
        sample_ids=tuple(sample_ids),
        trial_ids=tuple(frame_trial_ids),
        frame_times_s=time_array,
        frame_end_samples=end_array,
        frames_per_trial=int(ends.size),
    )


class AcousticTargetProvider(Protocol):
    def provenance(self) -> Mapping[str, Any]: ...

    def extract(
        self, audio_mono: np.ndarray, sample_rate_hz: int, target_times_s: np.ndarray
    ) -> Mapping[str, np.ndarray]: ...


class DefaultAcousticTargetProvider:
    """One MEL extraction plus one shared Whisper L3/L4/L5 forward per trial."""

    def __init__(self, config: PrimaryConfig, *, device: str) -> None:
        targets = config.payload["targets"]
        mel = targets["mel"]
        whisper = targets["whisper"]
        self.mel = VocalMindAuthorMelTargetExtractor(
            n_mels=int(mel["n_mels"]),
            sample_rate_hz=int(mel["sample_rate_hz"]),
            n_fft=int(mel["n_fft"]),
            hop_length=int(mel["hop_length"]),
            win_length=int(mel["win_length"]),
            window=str(mel["window"]),
            fmin_hz=float(mel["fmin_hz"]),
            fmax_hz=float(mel["fmax_hz"]),
            epsilon=float(mel["epsilon"]),
            peak_normalize=True,
        )
        self.whisper = WhisperLayerTargetExtractor(
            model_name=str(whisper["model"]),
            revision=str(whisper["revision"]),
            layers=tuple(int(value) for value in whisper["layers"]),
            device=str(device),
            peak_normalize=bool(whisper["peak_normalize"]),
        )

    def provenance(self) -> Mapping[str, Any]:
        return {"mel": self.mel.provenance(), "whisper": self.whisper.provenance()}

    def extract(
        self, audio_mono: np.ndarray, sample_rate_hz: int, target_times_s: np.ndarray
    ) -> Mapping[str, np.ndarray]:
        layers = self.whisper.extract_aligned(audio_mono, sample_rate_hz, target_times_s)
        return {
            "mel": self.mel.extract_aligned(audio_mono, sample_rate_hz, target_times_s),
            "L3": layers[3],
            "L4": layers[4],
            "L5": layers[5],
        }


@dataclass(frozen=True)
class RawTargetBundle:
    sample_ids: tuple[str, ...]
    frame_trial_ids: tuple[str, ...]
    frame_times_s: np.ndarray
    by_representation: Mapping[str, np.ndarray]
    train_frame_count: int
    frames_per_trial: int
    provenance: Mapping[str, Any]


def extract_train_validation_targets(
    *,
    adapter: VocalMindAdapter,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    provider: AcousticTargetProvider,
    regression_frame_hz: int = 1000,
    window_samples: int = 1001,
) -> RawTargetBundle:
    """Read acoustic data only for train/validation trials and preserve exact order."""

    ordered_trials = tuple(train_ids) + tuple(validation_ids)
    frame_spec = build_frame_spec(
        ordered_trials,
        trial_samples=adapter.contract.eeg_samples,
        sample_rate_hz=adapter.contract.eeg_sample_rate_hz,
        frame_hz=int(regression_frame_hz),
        window_samples=int(window_samples),
    )
    by_representation: dict[str, list[np.ndarray]] = {
        unit: [] for unit in REPRESENTATIONS
    }
    ends = valid_frame_end_samples(
        adapter.contract.eeg_samples,
        sample_rate_hz=adapter.contract.eeg_sample_rate_hz,
        frame_hz=int(regression_frame_hz),
        window_samples=int(window_samples),
    )
    target_times = ends.astype(np.float64) / adapter.contract.eeg_sample_rate_hz
    for trial_id in ordered_trials:
        trial = adapter.trial(trial_id)
        audio = adapter.load_audio(trial)
        mono = stereo_to_mono(audio)
        extracted = provider.extract(
            mono,
            adapter.contract.audio_sample_rate_hz,
            target_times,
        )
        if set(extracted) != set(REPRESENTATIONS):
            raise RuntimeError(
                f"target provider returned {sorted(extracted)}, expected {list(REPRESENTATIONS)}"
            )
        for unit in REPRESENTATIONS:
            values = np.asarray(extracted[unit], dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(target_times):
                raise RuntimeError(
                    f"{unit} target shape {values.shape} does not match {len(target_times)} frames"
                )
            if not np.isfinite(values).all():
                raise FloatingPointError(f"{unit} targets contain NaN or Infinity")
            by_representation[unit].append(values)
    combined = {
        unit: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
        for unit, parts in by_representation.items()
    }
    expected_dims = {"mel": 80, "L3": 512, "L4": 512, "L5": 512}
    for unit, expected_dim in expected_dims.items():
        if combined[unit].shape != (len(frame_spec.sample_ids), expected_dim):
            raise RuntimeError(
                f"{unit} raw target contract mismatch: {combined[unit].shape}, "
                f"expected {(len(frame_spec.sample_ids), expected_dim)}"
            )
        combined[unit].setflags(write=False)
    return RawTargetBundle(
        sample_ids=frame_spec.sample_ids,
        frame_trial_ids=frame_spec.trial_ids,
        frame_times_s=frame_spec.frame_times_s,
        by_representation=combined,
        train_frame_count=len(train_ids) * frame_spec.frames_per_trial,
        frames_per_trial=frame_spec.frames_per_trial,
        provenance=dict(provider.provenance()),
    )


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable array artifact already exists: {path}")
    temporary = path.with_name(f".{path.stem}.partial-{os.getpid()}.npz")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"immutable array artifact already exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_or_validate_npz(path: Path, **arrays: np.ndarray) -> str:
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(normalized):
                raise RuntimeError(f"existing NPZ keys differ: {path}")
            for key, expected in normalized.items():
                actual = existing[key]
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or not np.array_equal(actual, expected, equal_nan=False)
                ):
                    raise RuntimeError(f"existing NPZ array differs ({key}): {path}")
    else:
        _atomic_save_npz(path, **normalized)
    return sha256_file(path)


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    value = dict(payload)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    atomic_write_json(path, value, overwrite=False)


def save_raw_target_cache(
    directory: Path,
    bundle: RawTargetBundle,
    *,
    cache_key: str,
) -> None:
    directory = Path(directory)
    arrays_path = directory / "targets.npz"
    manifest_path = directory / "manifest.json"
    if arrays_path.exists() or manifest_path.exists():
        raise FileExistsError(f"target cache already exists or is incomplete: {directory}")
    _atomic_save_npz(
        arrays_path,
        sample_ids=np.asarray(bundle.sample_ids, dtype=np.str_),
        frame_trial_ids=np.asarray(bundle.frame_trial_ids, dtype=np.str_),
        frame_times_s=np.asarray(bundle.frame_times_s, dtype=np.float64),
        mel=np.asarray(bundle.by_representation["mel"], dtype=np.float32),
        L3=np.asarray(bundle.by_representation["L3"], dtype=np.float32),
        L4=np.asarray(bundle.by_representation["L4"], dtype=np.float32),
        L5=np.asarray(bundle.by_representation["L5"], dtype=np.float32),
    )
    manifest = {
        "schema_version": 1,
        "kind": "vocalmind_train_validation_raw_acoustic_targets",
        "cache_key": cache_key,
        "array_file": arrays_path.name,
        "array_sha256": sha256_file(arrays_path),
        "sample_count": len(bundle.sample_ids),
        "train_frame_count": bundle.train_frame_count,
        "frames_per_trial": bundle.frames_per_trial,
        "provenance": dict(bundle.provenance),
    }
    manifest["fingerprint"] = fingerprint_json(manifest)
    atomic_write_json(manifest_path, manifest, overwrite=False)


def load_raw_target_cache(directory: Path, *, expected_cache_key: str) -> RawTargetBundle:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    fingerprint = manifest.pop("fingerprint", None)
    if fingerprint != fingerprint_json(manifest):
        raise RuntimeError(f"target cache manifest fingerprint mismatch: {directory}")
    if manifest.get("kind") != "vocalmind_train_validation_raw_acoustic_targets":
        raise RuntimeError(f"unexpected target cache kind: {directory}")
    if manifest.get("cache_key") != expected_cache_key:
        raise RuntimeError(f"target cache belongs to a different config/split: {directory}")
    arrays_path = directory / str(manifest["array_file"])
    if sha256_file(arrays_path) != manifest.get("array_sha256"):
        raise RuntimeError(f"target cache checksum mismatch: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        sample_ids = tuple(str(value) for value in arrays["sample_ids"].tolist())
        trial_ids = tuple(str(value) for value in arrays["frame_trial_ids"].tolist())
        times = np.asarray(arrays["frame_times_s"], dtype=np.float64)
        targets = {
            unit: np.asarray(arrays[unit], dtype=np.float32)
            for unit in REPRESENTATIONS
        }
    if not (len(sample_ids) == len(trial_ids) == len(times)):
        raise RuntimeError("target cache arrays have inconsistent row counts")
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("target cache sample IDs are not unique")
    if any(len(values) != len(sample_ids) or not np.isfinite(values).all() for values in targets.values()):
        raise RuntimeError("target cache contains invalid target arrays")
    times.setflags(write=False)
    for values in targets.values():
        values.setflags(write=False)
    return RawTargetBundle(
        sample_ids=sample_ids,
        frame_trial_ids=trial_ids,
        frame_times_s=times,
        by_representation=targets,
        train_frame_count=int(manifest["train_frame_count"]),
        frames_per_trial=int(manifest["frames_per_trial"]),
        provenance=dict(manifest["provenance"]),
    )


class TrialSequenceDataset(Dataset):
    """Fixed hidden trajectories for the common closed-set word classifier."""

    def __init__(
        self,
        *,
        sequences: np.ndarray,
        labels: Sequence[int],
        trial_ids: Sequence[str],
        split_role: str,
        split_fingerprint: str | None = None,
    ) -> None:
        values = np.asarray(sequences, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int64)
        ids = tuple(str(value) for value in trial_ids)
        if split_role not in {"train", "validation", "held_out_test"}:
            raise ValueError("invalid split_role")
        if split_role == "held_out_test" and split_fingerprint is None:
            raise ValueError("held_out_test sequences require split_fingerprint")
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("sequences must have shape (trials, hidden_features, frames)")
        if len(ids) != len(set(ids)) or len(ids) != len(values) or targets.shape != (len(ids),):
            raise ValueError("one unique ID and scalar label are required per sequence")
        if not np.isfinite(values).all() or np.any(targets < 0):
            raise ValueError("hidden sequences/labels are invalid")
        self.sequences = np.ascontiguousarray(values)
        self.labels = targets
        self.sample_ids = ids
        self.split_role = split_role
        self.split_fingerprint = split_fingerprint

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "inputs": torch.from_numpy(self.sequences[int(index)]),
            "target": torch.tensor(int(self.labels[int(index)]), dtype=torch.long),
            "sample_id": self.sample_ids[int(index)],
        }


def _ecog_encoder(config: PrimaryConfig, *, input_channels: int) -> OneSecondEcogEncoder:
    values = config.payload["ecog_encoder"]
    model = OneSecondEcogEncoder(
        input_channels=int(input_channels),
        target_dim=int(values["target_dim"]),
        window_samples=int(values["window_samples"]),
        hidden_channels=int(values["hidden_channels"]),
        temporal_stride=int(values["temporal_stride"]),
        filtering_kernel=int(values["filtering_kernel"]),
        envelope_kernel=int(values["envelope_kernel"]),
        use_lstm=bool(values["use_lstm"]),
    )
    if model.hidden_dim != int(values["hidden_dim"]):
        raise RuntimeError("configured ECoG hidden dimension disagrees with implementation")
    return model


def _word_classifier(config: PrimaryConfig, *, num_classes: int) -> HiddenSequenceClassifier:
    values = config.payload["classifier"]
    if int(num_classes) != int(values["num_classes"]):
        raise RuntimeError(
            f"dataset has {num_classes} classes but primary classifier is fixed to "
            f"{values['num_classes']}"
        )
    return HiddenSequenceClassifier(
        input_features=int(values["input_features"]),
        num_classes=int(values["num_classes"]),
        convolution_channels=int(values["convolution_channels"]),
        convolution_kernel=int(values["convolution_kernel"]),
        pool_kernel=int(values["pool_kernel"]),
        lstm_hidden=int(values["lstm_hidden"]),
    )


def _frame_dataset(
    *,
    trial_arrays: Mapping[str, np.ndarray],
    trial_ids: Sequence[str],
    targets: np.ndarray,
    standardizer: ChannelStandardizerArtifact,
    split_role: str,
    split_fingerprint: str | None = None,
    regression_frame_hz: int = 1000,
    window_samples: int = 1001,
) -> FrameWindowDataset:
    if not trial_ids:
        raise ValueError("trial_ids cannot be empty")
    first = np.asarray(trial_arrays[str(trial_ids[0])])
    frame_spec = build_frame_spec(
        tuple(trial_ids),
        trial_samples=int(first.shape[0]),
        frame_hz=int(regression_frame_hz),
        window_samples=int(window_samples),
    )
    return FrameWindowDataset(
        trials={str(trial_id): trial_arrays[str(trial_id)] for trial_id in trial_ids},
        sample_ids=frame_spec.sample_ids,
        frame_trial_ids=frame_spec.trial_ids,
        frame_times_s=frame_spec.frame_times_s,
        targets=np.asarray(targets, dtype=np.float32),
        split_role=split_role,
        sample_rate_hz=1000,
        window_samples=int(window_samples),
        window_end_offset_samples=0,
        standardizer=standardizer,
        split_fingerprint=split_fingerprint,
    )


def encode_hidden_sequences(
    model: OneSecondEcogEncoder,
    dataset: FrameWindowDataset,
    *,
    ordered_trial_ids: Sequence[str],
    frames_per_trial: int,
    hidden_stride: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode the 1 kHz regression grid and retain the fixed 100 Hz hidden grid."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(hidden_stride) <= 0:
        raise ValueError("hidden_stride must be positive")
    expected = len(ordered_trial_ids) * int(frames_per_trial)
    if len(dataset) != expected:
        raise ValueError(f"frame dataset has {len(dataset)} rows, expected {expected}")
    runtime_device = torch.device(device)
    model.to(runtime_device).eval()
    selected_batches: list[np.ndarray] = []
    observed_ids: list[str] = []
    global_offset = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0):
            inputs = torch.as_tensor(batch["inputs"], dtype=torch.float32, device=runtime_device)
            hidden = model(inputs, return_hidden=True)
            if not torch.isfinite(hidden).all().item():
                raise FloatingPointError("ECoG hidden trajectory contains NaN or Infinity")
            batch_count = int(hidden.shape[0])
            positions = np.arange(global_offset, global_offset + batch_count, dtype=np.int64)
            within_trial = positions % int(frames_per_trial)
            keep = within_trial % int(hidden_stride) == 0
            if np.any(keep):
                selected_batches.append(hidden[keep].detach().cpu().float().numpy())
            observed_ids.extend(str(value) for value in batch["sample_id"])
            global_offset += batch_count
    if tuple(observed_ids) != tuple(dataset.sample_ids):
        raise RuntimeError("hidden extraction changed frame ordering")
    matrix = np.concatenate(selected_batches, axis=0)
    classifier_frames = (int(frames_per_trial) - 1) // int(hidden_stride) + 1
    if matrix.shape[0] != len(ordered_trial_ids) * classifier_frames:
        raise RuntimeError("hidden stride produced an unexpected classifier timeline")
    sequences = matrix.reshape(len(ordered_trial_ids), classifier_frames, -1).transpose(0, 2, 1)
    return np.ascontiguousarray(sequences, dtype=np.float32)


def closed_set_metrics(
    probabilities: np.ndarray,
    labels: Sequence[int],
    *,
    class_order: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    classes = tuple(str(value) for value in class_order)
    if values.ndim != 2 or values.shape != (len(targets), len(classes)):
        raise ValueError("probability matrix shape does not match labels/classes")
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(
        values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
    ):
        raise ValueError("invalid probability matrix")
    if np.any(targets < 0) or np.any(targets >= len(classes)):
        raise ValueError("label outside class order")
    predicted = np.argmax(values, axis=1)
    recalls: list[float] = []
    f1_values: list[float] = []
    per_class: dict[str, float] = {}
    for class_index, class_name in enumerate(classes):
        true_positive = int(np.sum((predicted == class_index) & (targets == class_index)))
        false_positive = int(np.sum((predicted == class_index) & (targets != class_index)))
        false_negative = int(np.sum((predicted != class_index) & (targets == class_index)))
        support = true_positive + false_negative
        recall = true_positive / support if support else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
        per_class[class_name] = recall
    top_k = min(3, len(classes))
    top_indices = np.argpartition(values, kth=len(classes) - top_k, axis=1)[:, -top_k:]
    top3 = float(np.mean(np.any(top_indices == targets[:, None], axis=1)))
    return {
        "sample_count": int(len(targets)),
        "accuracy": float(np.mean(predicted == targets)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "top3_accuracy": top3,
        "per_class_recall": per_class,
        "decision_rule": "closed_set_argmax_no_threshold",
    }


def arithmetic_probability_mean(
    outputs: Sequence[EvaluationResult],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Average a predeclared list of aligned softmax matrices without search."""

    if not outputs:
        raise ValueError("at least one probability output is required")
    reference_ids = outputs[0].sample_ids
    reference_shape = outputs[0].predictions.shape
    arrays: list[np.ndarray] = []
    for output in outputs:
        if output.sample_ids != reference_ids or output.predictions.shape != reference_shape:
            raise ValueError("probability ensemble members are not exactly aligned")
        values = np.asarray(output.predictions, dtype=np.float64)
        if not np.isfinite(values).all() or not np.allclose(
            values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("invalid softmax probabilities")
        arrays.append(values)
    averaged = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
    averaged.setflags(write=False)
    return reference_ids, averaged


def _load_trials(
    adapter: VocalMindAdapter,
    trial_ids: Sequence[str],
    preprocessor: VocalMindNeuralPreprocessor,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for trial_id in trial_ids:
        trial = adapter.trial(str(trial_id))
        values[str(trial_id)] = preprocessor.transform(adapter.load_eeg(trial))
    return values


def _labels(
    adapter: VocalMindAdapter,
    trial_ids: Sequence[str],
    class_order: Sequence[str],
) -> np.ndarray:
    lookup = {word: index for index, word in enumerate(class_order)}
    result = np.asarray([lookup[adapter.trial(trial_id).word] for trial_id in trial_ids], dtype=np.int64)
    result.setflags(write=False)
    return result


def _split_from_fold(
    fold: Mapping[str, Any],
    *,
    dataset_index_sha256: str,
) -> SplitManifest:
    return SplitManifest.create(
        dataset_id="VocalMind-v2:primary-overt",
        protocol_id=f"vocalmind-primary-reps1-5-fold-{int(fold['fold']):02d}",
        split_seed=0,
        train_ids=fold["train_trial_ids"],
        validation_ids=fold["validation_trial_ids"],
        test_ids=fold["test_trial_ids"],
        purge_gap_seconds=0.0,
        dataset_manifest_sha256=dataset_index_sha256,
    )


def _load_or_fit_standardizer(
    directory: Path,
    *,
    train_trials: Mapping[str, np.ndarray],
    train_ids: Sequence[str],
) -> ChannelStandardizerArtifact:
    expected_ids_hash = fingerprint_json(list(train_ids))
    if directory.exists():
        artifact = ChannelStandardizerArtifact.load(directory)
        if artifact.train_trial_ids_sha256 != expected_ids_hash:
            raise RuntimeError("existing neural standardizer was fitted on different trials")
        return artifact
    artifact = fit_train_only_channel_standardizer(
        [train_trials[trial_id] for trial_id in train_ids],
        list(train_ids),
        split_role="train",
    )
    artifact.save(directory)
    return artifact


def _load_or_fit_reducer(
    directory: Path,
    *,
    raw_train: np.ndarray,
    train_sample_ids: Sequence[str],
    n_components: int,
    whiten: bool,
    seed: int,
) -> ReducerArtifact:
    expected_ids_hash = fingerprint_json(list(train_sample_ids))
    if directory.exists():
        artifact = ReducerArtifact.load(directory)
        if (
            artifact.train_sample_ids_sha256 != expected_ids_hash
            or artifact.output_dim != int(n_components)
            or artifact.whiten != bool(whiten)
            or artifact.seed != int(seed)
        ):
            raise RuntimeError(f"existing reducer provenance differs: {directory}")
        return artifact
    artifact = fit_train_only_reducer(
        raw_train,
        train_sample_ids,
        n_components=int(n_components),
        whiten=bool(whiten),
        seed=int(seed),
        split_role="train",
    )
    artifact.save(directory)
    return artifact


@dataclass
class FixedUnitModels:
    encoder: OneSecondEcogEncoder
    classifier: HiddenSequenceClassifier
    regression_result: TrainingResult
    classifier_result: TrainingResult


@dataclass(frozen=True)
class PlannedTrainingUnit:
    key: str
    target_representation: str
    initialization_seed: int


def planned_training_units(
    config: PrimaryConfig,
    *,
    outer_seed: int,
) -> tuple[PlannedTrainingUnit, ...]:
    """Predeclare all models; no seed/layer subset is selected from outcomes."""

    mel_offsets = (
        tuple(int(value) for value in config.payload["mel_compute_matched"]["production_seed_offsets"])
        if config.run_scope == "production"
        else (0,)
    )
    mel_units = tuple(
        PlannedTrainingUnit(
            key="mel" if index == 0 else f"mel_init{index}",
            target_representation="mel",
            initialization_seed=int(outer_seed) + offset,
        )
        for index, offset in enumerate(mel_offsets)
    )
    whisper_units = tuple(
        PlannedTrainingUnit(
            key=f"L{layer}",
            target_representation=f"L{layer}",
            initialization_seed=int(outer_seed),
        )
        for layer in WHISPER_LAYERS
    )
    return mel_units + whisper_units


class VocalMindPrimaryRunner:
    """Run one pilot or a protocol-frozen multi-fold/multi-seed experiment."""

    def __init__(
        self,
        *,
        config: PrimaryConfig,
        data_root: Path,
        output_root: Path,
        device: str,
        target_provider_factory=None,
        contract: VocalMindContract = DEFAULT_VOCALMIND_CONTRACT,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.device = str(device)
        self.contract = contract
        self.adapter = VocalMindAdapter(self.data_root, contract)
        self.neural_preprocessor = VocalMindNeuralPreprocessor.from_mapping(
            dict(config.payload["neural_preprocessing"]),
            expected_channel_count=len(contract.channel_ids),
        )
        self.target_provider_factory = target_provider_factory
        self._validate_roots()
        self._require_confirmatory_run()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    def _require_confirmatory_run(self) -> None:
        if not (
            self.config.run_scope == "production"
            and self.config.status == "frozen_confirmatory"
        ):
            raise PrimaryConfigError(
                "numeric reps1-5 training is allowed only for a protocol-frozen "
                "production config; development/fast-smoke/pilot configs are "
                "strictly plan-only"
            )

    def _validate_roots(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"VocalMind data root does not exist: {self.data_root}")
        source_root = Path(__file__).resolve().parents[2]
        for forbidden, label in ((self.data_root, "data root"), (source_root, "source tree")):
            try:
                self.output_root.relative_to(forbidden)
            except ValueError:
                pass
            else:
                raise ValueError(f"output_root must not be inside the {label}: {self.output_root}")

    def _provider(self) -> AcousticTargetProvider:
        if self.target_provider_factory is None:
            return DefaultAcousticTargetProvider(self.config, device=self.device)
        return self.target_provider_factory(self.config, self.device)

    def plan(self) -> dict[str, Any]:
        dataset_index = self.adapter.build_index(deep=False, hash_files=False)
        split_payload = build_primary_split_manifest(dataset_index, self.contract)
        grid = self.config.payload["frame_grid"]
        ends = valid_frame_end_samples(
            self.contract.eeg_samples,
            sample_rate_hz=int(grid["neural_sample_rate_hz"]),
            frame_hz=int(grid["regression_frame_hz"]),
            window_samples=int(grid["window_samples"]),
            window_end_offset_samples=int(grid["window_end_offset_samples"]),
        )
        return {
            "schema_version": 1,
            "kind": "vocalmind_primary_execution_plan",
            "runner_version": RUNNER_VERSION,
            "config_fingerprint": self.config.fingerprint,
            "scope": self.config.run_scope,
            "status": self.config.status,
            "folds": list(self.config.folds),
            "seeds": list(self.config.seeds),
            "dataset_counts": dataset_index["counts"],
            "primary_rep6_forbidden": True,
            "rep6_policy": dict(REP6_DEVELOPMENT_AUDIT),
            "frames_per_trial": int(len(ends)),
            "classifier_frames_per_trial": int(
                math.ceil(len(ends) / int(grid["hidden_stride"]))
            ),
            "first_frame_time_s": float(ends[0] / 1000),
            "last_frame_time_s": float(ends[-1] / 1000),
            "numerical_training_allowed": bool(
                self.config.run_scope == "production"
                and self.config.status == "frozen_confirmatory"
            ),
            "test_gate_open": False,
            "pre_freeze_primary_policy": "metadata_only_plan_no_numeric_reps1_5",
            "numeric_test_access_before_gate": False,
            "fold_counts": {
                int(fold["fold"]): fold["counts"]
                for fold in split_payload["folds"]
                if int(fold["fold"]) in self.config.folds
            },
            "architecture": self.config.architecture_receipt(len(self.contract.channel_ids)),
            "neural_preprocessing_provenance": self.neural_preprocessor.provenance(),
        }

    def run(self, *, max_epochs_this_call: int | None = None) -> dict[str, Any]:
        # Keep this guard even though __init__ also checks it: tests and future
        # factory code must not be able to bypass the pre-freeze quarantine.
        self._require_confirmatory_run()
        source_identity = capture_source_identity()
        require_clean_frozen_source(source_identity)
        source_identity_fingerprint = str(source_identity["fingerprint"])
        preflight_sha256 = str(os.environ.get("WHISPER_ECOG_PREFLIGHT_SHA256", "")).lower()
        if len(preflight_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in preflight_sha256
        ):
            raise RuntimeError(
                "frozen production requires the hashed host preflight receipt from "
                "run_vocalmind_production.ps1"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        dataset_index = self.adapter.build_index(deep=False, hash_files=True)
        split_payload = build_primary_split_manifest(dataset_index, self.contract)
        run_manifest = {
            "schema_version": 1,
            "kind": "vocalmind_primary_run",
            "runner_version": RUNNER_VERSION,
            "config": dict(self.config.payload),
            "config_fingerprint": self.config.fingerprint,
            "dataset_index_sha256": dataset_index["dataset_index_sha256"],
            "dataset_counts": dataset_index["counts"],
            "device": self.device,
            "source_identity": source_identity,
            "source_identity_fingerprint": source_identity_fingerprint,
            "host_preflight_sha256": preflight_sha256,
            "test_access_policy": (
                "all configured outer seeds and every predeclared Whisper/MEL "
                "initialization fixed per fold"
            ),
            "neural_preprocessing": self.neural_preprocessor.provenance(),
            "rep6_policy": dict(REP6_DEVELOPMENT_AUDIT),
        }
        run_manifest["fingerprint"] = fingerprint_json(run_manifest)
        _write_or_validate_json(self.output_root / "run_manifest.json", run_manifest)
        summaries = []
        for fold in split_payload["folds"]:
            if int(fold["fold"]) in self.config.folds:
                summaries.append(
                    self._run_fold(
                        fold,
                        dataset_index_sha256=dataset_index["dataset_index_sha256"],
                        protocol_fingerprint=str(run_manifest["fingerprint"]),
                        source_identity_fingerprint=source_identity_fingerprint,
                        max_epochs_this_call=max_epochs_this_call,
                    )
                )
        result = {
            "schema_version": 1,
            "kind": "vocalmind_primary_run_summary",
            "config_fingerprint": self.config.fingerprint,
            "folds": summaries,
            "threshold_policy": "not_applicable_closed_set_argmax",
            "test_model_selection": False,
            "rep6_policy": dict(REP6_DEVELOPMENT_AUDIT),
        }
        result["fingerprint"] = fingerprint_json(result)
        _write_or_validate_json(self.output_root / "summary.json", result)
        return result

    def _run_fold(
        self,
        fold: Mapping[str, Any],
        *,
        dataset_index_sha256: str,
        protocol_fingerprint: str,
        source_identity_fingerprint: str,
        max_epochs_this_call: int | None,
    ) -> dict[str, Any]:
        fold_number = int(fold["fold"])
        fold_root = self.output_root / f"fold_{fold_number:02d}"
        shared_root = fold_root / "shared_train_validation"
        split = _split_from_fold(fold, dataset_index_sha256=dataset_index_sha256)
        split_path = fold_root / "split_manifest.json"
        if split_path.exists():
            if SplitManifest.load(split_path) != split:
                raise RuntimeError(f"existing split differs for fold {fold_number}")
        else:
            split.save(split_path)

        required_units = tuple(
            f"seed{outer_seed}_{unit.key}"
            for outer_seed in self.config.seeds
            for unit in planned_training_units(self.config, outer_seed=outer_seed)
        )
        gate = TestGate(
            state_directory=fold_root / "test_gate",
            split=split,
            required_units=required_units,
            protocol_fingerprint=protocol_fingerprint,
        )
        train_ids, validation_ids = gate.training_and_validation_ids()

        # No held-out test trial is loaded in this section.
        _write_or_validate_json(
            shared_root / "neural_preprocessing.json",
            self.neural_preprocessor.provenance(),
        )
        train_trials = _load_trials(self.adapter, train_ids, self.neural_preprocessor)
        validation_trials = _load_trials(
            self.adapter, validation_ids, self.neural_preprocessor
        )
        standardizer = _load_or_fit_standardizer(
            shared_root / "neural_standardizer",
            train_trials=train_trials,
            train_ids=train_ids,
        )

        cache_key = fingerprint_json(
            {
                "runner_version": RUNNER_VERSION,
                "source_identity_fingerprint": source_identity_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "dataset_index_sha256": dataset_index_sha256,
                "split_fingerprint": split.fingerprint,
                "train_ids": list(train_ids),
                "validation_ids": list(validation_ids),
                "target_contract": self.config.payload["targets"],
                "frame_grid": self.config.payload["frame_grid"],
            }
        )
        target_cache = shared_root / "raw_targets"
        if target_cache.exists():
            raw = load_raw_target_cache(target_cache, expected_cache_key=cache_key)
        else:
            provider = self._provider()
            raw = extract_train_validation_targets(
                adapter=self.adapter,
                train_ids=train_ids,
                validation_ids=validation_ids,
                provider=provider,
                regression_frame_hz=int(
                    self.config.payload["frame_grid"]["regression_frame_hz"]
                ),
                window_samples=int(
                    self.config.payload["frame_grid"]["window_samples"]
                ),
            )
            save_raw_target_cache(target_cache, raw, cache_key=cache_key)
            del provider
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        expected_train_frames = len(train_ids) * raw.frames_per_trial
        if raw.train_frame_count != expected_train_frames:
            raise RuntimeError("raw target cache train boundary is invalid")
        train_sample_ids = raw.sample_ids[: raw.train_frame_count]
        validation_sample_ids = raw.sample_ids[raw.train_frame_count :]
        if any(trial_id in set(split.held_out_test_ids) for trial_id in raw.frame_trial_ids):
            raise RuntimeError("held-out test trial entered the train/validation target cache")

        reduced: dict[str, np.ndarray] = {}
        for unit in REPRESENTATIONS:
            target_spec = (
                self.config.payload["targets"]["mel"]
                if unit == "mel"
                else self.config.payload["targets"]["whisper"]
            )
            reducer = _load_or_fit_reducer(
                shared_root / "reducers" / unit,
                raw_train=raw.by_representation[unit][: raw.train_frame_count],
                train_sample_ids=train_sample_ids,
                n_components=int(target_spec["pca_components"]),
                whiten=bool(target_spec["pca_whiten"]),
                seed=int(self.config.payload["targets"]["reducer_seed"]),
            )
            transformed = reducer.transform(raw.by_representation[unit])
            if transformed.shape != (len(raw.sample_ids), self.config.target_dim):
                raise RuntimeError(f"{unit} reduced target shape is invalid: {transformed.shape}")
            reduced[unit] = transformed

        class_order = tuple(self.contract.words)
        train_labels = _labels(self.adapter, train_ids, class_order)
        validation_labels = _labels(self.adapter, validation_ids, class_order)
        fixed_models: dict[tuple[int, str], FixedUnitModels] = {}
        for outer_seed in self.config.seeds:
            seed_root = fold_root / f"seed_{outer_seed}"
            for planned_unit in planned_training_units(
                self.config, outer_seed=outer_seed
            ):
                unit = planned_unit.key
                target_representation = planned_unit.target_representation
                initialization_seed = planned_unit.initialization_seed
                unit_root = seed_root / unit
                train_targets = reduced[target_representation][: raw.train_frame_count]
                validation_targets = reduced[target_representation][raw.train_frame_count :]
                train_frames = _frame_dataset(
                    trial_arrays=train_trials,
                    trial_ids=train_ids,
                    targets=train_targets,
                    standardizer=standardizer,
                    split_role="train",
                    regression_frame_hz=int(
                        self.config.payload["frame_grid"]["regression_frame_hz"]
                    ),
                    window_samples=int(
                        self.config.payload["frame_grid"]["window_samples"]
                    ),
                )
                validation_frames = _frame_dataset(
                    trial_arrays=validation_trials,
                    trial_ids=validation_ids,
                    targets=validation_targets,
                    standardizer=standardizer,
                    split_role="validation",
                    regression_frame_hz=int(
                        self.config.payload["frame_grid"]["regression_frame_hz"]
                    ),
                    window_samples=int(
                        self.config.payload["frame_grid"]["window_samples"]
                    ),
                )
                if train_frames.sample_ids != train_sample_ids:
                    raise RuntimeError("training frame order differs from target cache")
                if validation_frames.sample_ids != validation_sample_ids:
                    raise RuntimeError("validation frame order differs from target cache")

                encoder = _ecog_encoder(self.config, input_channels=standardizer.channel_count)
                regression_config = self.config.training_config(
                    "regression_training", seed=initialization_seed, device=self.device
                )
                regression_checkpoint = unit_root / "regression_checkpoint.pt"
                regression_result = train_regression(
                    encoder,
                    train_frames,
                    validation_frames,
                    config=regression_config,
                    checkpoint_path=regression_checkpoint,
                    resume=regression_checkpoint.exists(),
                    run_context={
                        "config_fingerprint": self.config.fingerprint,
                        "source_identity_fingerprint": source_identity_fingerprint,
                        "split_fingerprint": split.fingerprint,
                        "fold": fold_number,
                        "outer_seed": outer_seed,
                        "initialization_seed": initialization_seed,
                        "training_unit": unit,
                        "target_representation": target_representation,
                        "stage": "ecog_to_acoustic_pca50",
                    },
                    max_epochs_this_call=max_epochs_this_call,
                )
                if not regression_result.completed:
                    raise RunIncomplete(
                        f"fold {fold_number} seed {outer_seed} {unit} regression saved at epoch "
                        f"{regression_result.epochs_completed}; rerun the same command to resume"
                    )

                hidden_batch = int(self.config.payload["hidden_extraction"]["batch_size"])
                train_sequences = encode_hidden_sequences(
                    encoder,
                    train_frames,
                    ordered_trial_ids=train_ids,
                    frames_per_trial=raw.frames_per_trial,
                    hidden_stride=int(
                        self.config.payload["frame_grid"]["hidden_stride"]
                    ),
                    batch_size=hidden_batch,
                    device=self.device,
                )
                validation_sequences = encode_hidden_sequences(
                    encoder,
                    validation_frames,
                    ordered_trial_ids=validation_ids,
                    frames_per_trial=raw.frames_per_trial,
                    hidden_stride=int(
                        self.config.payload["frame_grid"]["hidden_stride"]
                    ),
                    batch_size=hidden_batch,
                    device=self.device,
                )
                train_hidden = TrialSequenceDataset(
                    sequences=train_sequences,
                    labels=train_labels,
                    trial_ids=train_ids,
                    split_role="train",
                )
                validation_hidden = TrialSequenceDataset(
                    sequences=validation_sequences,
                    labels=validation_labels,
                    trial_ids=validation_ids,
                    split_role="validation",
                )
                classifier = _word_classifier(self.config, num_classes=len(class_order))
                classifier_config = self.config.training_config(
                    "classifier_training", seed=initialization_seed, device=self.device
                )
                classifier_checkpoint = unit_root / "classifier_checkpoint.pt"
                classifier_result = train_hidden_classifier(
                    classifier,
                    train_hidden,
                    validation_hidden,
                    config=classifier_config,
                    checkpoint_path=classifier_checkpoint,
                    resume=classifier_checkpoint.exists(),
                    run_context={
                        "config_fingerprint": self.config.fingerprint,
                        "source_identity_fingerprint": source_identity_fingerprint,
                        "split_fingerprint": split.fingerprint,
                        "fold": fold_number,
                        "outer_seed": outer_seed,
                        "initialization_seed": initialization_seed,
                        "training_unit": unit,
                        "target_representation": target_representation,
                        "stage": "hidden_trajectory_to_20_words",
                        "regression_model_fingerprint": regression_result.selected_model_fingerprint,
                    },
                    max_epochs_this_call=max_epochs_this_call,
                )
                if not classifier_result.completed:
                    raise RunIncomplete(
                        f"fold {fold_number} seed {outer_seed} {unit} word head saved at epoch "
                        f"{classifier_result.epochs_completed}; rerun the same command to resume"
                    )
                validation_evaluation = evaluate_hidden_classifier(
                    classifier,
                    validation_hidden,
                    batch_size=int(classifier_config.batch_size),
                    device=self.device,
                    training_config_fingerprint=classifier_result.config_fingerprint,
                    evaluation_seed=initialization_seed,
                )
                validation_receipt_path = unit_root / "validation_prediction_receipt.json"
                _write_or_validate_json(validation_receipt_path, validation_evaluation.receipt)
                reducer_arrays = (
                    shared_root
                    / "reducers"
                    / target_representation
                    / "reducer_arrays.npz"
                )
                fixed_receipt = {
                    "schema_version": 1,
                    "kind": "vocalmind_validation_fixed_unit",
                    "fold": fold_number,
                    "outer_seed": outer_seed,
                    "initialization_seed": initialization_seed,
                    "training_unit": unit,
                    "target_representation": target_representation,
                    "config_fingerprint": self.config.fingerprint,
                    "split_fingerprint": split.fingerprint,
                    "target_reducer_sha256": sha256_file(reducer_arrays),
                    "regression_checkpoint_sha256": regression_result.checkpoint_sha256,
                    "regression_best_epoch": regression_result.best_epoch,
                    "regression_best_validation_loss": regression_result.best_validation_loss,
                    "classifier_checkpoint_sha256": classifier_result.checkpoint_sha256,
                    "classifier_best_epoch": classifier_result.best_epoch,
                    "classifier_best_validation_loss": classifier_result.best_validation_loss,
                    "validation_prediction_fingerprint": validation_evaluation.receipt["fingerprint"],
                    "frame_window_storage": {
                        "train": train_frames.storage_receipt(),
                        "validation": validation_frames.storage_receipt(),
                    },
                    "model_selection": "validation_loss_only",
                    "threshold_selection": "not_applicable_closed_set_argmax",
                    "test_data_opened": False,
                }
                fixed_receipt["fingerprint"] = fingerprint_json(fixed_receipt)
                fixed_path = unit_root / "validation_fixed.json"
                _write_or_validate_json(fixed_path, fixed_receipt)
                gate_unit = f"seed{outer_seed}_{unit}"
                artifact_hash = sha256_file(fixed_path)
                run_fingerprint = fingerprint_json(
                    {
                        "config_fingerprint": self.config.fingerprint,
                        "split_fingerprint": split.fingerprint,
                        "outer_seed": outer_seed,
                        "initialization_seed": initialization_seed,
                        "training_unit": unit,
                        "target_representation": target_representation,
                    }
                )
                if gate_unit in gate.missing_units():
                    gate.mark_completed(
                        unit=gate_unit,
                        artifact_sha256=artifact_hash,
                        run_fingerprint=run_fingerprint,
                    )
                else:
                    completion = read_json(
                        fold_root / "test_gate" / "completed" / f"{gate_unit}.json"
                    )
                    completion_fingerprint = completion.pop("fingerprint", None)
                    if completion_fingerprint != fingerprint_json(completion):
                        raise RuntimeError(f"gate completion fingerprint mismatch: {gate_unit}")
                    if (
                        completion.get("artifact_sha256") != artifact_hash
                        or completion.get("run_fingerprint") != run_fingerprint
                    ):
                        raise RuntimeError(f"gate completion differs from fixed unit: {gate_unit}")
                encoder.to("cpu")
                classifier.to("cpu")
                fixed_models[(outer_seed, unit)] = FixedUnitModels(
                    encoder=encoder,
                    classifier=classifier,
                    regression_result=regression_result,
                    classifier_result=classifier_result,
                )

        # This is the first point at which numeric held-out ECoG may be read.
        test_ids = gate.open_test()
        test_trials = _load_trials(self.adapter, test_ids, self.neural_preprocessor)
        test_labels = _labels(self.adapter, test_ids, class_order)
        dummy_targets = np.zeros(
            (len(test_ids) * raw.frames_per_trial, self.config.target_dim), dtype=np.float32
        )
        test_frames = _frame_dataset(
            trial_arrays=test_trials,
            trial_ids=test_ids,
            targets=dummy_targets,
            standardizer=standardizer,
            split_role="held_out_test",
            split_fingerprint=split.fingerprint,
            regression_frame_hz=int(
                self.config.payload["frame_grid"]["regression_frame_hz"]
            ),
            window_samples=int(self.config.payload["frame_grid"]["window_samples"]),
        )
        seed_summaries = []
        for seed in self.config.seeds:
            predictions: dict[str, EvaluationResult] = {}
            seed_units = planned_training_units(self.config, outer_seed=seed)
            for planned_unit in seed_units:
                unit = planned_unit.key
                fixed = fixed_models[(seed, unit)]
                sequences = encode_hidden_sequences(
                    fixed.encoder,
                    test_frames,
                    ordered_trial_ids=test_ids,
                    frames_per_trial=raw.frames_per_trial,
                    hidden_stride=int(
                        self.config.payload["frame_grid"]["hidden_stride"]
                    ),
                    batch_size=int(self.config.payload["hidden_extraction"]["batch_size"]),
                    device=self.device,
                )
                test_hidden = TrialSequenceDataset(
                    sequences=sequences,
                    labels=test_labels,
                    trial_ids=test_ids,
                    split_role="held_out_test",
                    split_fingerprint=split.fingerprint,
                )
                predictions[unit] = evaluate_hidden_classifier(
                    fixed.classifier,
                    test_hidden,
                    batch_size=int(self.config.payload["classifier_training"]["batch_size"]),
                    device=self.device,
                    training_config_fingerprint=fixed.classifier_result.config_fingerprint,
                    evaluation_seed=planned_unit.initialization_seed,
                    test_gate_authorization=gate.authorization(),
                )
            layer_outputs = {
                layer: LayerProbabilities.create(
                    predictions[f"L{layer}"].sample_ids,
                    predictions[f"L{layer}"].predictions,
                )
                for layer in WHISPER_LAYERS
            }
            ensemble = fixed_l345_probability_ensemble(layer_outputs)
            mel_keys = tuple(
                unit.key
                for unit in seed_units
                if unit.target_representation == "mel"
            )
            _, mel_probability_mean = arithmetic_probability_mean(
                [predictions[key] for key in mel_keys]
            )
            metrics = {
                unit: closed_set_metrics(
                    result.predictions, test_labels, class_order=class_order
                )
                for unit, result in predictions.items()
            }
            metrics["L3+L4+L5"] = closed_set_metrics(
                ensemble.probabilities,
                test_labels,
                class_order=class_order,
            )
            if self.config.run_scope == "production":
                metrics["MELx3"] = closed_set_metrics(
                    mel_probability_mean,
                    test_labels,
                    class_order=class_order,
                )
            seed_result = {
                "schema_version": 1,
                "kind": "vocalmind_primary_seed_result",
                "fold": fold_number,
                "seed": seed,
                "class_order": list(class_order),
                "test_trial_ids": list(test_ids),
                "metrics": metrics,
                "selection": {
                    "model": "validation_loss_only",
                    "threshold": "not_applicable_closed_set_argmax",
                    "test_used_for_selection": False,
                },
                "ensemble": {
                    "layers": list(WHISPER_LAYERS),
                    "rule": ensemble.rule,
                    "subset_search": False,
                },
                "mel_compute_matched_control": {
                    "matching_scope": "three_downstream_branches_not_exact_total_flops",
                    "mode": (
                        "required_three_initialization_probability_mean"
                        if self.config.run_scope == "production"
                        else "single_initialization_development_only"
                    ),
                    "training_units": list(mel_keys),
                    "initialization_seeds": [
                        unit.initialization_seed
                        for unit in seed_units
                        if unit.target_representation == "mel"
                    ],
                    "rule": "arithmetic_mean_of_softmax_probabilities",
                    "subset_search": False,
                },
                "primary_contrast": (
                    "L3+L4+L5_vs_MELx3"
                    if self.config.run_scope == "production"
                    else "not_claimed_development_only"
                ),
                "secondary_contrasts": ["L3+L4+L5_vs_single_MEL", "L3+L4+L5_vs_L4"],
                "prediction_receipts": {
                    unit: result.receipt for unit, result in predictions.items()
                },
            }
            seed_root = fold_root / f"seed_{seed}"
            prediction_path = seed_root / "held_out_test_predictions.npz"
            prediction_arrays = {
                "trial_ids": np.asarray(test_ids, dtype=np.str_),
                "labels": np.asarray(test_labels, dtype=np.int64),
                **{
                    key: np.asarray(value.predictions, dtype=np.float32)
                    for key, value in predictions.items()
                },
                "L345": np.asarray(ensemble.probabilities, dtype=np.float32),
                "MEL_mean": np.asarray(mel_probability_mean, dtype=np.float32),
            }
            seed_result["prediction_npz_sha256"] = _write_or_validate_npz(
                prediction_path,
                **prediction_arrays,
            )
            seed_result["fingerprint"] = fingerprint_json(seed_result)
            _write_or_validate_json(seed_root / "result.json", seed_result)
            seed_summaries.append(
                {
                    "seed": seed,
                    "result_fingerprint": seed_result["fingerprint"],
                    "metrics": metrics,
                }
            )
        return {
            "fold": fold_number,
            "split_fingerprint": split.fingerprint,
            "test_gate_open": True,
            "test_repetition": int(fold["test_repetition"]),
            "validation_repetition": int(fold["validation_repetition"]),
            "seeds": seed_summaries,
        }


def _config_summary(config: PrimaryConfig) -> dict[str, Any]:
    numerical_run_allowed = bool(
        config.run_scope == "production" and config.status == "frozen_confirmatory"
    )
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "config_fingerprint": config.fingerprint,
        "scope": config.run_scope,
        "status": config.status,
        "folds": list(config.folds),
        "seeds": list(config.seeds),
        "representations": list(REPRESENTATIONS),
        "target_dim": config.target_dim,
        "threshold_policy": config.payload["selection"]["threshold"],
        "numerical_run_allowed": numerical_run_allowed,
        "production_runnable": numerical_run_allowed,
        "test_gate_open": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--config", type=Path, required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--data-root", type=Path, required=True)
    plan_parser.add_argument("--json-out", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument(
        "--max-epochs-this-call",
        type=int,
        help="bounded resumable diagnostic; omit for an uninterrupted stage",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_primary_config(args.config)
    if args.command == "validate-config":
        print(json.dumps(_config_summary(config), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "plan":
        # A temporary output path outside source/data is not needed: plan is read-only.
        runner = object.__new__(VocalMindPrimaryRunner)
        runner.config = config
        runner.data_root = Path(args.data_root).expanduser().resolve()
        runner.output_root = Path.cwd().resolve().parent / ".unused-vocalmind-plan-output"
        runner.device = "cpu"
        runner.contract = DEFAULT_VOCALMIND_CONTRACT
        runner.adapter = VocalMindAdapter(runner.data_root)
        runner.neural_preprocessor = VocalMindNeuralPreprocessor.from_mapping(
            dict(config.payload["neural_preprocessing"]),
            expected_channel_count=len(DEFAULT_VOCALMIND_CONTRACT.channel_ids),
        )
        runner.target_provider_factory = None
        if not runner.data_root.is_dir():
            raise FileNotFoundError(f"VocalMind data root does not exist: {runner.data_root}")
        plan = runner.plan()
        plan["source_identity"] = capture_source_identity()
        plan["fingerprint"] = fingerprint_json(plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        if args.json_out:
            atomic_write_json(args.json_out, plan, overwrite=False)
        return 0
    runner = VocalMindPrimaryRunner(
        config=config,
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
    )
    result = runner.run(max_epochs_this_call=args.max_epochs_this_call)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunIncomplete as exc:
        print(f"INCOMPLETE: {exc}")
        raise SystemExit(3)
    except (PrimaryConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
