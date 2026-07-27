from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from whisper_ecog_ext.classifier import HiddenSequenceClassifier
from whisper_ecog_ext.model import OneSecondEcogEncoder
from whisper_ecog_ext.neural_data import FrameWindowDataset
from whisper_ecog_ext.reducer import fit_train_only_reducer
from whisper_ecog_ext.targets import WhisperLayerTargetExtractor
from whisper_ecog_ext.vocalmind_primary import encode_hidden_sequences, load_primary_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ROOT = REPOSITORY_ROOT / "02_whisper_sync" / "library"


def _load_historical_module(name: str, filename: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        name, HISTORICAL_ROOT / filename
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _copy_mapped_state(
    target: torch.nn.Module,
    source: torch.nn.Module,
    prefix_map: dict[str, str],
) -> None:
    source_state = source.state_dict()
    mapped = {}
    for target_name in target.state_dict():
        matches = [prefix for prefix in prefix_map if target_name.startswith(prefix)]
        assert len(matches) == 1, target_name
        target_prefix = matches[0]
        source_name = prefix_map[target_prefix] + target_name[len(target_prefix) :]
        mapped[target_name] = source_state[source_name].detach().clone()
    target.load_state_dict(mapped, strict=True)


def test_one_second_encoder_is_numerically_equivalent_to_historical_simplenet() -> None:
    historical = _load_historical_module("historical_models_regression", "models_regression.py")
    torch.manual_seed(5)
    reference = historical.SimpleNet(4, 7, 1000, 0, use_lstm=True).eval()
    candidate = OneSecondEcogEncoder(input_channels=4, target_dim=7).eval()
    _copy_mapped_state(
        candidate,
        reference,
        {
            "unmix.": "unmixing_layer.",
            "unmix_norm.": "unmixed_channels_batchnorm.",
            "band_filter.": "detector.0.",
            "band_norm.": "detector.1.",
            "envelope_smoother.": "detector.3.",
            "temporal_model.": "lstm.",
            "hidden_norm.": "features_batchnorm.",
            "projection.": "fc_layer.0.",
        },
    )
    inputs = torch.randn(3, 4, 1001)
    with torch.inference_mode():
        reference_hidden = reference(inputs, return_hidden=True)
        candidate_hidden = candidate(inputs, return_hidden=True)
        reference_output = reference.fc_layer(reference_hidden)
        candidate_output = candidate(inputs)
    torch.testing.assert_close(candidate_hidden, reference_hidden, rtol=0, atol=1e-7)
    torch.testing.assert_close(candidate_output, reference_output, rtol=0, atol=1e-7)
    assert candidate.hidden_dim == reference.final_out_features == 3030


def test_hidden_classifier_is_numerically_equivalent_to_mel2wordhidden() -> None:
    historical = _load_historical_module(
        "historical_models_classification", "models_classification.py"
    )
    torch.manual_seed(8)
    reference = historical.Mel2WordHidden(12, 5).eval()
    candidate = HiddenSequenceClassifier(input_features=12, num_classes=5).eval()
    _copy_mapped_state(
        candidate,
        reference,
        {
            "temporal_convolution.": "mels2features.0.",
            "temporal_lstm.": "lstm.",
            "classifier.": "fc_layer.0.",
        },
    )
    inputs = torch.randn(4, 12, 45)
    with torch.inference_mode():
        expected = reference(inputs)
        actual = candidate(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    receipt = candidate.architecture_receipt()
    assert receipt["pre_downsampling"] == 1
    assert receipt["convolution_kernel"] == 10
    assert receipt["pool_kernel"] == 10
    assert receipt["lstm_hidden"] == 100


def test_frame_window_matches_historical_lag_1000_0_slice_formula() -> None:
    historical = _load_historical_module("historical_runner_common", "runner_common.py")
    neural = np.arange(1_005 * 3, dtype=np.float32).reshape(1_005, 3)
    targets = np.arange(1_005 * 2, dtype=np.float32).reshape(1_005, 2)
    generator = historical.data_generator(
        neural,
        targets,
        batch_size=2,
        lag_backward=1000,
        lag_forward=0,
        shuffle=False,
        infinite=False,
    )
    historical_windows, historical_targets = next(generator)
    candidate = FrameWindowDataset(
        trials={"trial": neural},
        sample_ids=("frame-1000", "frame-1001"),
        frame_trial_ids=("trial", "trial"),
        frame_times_s=(1.0, 1.001),
        targets=targets[[1000, 1001]],
        split_role="train",
    )
    candidate_windows = np.stack([candidate[index]["inputs"].numpy() for index in range(2)])
    candidate_targets = np.stack([candidate[index]["target"].numpy() for index in range(2)])
    np.testing.assert_array_equal(candidate_windows, historical_windows)
    np.testing.assert_array_equal(candidate_targets, historical_targets)


def test_train_only_reducer_matches_standard_scaler_pca_whitening_formula() -> None:
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(44)
    features = rng.normal(size=(120, 80)).astype(np.float32)
    sample_ids = tuple(f"train-{index:03d}" for index in range(len(features)))
    candidate = fit_train_only_reducer(
        features,
        sample_ids,
        n_components=50,
        whiten=True,
        seed=4,
        split_role="train",
    )
    reference = make_pipeline(
        StandardScaler(),
        PCA(n_components=50, whiten=True, svd_solver="full", random_state=4),
    ).fit(features)
    np.testing.assert_allclose(
        candidate.transform(features[:11]),
        reference.transform(features[:11]),
        # The new artifact deliberately fits/stores in float64; the historical
        # pipeline inherited float32 from its cache. The formula is the same.
        rtol=2e-4,
        atol=2e-4,
    )


class _RecordingFeatureExtractor:
    def __init__(self) -> None:
        self.audio = None

    def __call__(self, audio, **kwargs):
        del kwargs
        self.audio = np.asarray(audio)
        return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))


class _CountingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_features, **kwargs):
        del input_features, kwargs
        self.calls += 1
        return SimpleNamespace(
            hidden_states=tuple(
                torch.full((1, 1500, 4), float(layer)) for layer in range(7)
            )
        )


def test_whisper_l345_is_peak_normalized_pinned_and_single_forward() -> None:
    feature_extractor = _RecordingFeatureExtractor()
    encoder = _CountingEncoder()
    extractor = WhisperLayerTargetExtractor(
        revision="a" * 40,
        encoder=encoder,
        feature_extractor=feature_extractor,
        device="cpu",
    )
    audio = (7.0 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000)).astype(
        np.float32
    )
    outputs = extractor.extract_aligned(
        audio, 16_000, np.array([0.01, 0.25, 0.75, 0.99])
    )
    assert encoder.calls == 1
    assert tuple(outputs) == (3, 4, 5)
    assert np.max(np.abs(feature_extractor.audio)) == 1.0
    provenance = extractor.provenance()
    assert provenance["revision"] == "a" * 40
    assert provenance["peak_normalize"] is True


class _IndexedFrameDataset(Dataset):
    def __init__(self, frame_count: int) -> None:
        self.sample_ids = tuple(f"frame-{index:03d}" for index in range(frame_count))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        return {
            "inputs": torch.tensor([float(index)], dtype=torch.float32),
            "target": torch.tensor([0.0], dtype=torch.float32),
            "sample_id": self.sample_ids[index],
        }


class _HiddenIdentity(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor:
        assert return_hidden is True
        return torch.cat((inputs, inputs + 100.0), dim=1)


def test_production_grid_is_1000_hz_then_stride_ten_to_100_hz() -> None:
    config = load_primary_config(
        REPOSITORY_ROOT
        / "05_external_validation"
        / "configs"
        / "experiments"
        / "vocalmind_primary_production.json"
    )
    grid = config.payload["frame_grid"]
    assert grid["regression_frame_hz"] == 1000
    assert grid["hidden_stride"] == 10
    assert grid["classifier_frame_hz"] == 100
    assert grid["strict_historical_parity"] is True


def test_hidden_stride_resets_at_each_trial_boundary() -> None:
    # A small stride=3 analogue makes every selected position visible. The
    # production config applies the same rule with 1000 Hz / 10 = 100 Hz.
    sequences = encode_hidden_sequences(
        _HiddenIdentity(),
        _IndexedFrameDataset(14),
        ordered_trial_ids=("trial-a", "trial-b"),
        frames_per_trial=7,
        hidden_stride=3,
        batch_size=4,
        device="cpu",
    )
    expected = np.array(
        [
            [[0.0, 3.0, 6.0], [100.0, 103.0, 106.0]],
            [[7.0, 10.0, 13.0], [107.0, 110.0, 113.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(sequences, expected)
