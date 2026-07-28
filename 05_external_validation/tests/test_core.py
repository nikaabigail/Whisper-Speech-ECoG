from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from whisper_ecog_ext.ensemble import (
    LayerProbabilities,
    fixed_l345_probability_ensemble,
)
from whisper_ecog_ext.model import OneSecondEcogEncoder
from whisper_ecog_ext.protocol import (
    SplitManifest,
    TestGate,
    TestGateClosed,
    make_swpd_fixed_neural_split,
    make_swpd_rotating_linear_splits,
    swpd_neural_pair_assignment,
)
from whisper_ecog_ext.reducer import ReducerArtifact, fit_train_only_reducer
from whisper_ecog_ext.reproducibility import set_deterministic_seed
from whisper_ecog_ext.targets import (
    MelTargetExtractor,
    WhisperLayerTargetExtractor,
    align_features_local,
)


def test_deterministic_seed_repeats_python_numpy_and_torch() -> None:
    import random

    set_deterministic_seed(123)
    first = (random.random(), np.random.rand(3), torch.rand(3))
    set_deterministic_seed(123)
    second = (random.random(), np.random.rand(3), torch.rand(3))
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_train_only_reducer_round_trip_is_numeric_and_immutable(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(30, 8)).astype(np.float32)
    ids = [f"train:{index:03d}" for index in range(len(values))]
    with pytest.raises(ValueError, match="split_role='train'"):
        fit_train_only_reducer(
            values, ids, n_components=4, split_role="validation"
        )
    fitted = fit_train_only_reducer(
        values, ids, n_components=4, whiten=True, seed=7, split_role="train"
    )
    artifact_dir = tmp_path / "fold_00_l3_reducer"
    fitted.save(artifact_dir)
    loaded = ReducerArtifact.load(artifact_dir)
    np.testing.assert_allclose(
        fitted.transform(values[:5]), loaded.transform(values[:5]), rtol=1e-6, atol=1e-6
    )
    assert loaded.output_dim == 4
    with pytest.raises(FileExistsError):
        fitted.save(artifact_dir)


def test_neighbor_alignment_is_local_linear_interpolation() -> None:
    source_times = np.array([0.0, 1.0, 2.0])
    features = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    aligned = align_features_local(
        source_times, features, np.array([0.25, 1.5], dtype=np.float64)
    )
    np.testing.assert_allclose(aligned[:, 0], [2.5, 15.0])
    with pytest.raises(ValueError, match="edge padding is forbidden"):
        align_features_local(source_times, features, np.array([-0.01]))


class _FakeFeatureExtractor:
    def __call__(self, audio, **kwargs):
        del audio, kwargs
        return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))


class _FakeWhisperEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_features, **kwargs):
        del input_features, kwargs
        self.calls += 1
        states = tuple(
            torch.full((1, 1500, 6), float(layer), dtype=torch.float32)
            for layer in range(7)
        )
        return SimpleNamespace(hidden_states=states)


def test_whisper_l345_share_one_forward_and_one_time_grid() -> None:
    encoder = _FakeWhisperEncoder()
    extractor = WhisperLayerTargetExtractor(
        revision="1" * 40,
        encoder=encoder,
        feature_extractor=_FakeFeatureExtractor(),
        device="cpu",
    )
    audio = np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000).astype(np.float32)
    target_times = np.array([0.01, 0.25, 0.75, 0.99])
    outputs = extractor.extract_aligned(audio, 16_000, target_times)
    assert encoder.calls == 1
    assert tuple(outputs) == (3, 4, 5)
    for layer in (3, 4, 5):
        assert outputs[layer].shape == (4, 6)
        np.testing.assert_array_equal(outputs[layer], np.full((4, 6), layer))


def test_whisper_revision_must_be_exact_commit() -> None:
    with pytest.raises(ValueError, match="exact 40-character"):
        WhisperLayerTargetExtractor(
            revision="main",
            encoder=_FakeWhisperEncoder(),
            feature_extractor=_FakeFeatureExtractor(),
            device="cpu",
        )


def test_mel_target_has_explicit_local_time_alignment() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate) / sample_rate
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    extractor = MelTargetExtractor(n_mels=8)
    aligned = extractor.extract_aligned(
        audio, sample_rate, np.linspace(0.02, 0.9, 12)
    )
    assert aligned.shape == (12, 8)
    assert np.isfinite(aligned).all()
    assert extractor.provenance()["center"] is False
    assert extractor.provenance()["peak_normalize"] is True


def test_mel_peak_normalization_preserves_historical_amplitude_invariance() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate) / sample_rate
    audio = np.sin(2 * np.pi * 330 * time).astype(np.float32)
    extractor = MelTargetExtractor(n_mels=8)
    target_times = np.linspace(0.02, 0.9, 10)
    np.testing.assert_allclose(
        extractor.extract_aligned(audio, sample_rate, target_times),
        extractor.extract_aligned(7.5 * audio, sample_rate, target_times),
        rtol=1e-5,
        atol=1e-5,
    )


def test_primary_mel_defaults_support_train_only_pca50() -> None:
    extractor = MelTargetExtractor()
    provenance = extractor.provenance()
    assert provenance["n_mels"] == 80
    assert provenance["fmax"] == 8_000.0
    assert provenance["sample_rate"] == 16_000


def test_historical_mel_variant_requires_explicit_configuration() -> None:
    historical = MelTargetExtractor(n_mels=40, fmax=2_000.0)
    provenance = historical.provenance()
    assert provenance["n_mels"] == 40
    assert provenance["fmax"] == 2_000.0


def test_one_second_encoder_preserves_1001_to_3030_contract() -> None:
    model = OneSecondEcogEncoder(input_channels=4, target_dim=50).eval()
    inputs = torch.zeros(2, 4, 1001)
    with torch.no_grad():
        hidden = model(inputs, return_hidden=True)
        output = model(inputs)
    assert model.hidden_dim == 3030
    assert hidden.shape == (2, 3030)
    assert output.shape == (2, 50)
    assert model.architecture_receipt()["canonical_sample_rate_hz"] == 1000
    with pytest.raises(ValueError, match="exactly 1001"):
        model(torch.zeros(2, 4, 1000))


def test_fixed_l345_ensemble_checks_exact_sample_identity() -> None:
    ids = ("trial-a", "trial-b")
    l3 = LayerProbabilities.create(ids, [[0.8, 0.2], [0.3, 0.7]])
    l4 = LayerProbabilities.create(ids, [[0.6, 0.4], [0.4, 0.6]])
    l5 = LayerProbabilities.create(ids, [[0.7, 0.3], [0.2, 0.8]])
    result = fixed_l345_probability_ensemble({3: l3, 4: l4, 5: l5})
    np.testing.assert_allclose(result.probabilities, [[0.7, 0.3], [0.3, 0.7]])
    wrong_order = LayerProbabilities.create(ids[::-1], [[0.7, 0.3], [0.2, 0.8]])
    with pytest.raises(ValueError, match="sample IDs/order"):
        fixed_l345_probability_ensemble({3: l3, 4: l4, 5: wrong_order})
    with pytest.raises(ValueError, match="pre-specified"):
        fixed_l345_probability_ensemble({3: l3, 4: l4})


def _split() -> SplitManifest:
    return SplitManifest.create(
        dataset_id="synthetic",
        protocol_id="external-v1",
        split_seed=4,
        train_ids=["train-1", "train-2"],
        validation_ids=["val-1"],
        test_ids=["test-1"],
        purge_gap_seconds=1.0,
        dataset_manifest_sha256="a" * 64,
    )


def test_split_manifest_is_disjoint_fingerprinted_and_immutable(tmp_path: Path) -> None:
    split = _split()
    path = tmp_path / "split_manifest.json"
    split.save(path)
    assert SplitManifest.load(path) == split
    with pytest.raises(FileExistsError):
        split.save(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["train_ids"].append("tampered")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        SplitManifest.load(path)
    with pytest.raises(ValueError, match="disjoint"):
        SplitManifest.create(
            dataset_id="bad",
            protocol_id="bad",
            split_seed=1,
            train_ids=["same"],
            validation_ids=["val"],
            test_ids=["same"],
            purge_gap_seconds=0,
            dataset_manifest_sha256="b" * 64,
        )


def test_swpd_full_neural_split_obeys_subject_pair_rule() -> None:
    pairs = tuple(
        (f"pair-{pair_index}-trial-a", f"pair-{pair_index}-trial-b")
        for pair_index in range(5)
    )
    assignment = swpd_neural_pair_assignment(subject_number=4)
    assert assignment.test_pair_index == 2
    assert assignment.validation_pair_index == 3
    assert assignment.training_pair_indices == (0, 1, 4)
    split = make_swpd_fixed_neural_split(
        subject_number=4,
        adjacent_trial_pairs=pairs,
        dataset_id="swpd-subject-04",
        dataset_manifest_sha256="8" * 64,
    )
    assert split.held_out_test_ids == pairs[2]
    assert split.validation_ids == pairs[3]
    assert split.train_ids == pairs[0] + pairs[1] + pairs[4]
    assert "test-pair-2" in split.protocol_id


def test_swpd_matched_linear_splits_rotate_all_pairs_once() -> None:
    pairs = tuple((f"pair-{pair_index}",) for pair_index in range(5))
    folds = make_swpd_rotating_linear_splits(
        adjacent_trial_pairs=pairs,
        dataset_id="swpd-linear",
        dataset_manifest_sha256="9" * 64,
    )
    assert len(folds) == 5
    assert tuple(fold.held_out_test_ids[0] for fold in folds) == tuple(
        f"pair-{index}" for index in range(5)
    )
    for index, fold in enumerate(folds):
        assert fold.validation_ids == pairs[(index + 1) % 5]
        assert len(fold.train_ids) == 3


def test_swpd_pair_protocol_rejects_non_unique_or_wrong_pair_count() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        make_swpd_rotating_linear_splits(
            adjacent_trial_pairs=(("a",),) * 4,
            dataset_id="bad",
            dataset_manifest_sha256="7" * 64,
        )
    with pytest.raises(ValueError, match="unique across"):
        make_swpd_fixed_neural_split(
            subject_number=2,
            adjacent_trial_pairs=(("duplicate",),) * 5,
            dataset_id="bad",
            dataset_manifest_sha256="7" * 64,
        )
    with pytest.raises(ValueError, match="equal size"):
        make_swpd_fixed_neural_split(
            subject_number=2,
            adjacent_trial_pairs=(("a",), ("b", "c"), ("d",), ("e",), ("f",)),
            dataset_id="bad",
            dataset_manifest_sha256="7" * 64,
        )
    with pytest.raises(ValueError, match="exact integer"):
        swpd_neural_pair_assignment(2.5)


def test_test_gate_releases_ids_only_after_every_required_unit(tmp_path: Path) -> None:
    gate = TestGate(
        state_directory=tmp_path / "gate",
        split=_split(),
        required_units=("mel", "L3", "L4", "L5"),
        protocol_fingerprint="c" * 64,
    )
    with pytest.raises(TestGateClosed):
        gate.test_ids()
    for index, unit in enumerate(("mel", "L3", "L4")):
        gate.mark_completed(
            unit=unit,
            artifact_sha256=f"{index + 1:064x}",
            run_fingerprint="d" * 64,
        )
    with pytest.raises(TestGateClosed, match="L5"):
        gate.open_test()
    gate.mark_completed(
        unit="L5", artifact_sha256="e" * 64, run_fingerprint="d" * 64
    )
    assert gate.open_test() == ("test-1",)
    assert gate.test_ids() == ("test-1",)
    authorization = gate.authorization()
    assert authorization.split_fingerprint == gate.split.fingerprint
    assert authorization.held_out_sample_ids_sha256
    with pytest.raises(FileExistsError):
        gate.mark_completed(
            unit="L5", artifact_sha256="f" * 64, run_fingerprint="d" * 64
        )


def test_existing_test_gate_receipt_cannot_be_reused_with_fewer_units(
    tmp_path: Path,
) -> None:
    state = tmp_path / "gate"
    original = TestGate(
        state_directory=state,
        split=_split(),
        required_units=("L3", "L4"),
        protocol_fingerprint="c" * 64,
    )
    for unit, artifact in (("L3", "1" * 64), ("L4", "2" * 64)):
        original.mark_completed(
            unit=unit, artifact_sha256=artifact, run_fingerprint="d" * 64
        )
    original.open_test()
    weakened = TestGate(
        state_directory=state,
        split=_split(),
        required_units=("L3",),
        protocol_fingerprint="c" * 64,
    )
    with pytest.raises(RuntimeError, match="another experiment"):
        weakened.open_test()
