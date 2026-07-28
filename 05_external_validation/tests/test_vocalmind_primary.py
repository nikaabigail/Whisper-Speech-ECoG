from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from whisper_ecog_ext.model import OneSecondEcogEncoder
from whisper_ecog_ext.data.vocalmind import DEFAULT_VOCALMIND_CONTRACT
from whisper_ecog_ext.neural_data import (
    FrameWindowDataset,
    fit_train_only_channel_standardizer,
)
from whisper_ecog_ext.vocalmind_neural import VocalMindNeuralPreprocessor
from whisper_ecog_ext.vocalmind_primary import (
    PrimaryConfigError,
    VocalMindPrimaryRunner,
    build_frame_spec,
    planned_training_units,
    stereo_to_mono,
    valid_frame_end_samples,
    validate_primary_config,
)
from whisper_ecog_ext.vocalmind_targets import VocalMindAuthorMelTargetExtractor


ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = ROOT / "configs" / "experiments" / "vocalmind_pilot.json"
PRODUCTION_CONFIG = (
    ROOT / "configs" / "experiments" / "vocalmind_primary_production.json"
)


def _config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_configs_pin_strict_time_grid_and_compute_matched_control() -> None:
    pilot = validate_primary_config(_config(PILOT_CONFIG))
    production = validate_primary_config(_config(PRODUCTION_CONFIG))
    assert pilot.payload["frame_grid"]["regression_frame_hz"] == 1000
    assert pilot.payload["frame_grid"]["hidden_stride"] == 10
    assert pilot.payload["frame_grid"]["classifier_frame_hz"] == 100
    assert pilot.payload["hidden_extraction"]["batch_size"] >= 32
    assert (
        pilot.payload["targets"]["mel"]["waveform_normalization"]
        == "per_trial_peak_abs_epsilon_1e-8_shared_with_whisper"
    )
    assert pilot.payload["targets"]["whisper"]["peak_normalize"] is True
    assert [unit.key for unit in planned_training_units(pilot, outer_seed=4)] == [
        "mel",
        "L3",
        "L4",
        "L5",
    ]
    production_units = planned_training_units(production, outer_seed=4)
    assert [unit.key for unit in production_units] == [
        "mel",
        "mel_init1",
        "mel_init2",
        "L3",
        "L4",
        "L5",
    ]
    assert [unit.initialization_seed for unit in production_units[:3]] == [4, 1004, 2004]


def test_50_hz_is_accepted_only_when_explicitly_fast_smoke() -> None:
    value = _config(PILOT_CONFIG)
    value["frame_grid"].update(
        {
            "regression_frame_hz": 50,
            "hidden_stride": 1,
            "classifier_frame_hz": 50,
            "strict_historical_parity": False,
        }
    )
    with pytest.raises(PrimaryConfigError, match="regression_frame_hz"):
        validate_primary_config(value)
    value["run_scope"] = "fast_smoke"
    fast = validate_primary_config(value)
    assert fast.run_scope == "fast_smoke"


def test_strict_valid_grid_has_2000_unpadded_windows_and_100_hz_hidden() -> None:
    ends = valid_frame_end_samples(3000, sample_rate_hz=1000, frame_hz=1000)
    assert len(ends) == 2000
    assert (int(ends[0]), int(ends[-1])) == (1000, 2999)
    starts = ends - 1000
    assert int(starts.min()) == 0
    assert int(ends.max()) < 3000
    assert len(ends[::10]) == 200
    fast = valid_frame_end_samples(3000, sample_rate_hz=1000, frame_hz=50)
    assert len(fast) == 100
    assert (int(fast[0]), int(fast[-1])) == (1000, 2980)


def test_author_mel_matches_pinned_official_formula() -> None:
    import librosa

    rng = np.random.default_rng(17)
    audio = rng.normal(scale=0.03, size=16_000).astype(np.float32)
    extractor = VocalMindAuthorMelTargetExtractor()
    times, actual = extractor.extract_native(audio, 16_000)
    stft = librosa.stft(
        audio,
        n_fft=1024,
        hop_length=320,
        win_length=1024,
        window="hann",
        center=True,
        pad_mode="constant",
    )
    basis = librosa.filters.mel(
        sr=16_000,
        n_fft=1024,
        n_mels=80,
        fmin=80,
        fmax=7600,
    )
    expected = np.log10(np.maximum(1e-6, basis @ np.abs(stft))).T.astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(times, np.arange(len(times)) * 320 / 16_000)
    provenance = extractor.provenance()
    assert provenance["spectrum"] == "magnitude_not_power"
    assert provenance["source_revision"].startswith("e1202bab")
    assert provenance["waveform_normalization"] == "none_official_fidelity"


def test_matched_mel_uses_same_peak_policy_as_whisper() -> None:
    rng = np.random.default_rng(23)
    audio = rng.normal(scale=0.02, size=16_000).astype(np.float32)
    matched = VocalMindAuthorMelTargetExtractor(peak_normalize=True)
    _, reference = matched.extract_native(audio, 16_000)
    _, rescaled = matched.extract_native(audio * 0.125, 16_000)
    np.testing.assert_allclose(reference, rescaled, rtol=2e-6, atol=2e-6)
    provenance = matched.provenance()
    assert provenance["official_fidelity_difference"] is True
    assert (
        provenance["waveform_normalization"]
        == "per_trial_peak_abs_epsilon_1e-8_shared_with_whisper"
    )


def test_development_primary_is_metadata_plan_only_and_never_loads_trials(
    tmp_path: Path,
) -> None:
    contract = DEFAULT_VOCALMIND_CONTRACT
    records = [
        {
            "trial_id": f"vocalized_word:{word}:rep{repetition:02d}",
            "word": word,
            "repetition": repetition,
        }
        for word in contract.words
        for repetition in (*contract.primary_repetitions, contract.secondary_repetition)
        if (word, repetition) not in contract.expected_missing
    ]

    class MetadataOnlyAdapter:
        numeric_trial_ids: list[str] = []

        def build_index(self, *, deep: bool, hash_files: bool) -> dict:
            assert deep is False
            assert hash_files is False
            return {
                "dataset_index_sha256": "0" * 64,
                "counts": {
                    "all_paired_trials": 119,
                    "primary_trials_reps_1_5": 100,
                    "secondary_rep6_trials": 19,
                },
                "trials": records,
            }

        def load_eeg(self, trial) -> np.ndarray:
            self.numeric_trial_ids.append(str(trial.trial_id))
            raise AssertionError("development plan attempted numeric EEG access")

        def load_audio(self, trial) -> np.ndarray:
            self.numeric_trial_ids.append(str(trial.trial_id))
            raise AssertionError("development plan attempted numeric audio access")

    runner = object.__new__(VocalMindPrimaryRunner)
    runner.config = validate_primary_config(_config(PILOT_CONFIG))
    runner.contract = contract
    runner.adapter = MetadataOnlyAdapter()
    runner.neural_preprocessor = VocalMindNeuralPreprocessor(
        expected_channel_count=len(contract.channel_ids)
    )
    runner.output_root = tmp_path / "must_not_be_created"
    plan = runner.plan()
    assert plan["test_gate_open"] is False
    assert plan["numerical_training_allowed"] is False
    assert runner.adapter.numeric_trial_ids == []
    with pytest.raises(PrimaryConfigError, match="strictly plan-only"):
        runner.run()
    assert runner.adapter.numeric_trial_ids == []
    assert not runner.output_root.exists()


def test_mono_is_arithmetic_mean_without_channel_selection() -> None:
    stereo = np.asarray([[1.0, -1.0], [0.25, 0.75]], dtype=np.float32)
    mono = stereo_to_mono(stereo)
    np.testing.assert_array_equal(mono, [0.0, 0.5])
    assert not mono.flags.writeable


def test_shared_neural_preprocessing_and_train_only_scaling_cpu_integration() -> None:
    rng = np.random.default_rng(9)
    time = np.arange(3000, dtype=np.float64) / 1000.0
    common = 4.0 * np.sin(2 * np.pi * 50 * time)
    train_raw = np.stack(
        [
            common + np.sin(2 * np.pi * frequency * time) + 0.1 * rng.normal(size=3000)
            for frequency in (20, 30, 40, 60)
        ],
        axis=1,
    )
    validation_raw = train_raw * 1.3 + 2.0
    held_out_raw = train_raw * 50.0 - 100.0
    preprocessor = VocalMindNeuralPreprocessor(expected_channel_count=4)
    train = preprocessor.transform(train_raw)
    validation = preprocessor.transform(validation_raw)
    held_out = preprocessor.transform(held_out_raw)
    np.testing.assert_array_equal(train, preprocessor.transform(train_raw))
    assert np.max(np.abs(train.mean(axis=1))) < 2e-5
    assert preprocessor.provenance()["representation_independent"] is True
    assert preprocessor.provenance()["per_trial_z_score"] is False

    standardizer = fit_train_only_channel_standardizer(
        [train], ["train-trial"], split_role="train"
    )
    repeated = fit_train_only_channel_standardizer(
        [train], ["train-trial"], split_role="train"
    )
    np.testing.assert_array_equal(standardizer.mean, repeated.mean)
    # held_out is deliberately extreme but never enters fitted statistics.
    assert float(np.std(standardizer.transform(held_out))) > 10.0

    frame_spec = build_frame_spec(["train-trial"], trial_samples=3000, frame_hz=1000)
    targets = np.zeros((len(frame_spec.sample_ids), 50), dtype=np.float32)
    dataset = FrameWindowDataset(
        trials={"train-trial": train},
        sample_ids=frame_spec.sample_ids,
        frame_trial_ids=frame_spec.trial_ids,
        frame_times_s=frame_spec.frame_times_s,
        targets=targets,
        split_role="train",
        standardizer=standardizer,
    )
    assert dataset.storage_receipt()["materialized_window_count"] == 0
    batch = torch.stack([dataset[0]["inputs"], dataset[1]["inputs"]])
    model = OneSecondEcogEncoder(input_channels=4, target_dim=50)
    output = model(batch)
    loss = output.square().mean()
    loss.backward()
    assert output.shape == (2, 50)
    assert torch.isfinite(loss)


def test_config_rejects_non_author_mel_or_per_trial_zscore() -> None:
    value = _config(PILOT_CONFIG)
    bad_mel = copy.deepcopy(value)
    bad_mel["targets"]["mel"]["n_fft"] = 400
    with pytest.raises(PrimaryConfigError, match="targets.mel.n_fft"):
        validate_primary_config(bad_mel)
    bad_neural = copy.deepcopy(value)
    bad_neural["neural_preprocessing"]["per_trial_z_score"] = True
    with pytest.raises(PrimaryConfigError, match="per_trial_z_score"):
        validate_primary_config(bad_neural)
