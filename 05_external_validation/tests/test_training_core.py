from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from whisper_ecog_ext.classifier import HiddenSequenceClassifier
from whisper_ecog_ext.evaluation import (
    evaluate_hidden_classifier,
    evaluate_regression,
)
from whisper_ecog_ext.model import OneSecondEcogEncoder
from whisper_ecog_ext.neural_data import (
    ChannelStandardizerArtifact,
    FrameWindowDataset,
    fit_train_only_channel_standardizer,
)
from whisper_ecog_ext.protocol import SplitManifest, TestGate, TestGateClosed
from whisper_ecog_ext.training import (
    TrainingConfig,
    train_hidden_classifier,
    train_regression,
)


def test_channel_standardizer_is_train_only_streamed_and_portable(tmp_path: Path) -> None:
    trials = (
        np.array([[1.0, 5.0], [3.0, 5.0]]),
        np.array([[5.0, 5.0], [7.0, 5.0]]),
    )
    with pytest.raises(ValueError, match="split_role='train'"):
        fit_train_only_channel_standardizer(
            trials, ("trial-a", "trial-b"), split_role="validation"
        )
    fitted = fit_train_only_channel_standardizer(
        trials, ("trial-a", "trial-b"), split_role="train"
    )
    np.testing.assert_allclose(fitted.mean, [4.0, 5.0])
    np.testing.assert_allclose(fitted.variance, [5.0, 0.0])
    np.testing.assert_allclose(fitted.scale, [np.sqrt(5.0), 1.0])
    combined = np.concatenate(trials, axis=0)
    transformed = fitted.transform(combined)
    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-7)

    artifact_path = tmp_path / "neural_scaler"
    fitted.save(artifact_path)
    loaded = ChannelStandardizerArtifact.load(artifact_path)
    np.testing.assert_array_equal(loaded.mean, fitted.mean)
    np.testing.assert_array_equal(loaded.transform(combined), transformed)
    with pytest.raises(FileExistsError):
        fitted.save(artifact_path)


def test_frame_window_dataset_slices_exact_1001_samples_without_padding() -> None:
    trial = np.arange(1_201 * 2, dtype=np.float32).reshape(1_201, 2)
    scaler = fit_train_only_channel_standardizer(
        (trial,), ("trial-a",), split_role="train"
    )
    dataset = FrameWindowDataset(
        trials={"trial-a": trial},
        sample_ids=("frame-1000",),
        frame_trial_ids=("trial-a",),
        frame_times_s=(1.0,),
        targets=np.array([[0.25, -0.5]], dtype=np.float32),
        split_role="train",
        standardizer=scaler,
    )
    item = dataset[0]
    assert item["inputs"].shape == (2, 1001)
    expected = scaler.transform(trial[:1001]).T
    np.testing.assert_array_equal(item["inputs"].numpy(), expected)
    assert dataset.records[0].start_index == 0
    assert dataset.records[0].end_index == 1000


def test_whole_trial_pretransform_matches_per_window_and_has_bounded_storage() -> None:
    rng = np.random.default_rng(20260727)
    trial = rng.normal(size=(1_301, 3)).astype(np.float32)
    scaler = fit_train_only_channel_standardizer(
        (trial,), ("trial-a",), split_role="train"
    )
    common = {
        "trials": {"trial-a": trial},
        "sample_ids": ("frame-1000", "frame-1100", "frame-1300"),
        "frame_trial_ids": ("trial-a",) * 3,
        "frame_times_s": (1.0, 1.1, 1.3),
        "targets": np.arange(6, dtype=np.float32).reshape(3, 2),
        "split_role": "train",
        "standardizer": scaler,
    }
    pretransformed = FrameWindowDataset(
        **common,
        standardization_mode="pretransform_trials",
    )
    reference = FrameWindowDataset(
        **common,
        standardization_mode="per_window",
    )

    for index in range(len(pretransformed)):
        np.testing.assert_array_equal(
            pretransformed[index]["inputs"].numpy(),
            reference[index]["inputs"].numpy(),
        )

    stored_trial = pretransformed.trials["trial-a"]
    assert stored_trial.dtype == np.float32
    assert stored_trial.flags.c_contiguous
    assert stored_trial.flags.writeable is False
    assert not np.shares_memory(stored_trial, trial)
    receipt = pretransformed.storage_receipt()
    assert receipt["standardization_mode"] == "pretransform_trials"
    assert receipt["per_item_standardization"] is False
    assert receipt["materialized_window_count"] == 0
    assert receipt["stored_time_samples"] == trial.shape[0]
    assert receipt["stored_bytes"] == trial.size * np.dtype(np.float32).itemsize
    assert receipt["stored_dtype"] == "float32"

    reference_receipt = reference.storage_receipt()
    assert reference_receipt["standardization_mode"] == "per_window"
    assert reference_receipt["per_item_standardization"] is True
    assert reference_receipt["materialized_window_count"] == 0


def test_frame_window_dataset_rejects_boundary_and_off_grid_frames() -> None:
    trial = np.zeros((1_201, 2), dtype=np.float32)
    common = {
        "trials": {"trial-a": trial},
        "sample_ids": ("frame",),
        "frame_trial_ids": ("trial-a",),
        "targets": np.zeros((1, 1), dtype=np.float32),
        "split_role": "train",
    }
    with pytest.raises(ValueError, match="boundary padding"):
        FrameWindowDataset(frame_times_s=(0.5,), **common)
    with pytest.raises(ValueError, match="1000 Hz grid"):
        FrameWindowDataset(frame_times_s=(1.0002,), **common)
    with pytest.raises(ValueError, match="boundary padding"):
        FrameWindowDataset(
            trials={"trial-a": np.zeros((600, 2)), "trial-b": np.zeros((600, 2))},
            sample_ids=("frame",),
            frame_trial_ids=("trial-a",),
            frame_times_s=(0.7,),
            targets=np.zeros((1, 1)),
            split_role="train",
        )


class _TensorDataset(Dataset):
    def __init__(self, inputs, targets, *, split_role: str, prefix: str) -> None:
        self.inputs = torch.as_tensor(inputs)
        self.targets = torch.as_tensor(targets)
        self.split_role = split_role
        self.sample_ids = tuple(f"{prefix}-{index:03d}" for index in range(len(inputs)))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        return {
            "inputs": self.inputs[index],
            "target": self.targets[index],
            "sample_id": self.sample_ids[index],
        }


def _regression_datasets() -> tuple[FrameWindowDataset, FrameWindowDataset]:
    rng = np.random.default_rng(12)
    train_trial = rng.normal(size=(1_801, 2)).astype(np.float32)
    validation_trial = rng.normal(size=(1_401, 2)).astype(np.float32)
    scaler = fit_train_only_channel_standardizer(
        (train_trial,), ("train-trial",), split_role="train"
    )
    train_times = np.arange(1.0, 1.8, 0.1)
    validation_times = np.arange(1.0, 1.4, 0.1)
    train = FrameWindowDataset(
        trials={"train-trial": train_trial},
        sample_ids=tuple(f"train-{index}" for index in range(len(train_times))),
        frame_trial_ids=("train-trial",) * len(train_times),
        frame_times_s=train_times,
        targets=np.column_stack((np.sin(train_times), np.cos(train_times))),
        split_role="train",
        standardizer=scaler,
    )
    validation = FrameWindowDataset(
        trials={"validation-trial": validation_trial},
        sample_ids=tuple(f"validation-{index}" for index in range(len(validation_times))),
        frame_trial_ids=("validation-trial",) * len(validation_times),
        frame_times_s=validation_times,
        targets=np.column_stack((np.sin(validation_times), np.cos(validation_times))),
        split_role="validation",
        standardizer=scaler,
    )
    return train, validation


def _small_regressor() -> OneSecondEcogEncoder:
    return OneSecondEcogEncoder(
        input_channels=2,
        target_dim=2,
        hidden_channels=2,
        filtering_kernel=3,
        envelope_kernel=3,
        use_lstm=False,
    )


def test_regression_resume_is_identical_and_checkpoint_is_complete(tmp_path: Path) -> None:
    train, validation = _regression_datasets()
    config = TrainingConfig(
        seed=17,
        max_epochs=3,
        batch_size=4,
        learning_rate=3e-3,
        patience=10,
        device="cpu",
    )
    full_model = _small_regressor()
    full = train_regression(
        full_model,
        train,
        validation,
        config=config,
        checkpoint_path=tmp_path / "full.pt",
        run_context={"split_fingerprint": "a" * 64},
    )
    resumed_model = _small_regressor()
    interrupted = train_regression(
        resumed_model,
        train,
        validation,
        config=config,
        checkpoint_path=tmp_path / "resumed.pt",
        run_context={"split_fingerprint": "a" * 64},
        max_epochs_this_call=1,
    )
    assert interrupted.completed is False
    assert interrupted.epochs_completed == 1
    resumed = train_regression(
        resumed_model,
        train,
        validation,
        config=config,
        checkpoint_path=tmp_path / "resumed.pt",
        resume=True,
        run_context={"split_fingerprint": "a" * 64},
    )
    assert full.completed and resumed.completed
    assert full.history == resumed.history
    assert full.best_epoch == resumed.best_epoch
    assert full.selected_model_fingerprint == resumed.selected_model_fingerprint

    full_eval = evaluate_regression(
        full_model,
        validation,
        batch_size=2,
        training_config_fingerprint=full.config_fingerprint,
    )
    resumed_eval = evaluate_regression(
        resumed_model,
        validation,
        batch_size=2,
        training_config_fingerprint=resumed.config_fingerprint,
    )
    np.testing.assert_array_equal(full_eval.predictions, resumed_eval.predictions)
    assert full_eval.receipt == resumed_eval.receipt
    receipt_path = tmp_path / "validation_prediction_receipt.json"
    full_eval.save_receipt(receipt_path)
    with pytest.raises(FileExistsError):
        full_eval.save_receipt(receipt_path)

    checkpoint = torch.load(tmp_path / "resumed.pt", map_location="cpu", weights_only=False)
    assert {"optimizer_state", "rng_state", "config_fingerprint", "best_model_state"}.issubset(
        checkpoint
    )
    changed = TrainingConfig(**{**config.__dict__, "learning_rate": 4e-3})
    with pytest.raises(RuntimeError, match="configuration fingerprint"):
        train_regression(
            _small_regressor(),
            train,
            validation,
            config=changed,
            checkpoint_path=tmp_path / "resumed.pt",
            resume=True,
            run_context={"split_fingerprint": "a" * 64},
        )


def test_trainer_rejects_nonfinite_targets_and_wrong_validation_role(tmp_path: Path) -> None:
    inputs = np.zeros((4, 2), dtype=np.float32)
    bad_targets = np.array([[0.0], [np.inf], [0.0], [0.0]], dtype=np.float32)
    train = _TensorDataset(inputs, bad_targets, split_role="train", prefix="train")
    validation = _TensorDataset(
        inputs, np.zeros((4, 1), dtype=np.float32), split_role="validation", prefix="val"
    )
    config = TrainingConfig(max_epochs=1, batch_size=2, patience=1)
    with pytest.raises(FloatingPointError, match="targets"):
        train_regression(
            torch.nn.Linear(2, 1),
            train,
            validation,
            config=config,
            checkpoint_path=tmp_path / "nonfinite.pt",
        )
    with pytest.raises(ValueError, match="split_role='validation'"):
        train_regression(
            torch.nn.Linear(2, 1),
            train,
            train,
            config=config,
            checkpoint_path=tmp_path / "wrong-role.pt",
        )


def test_hidden_classifier_topology_and_validation_selected_training(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    train_inputs = rng.normal(size=(12, 6, 8)).astype(np.float32)
    validation_inputs = rng.normal(size=(6, 6, 8)).astype(np.float32)
    train_labels = (train_inputs.mean(axis=(1, 2)) > 0).astype(np.int64)
    validation_labels = (validation_inputs.mean(axis=(1, 2)) > 0).astype(np.int64)
    train = _TensorDataset(train_inputs, train_labels, split_role="train", prefix="train-cls")
    validation = _TensorDataset(
        validation_inputs,
        validation_labels,
        split_role="validation",
        prefix="val-cls",
    )
    model = HiddenSequenceClassifier(
        input_features=6,
        num_classes=2,
        convolution_channels=8,
        convolution_kernel=3,
        pool_kernel=2,
        lstm_hidden=4,
    )
    assert model(torch.zeros(2, 6, 8)).shape == (2, 2)
    assert model.architecture_receipt()["pre_downsampling"] == 1
    with pytest.raises(ValueError, match="temporal frames"):
        model(torch.zeros(2, 6, 3))

    config = TrainingConfig(
        seed=9,
        max_epochs=2,
        batch_size=3,
        learning_rate=5e-3,
        patience=5,
    )
    result = train_hidden_classifier(
        model,
        train,
        validation,
        config=config,
        checkpoint_path=tmp_path / "classifier.pt",
    )
    assert result.completed
    assert result.receipt()["selected_by"] == "validation_loss_only"
    evaluation = evaluate_hidden_classifier(
        model,
        validation,
        batch_size=2,
        training_config_fingerprint=result.config_fingerprint,
    )
    assert evaluation.predictions.shape == (6, 2)
    np.testing.assert_allclose(evaluation.predictions.sum(axis=1), 1.0, atol=1e-6)
    assert 0.0 <= evaluation.receipt["metrics"]["accuracy"] <= 1.0


def test_caller_supplied_initialization_can_resume_into_fresh_model(tmp_path: Path) -> None:
    inputs = np.arange(16, dtype=np.float32).reshape(8, 2) / 10.0
    targets = inputs[:, :1] - inputs[:, 1:]
    train = _TensorDataset(inputs, targets, split_role="train", prefix="train-warm")
    validation = _TensorDataset(
        inputs[:4], targets[:4], split_role="validation", prefix="val-warm"
    )
    config = TrainingConfig(
        seed=31,
        max_epochs=2,
        batch_size=4,
        patience=5,
        initialize_from_seed=False,
    )
    path = tmp_path / "caller-supplied.pt"
    interrupted = train_regression(
        torch.nn.Linear(2, 1),
        train,
        validation,
        config=config,
        checkpoint_path=path,
        max_epochs_this_call=1,
    )
    assert not interrupted.completed
    resumed = train_regression(
        torch.nn.Linear(2, 1),
        train,
        validation,
        config=config,
        checkpoint_path=path,
        resume=True,
    )
    assert resumed.completed
    assert resumed.epochs_completed == 2


def test_held_out_evaluation_requires_matching_open_test_gate(tmp_path: Path) -> None:
    split = SplitManifest.create(
        dataset_id="synthetic",
        protocol_id="gate-evaluation-v1",
        split_seed=0,
        train_ids=("train-1",),
        validation_ids=("val-1",),
        test_ids=("test-1",),
        purge_gap_seconds=0.0,
        dataset_manifest_sha256="a" * 64,
    )
    gate = TestGate(
        state_directory=tmp_path / "gate",
        split=split,
        required_units=("regressor",),
        protocol_fingerprint="b" * 64,
    )
    gate.mark_completed(
        unit="regressor",
        artifact_sha256="c" * 64,
        run_fingerprint="d" * 64,
    )
    trial = np.zeros((1_101, 2), dtype=np.float32)
    held_out = FrameWindowDataset(
        trials={"test-trial": trial},
        sample_ids=("test-1",),
        frame_trial_ids=("test-trial",),
        frame_times_s=(1.0,),
        targets=np.zeros((1, 1), dtype=np.float32),
        split_role="held_out_test",
        split_fingerprint=split.fingerprint,
    )
    model = OneSecondEcogEncoder(
        input_channels=2,
        target_dim=1,
        hidden_channels=2,
        filtering_kernel=3,
        envelope_kernel=3,
        use_lstm=False,
    )
    with pytest.raises(TestGateClosed, match="Authorization"):
        evaluate_regression(model, held_out)
    with pytest.raises(TestGateClosed, match="explicitly opened"):
        gate.authorization()
    gate.open_test()
    authorization = gate.authorization()
    result = evaluate_regression(
        model, held_out, test_gate_authorization=authorization
    )
    assert result.receipt["test_gate_authorization"]["split_fingerprint"] == (
        split.fingerprint
    )

    wrong_ids = FrameWindowDataset(
        trials={"test-trial": trial},
        sample_ids=("wrong-test-id",),
        frame_trial_ids=("test-trial",),
        frame_times_s=(1.0,),
        targets=np.zeros((1, 1), dtype=np.float32),
        split_role="held_out_test",
        split_fingerprint=split.fingerprint,
    )
    with pytest.raises(TestGateClosed, match="sample IDs/order"):
        evaluate_regression(
            model, wrong_ids, test_gate_authorization=authorization
        )
