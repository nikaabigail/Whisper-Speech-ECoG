from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from whisper_ecog_ext.data.vocalmind import DEFAULT_VOCALMIND_CONTRACT
from whisper_ecog_ext.integrity import atomic_write_json, fingerprint_json, sha256_file
from whisper_ecog_ext.protocol import SplitManifest, TestGate
from whisper_ecog_ext.vocalmind_oof import OofIntegrityError, aggregate_vocalmind_oof
from whisper_ecog_ext import vocalmind_oof as oof_module
from whisper_ecog_ext.vocalmind_primary import (
    PRIMARY_FOLDS,
    PRIMARY_SEEDS,
    closed_set_metrics,
    planned_training_units,
    validate_primary_config,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs" / "experiments" / "vocalmind_primary_production.json"


def _synthetic_source_identity() -> dict:
    body = {
        "schema_version": 1,
        "kind": "external_validation_source_identity",
        "root_name": "05_external_validation",
        "files": {"sentinel.py": "d" * 64},
        "files_fingerprint": fingerprint_json({"sentinel.py": "d" * 64}),
        "git": {
            "available": True,
            "commit": "e" * 40,
            "dirty": False,
            "status": [],
        },
        "python": "3.10.synthetic",
        "runtime_distributions": {},
        "required_runtime_distributions": {},
    }
    body["fingerprint"] = fingerprint_json(body)
    return body


@pytest.fixture(autouse=True)
def _aggregation_uses_the_frozen_synthetic_source(monkeypatch) -> None:
    monkeypatch.setattr(
        oof_module,
        "capture_source_identity",
        lambda: _synthetic_source_identity(),
    )


def _write_fingerprinted(path: Path, body: dict) -> dict:
    value = dict(body)
    value["fingerprint"] = fingerprint_json(value)
    atomic_write_json(path, value, overwrite=False)
    return value


def _probabilities(labels: np.ndarray, correct_until: int, global_offset: int) -> np.ndarray:
    values = np.full((20, 20), 0.1 / 19.0, dtype=np.float32)
    for row, target in enumerate(labels):
        predicted = int(target) if global_offset + row < correct_until else (int(target) + 1) % 20
        values[row, predicted] = np.float32(0.9)
    values /= values.sum(axis=1, keepdims=True, dtype=np.float32)
    return values


def _prediction_receipt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    trial_ids: tuple[str, ...],
    *,
    initialization_seed: int,
    authorization: dict,
) -> dict:
    chosen = probabilities[np.arange(len(labels)), labels]
    body = {
        "schema_version": 1,
        "kind": "deterministic_ordered_prediction_receipt",
        "task": "classification",
        "split_role": "held_out_test",
        "batch_size": 4,
        "device": "cpu",
        "evaluation_seed": initialization_seed,
        "sample_count": 20,
        "sample_ids_sha256": fingerprint_json(list(trial_ids)),
        "predictions_sha256": oof_module._array_fingerprint(probabilities),
        "targets_sha256": oof_module._array_fingerprint(labels),
        "model_state_sha256": "a" * 64,
        "training_config_fingerprint": "b" * 64,
        "metrics": {
            "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == labels)),
            "cross_entropy": float(-np.mean(np.log(np.maximum(chosen, 1e-300)))),
            "logits_sha256": "c" * 64,
        },
        "ordering": "dataset_order_no_shuffle",
        "test_gate_authorization": {
            key: authorization[key]
            for key in (
                "split_fingerprint",
                "protocol_fingerprint",
                "open_receipt_fingerprint",
            )
        },
    }
    body["fingerprint"] = fingerprint_json(body)
    return body


def _build_complete_run(
    root: Path,
    *,
    corrupt_ensemble: bool = False,
) -> Path:
    run_root = root / "production_run"
    run_root.mkdir()
    config_payload = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    config_payload["status"] = "frozen_confirmatory"
    config = validate_primary_config(config_payload)
    source_identity = _synthetic_source_identity()
    run_manifest = _write_fingerprinted(
        run_root / "run_manifest.json",
        {
            "schema_version": 1,
            "kind": "vocalmind_primary_run",
            "runner_version": "synthetic-test",
            "config": config_payload,
            "config_fingerprint": config.fingerprint,
            "dataset_index_sha256": "f" * 64,
            "host_preflight_sha256": "9" * 64,
            "dataset_counts": {"primary": 100},
            "device": "cpu",
            "source_identity": source_identity,
            "source_identity_fingerprint": source_identity["fingerprint"],
            "test_access_policy": "all units fixed",
            "neural_preprocessing": {},
            "rep6_policy": {"fit_allowed": False},
        },
    )

    class_order = tuple(DEFAULT_VOCALMIND_CONTRACT.words)
    fold_summaries: list[dict] = []
    for fold in PRIMARY_FOLDS:
        fold_root = run_root / f"fold_{fold:02d}"
        test_ids = tuple(
            f"vocalized_word:{word}:rep{fold:02d}" for word in class_order
        )
        split = SplitManifest.create(
            dataset_id="VocalMind-v2:primary-overt",
            protocol_id=f"vocalmind-primary-reps1-5-fold-{fold:02d}",
            split_seed=0,
            train_ids=tuple(
                f"vocalized_word:{word}:rep{repetition:02d}"
                for repetition in PRIMARY_FOLDS
                if repetition not in {fold, fold % 5 + 1}
                for word in class_order
            ),
            validation_ids=tuple(
                f"vocalized_word:{word}:rep{fold % 5 + 1:02d}"
                for word in class_order
            ),
            test_ids=test_ids,
            purge_gap_seconds=0.0,
            dataset_manifest_sha256="f" * 64,
        )
        split.save(fold_root / "split_manifest.json")
        required_units = tuple(
            f"seed{seed}_{unit.key}"
            for seed in config.seeds
            for unit in planned_training_units(config, outer_seed=seed)
        )
        gate = TestGate(
            state_directory=fold_root / "test_gate",
            split=split,
            required_units=required_units,
            protocol_fingerprint=run_manifest["fingerprint"],
        )
        for seed in config.seeds:
            for unit in planned_training_units(config, outer_seed=seed):
                unit_root = fold_root / f"seed_{seed}" / unit.key
                fixed = _write_fingerprinted(
                    unit_root / "validation_fixed.json",
                    {
                        "schema_version": 1,
                        "kind": "vocalmind_validation_fixed_unit",
                        "fold": fold,
                        "outer_seed": seed,
                        "initialization_seed": unit.initialization_seed,
                        "training_unit": unit.key,
                        "target_representation": unit.target_representation,
                        "config_fingerprint": config.fingerprint,
                        "split_fingerprint": split.fingerprint,
                        "target_reducer_sha256": "1" * 64,
                        "regression_checkpoint_sha256": "2" * 64,
                        "regression_best_epoch": 3,
                        "regression_best_validation_loss": 0.5,
                        "classifier_checkpoint_sha256": "3" * 64,
                        "classifier_best_epoch": 4,
                        "classifier_best_validation_loss": 0.4,
                        "validation_prediction_fingerprint": "4" * 64,
                        "frame_window_storage": {},
                        "model_selection": "validation_loss_only",
                        "threshold_selection": "not_applicable_closed_set_argmax",
                        "test_data_opened": False,
                    },
                )
                gate.mark_completed(
                    unit=f"seed{seed}_{unit.key}",
                    artifact_sha256=sha256_file(unit_root / "validation_fixed.json"),
                    run_fingerprint=fingerprint_json(
                        {
                            "config_fingerprint": config.fingerprint,
                            "split_fingerprint": split.fingerprint,
                            "outer_seed": seed,
                            "initialization_seed": unit.initialization_seed,
                            "training_unit": unit.key,
                            "target_representation": unit.target_representation,
                        }
                    ),
                )
        gate.open_test()
        authorization = asdict(gate.authorization())

        labels = np.arange(20, dtype=np.int64)
        seed_summaries: list[dict] = []
        for seed_index, seed in enumerate(config.seeds):
            planned = planned_training_units(config, outer_seed=seed)
            thresholds = {
                "mel": 45 + seed_index * 5,
                "mel_init1": 50 + seed_index * 5,
                "mel_init2": 55 + seed_index * 5,
                "L3": 60 + seed_index * 5,
                "L4": 55 + seed_index * 5,
                "L5": 50 + seed_index * 5,
            }
            raw = {
                unit.key: _probabilities(
                    labels,
                    thresholds[unit.key],
                    global_offset=(fold - 1) * 20,
                )
                for unit in planned
            }
            l345 = np.mean(
                np.stack([raw["L3"], raw["L4"], raw["L5"]]),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            mel_mean = np.mean(
                np.stack([raw["mel"], raw["mel_init1"], raw["mel_init2"]]),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            if corrupt_ensemble and fold == 4 and seed == 3:
                l345 = raw["L3"].copy()

            seed_root = fold_root / f"seed_{seed}"
            prediction_path = seed_root / "held_out_test_predictions.npz"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with prediction_path.open("xb") as handle:
                np.savez(
                    handle,
                    trial_ids=np.asarray(test_ids, dtype=np.str_),
                    labels=labels,
                    **raw,
                    L345=l345,
                    MEL_mean=mel_mean,
                )
            raw_metrics = {
                key: closed_set_metrics(values, labels, class_order=class_order)
                for key, values in raw.items()
            }
            raw_metrics["L3+L4+L5"] = closed_set_metrics(
                l345, labels, class_order=class_order
            )
            raw_metrics["MELx3"] = closed_set_metrics(
                mel_mean, labels, class_order=class_order
            )
            receipts = {
                unit.key: _prediction_receipt(
                    raw[unit.key],
                    labels,
                    test_ids,
                    initialization_seed=unit.initialization_seed,
                    authorization=authorization,
                )
                for unit in planned
            }
            result = _write_fingerprinted(
                seed_root / "result.json",
                {
                    "schema_version": 1,
                    "kind": "vocalmind_primary_seed_result",
                    "fold": fold,
                    "seed": seed,
                    "class_order": list(class_order),
                    "test_trial_ids": list(test_ids),
                    "metrics": raw_metrics,
                    "selection": {
                        "model": "validation_loss_only",
                        "threshold": "not_applicable_closed_set_argmax",
                        "test_used_for_selection": False,
                    },
                    "ensemble": {
                        "layers": [3, 4, 5],
                        "rule": "arithmetic_mean_of_probabilities",
                        "subset_search": False,
                    },
                    "mel_compute_matched_control": {
                        "mode": "required_three_initialization_probability_mean",
                        "training_units": ["mel", "mel_init1", "mel_init2"],
                        "initialization_seeds": [seed, seed + 1000, seed + 2000],
                        "rule": "arithmetic_mean_of_softmax_probabilities",
                        "subset_search": False,
                    },
                    "primary_contrast": "L3+L4+L5_vs_MELx3",
                    "secondary_contrasts": [
                        "L3+L4+L5_vs_single_MEL",
                        "L3+L4+L5_vs_L4",
                    ],
                    "prediction_receipts": receipts,
                    "prediction_npz_sha256": sha256_file(prediction_path),
                },
            )
            seed_summaries.append(
                {
                    "seed": seed,
                    "result_fingerprint": result["fingerprint"],
                    "metrics": raw_metrics,
                }
            )
        fold_summaries.append(
            {
                "fold": fold,
                "split_fingerprint": split.fingerprint,
                "test_gate_open": True,
                "test_repetition": fold,
                "validation_repetition": fold % 5 + 1,
                "seeds": seed_summaries,
            }
        )

    _write_fingerprinted(
        run_root / "summary.json",
        {
            "schema_version": 1,
            "kind": "vocalmind_primary_run_summary",
            "config_fingerprint": config.fingerprint,
            "folds": fold_summaries,
            "threshold_policy": "not_applicable_closed_set_argmax",
            "test_model_selection": False,
            "rep6_policy": {"fit_allowed": False},
        },
    )
    return run_root


def test_complete_oof_is_exact_immutable_and_descriptive(tmp_path: Path) -> None:
    run_root = _build_complete_run(tmp_path)
    output = tmp_path / "oof"
    first = aggregate_vocalmind_oof(run_root, output)
    assert first["reused_identical_output"] is False
    assert first["trials_per_seed"] == 100
    assert first["biological_n"] == 1
    payload = json.loads((output / "vocalmind_oof_summary.json").read_text("utf-8"))
    assert payload["frozen_analysis"]["models"] == ["L3", "L4", "L5", "L345", "MELx3"]
    assert len(payload["per_seed_oof"]) == 5
    assert all(item["trial_count"] == 100 for item in payload["per_seed_oof"])
    assert all(len(set(item["trial_ids"])) == 100 for item in payload["per_seed_oof"])
    assert payload["statistical_scope"]["participant_count"] == 1
    assert payload["statistical_scope"]["biological_inference"] is False
    assert payload["source_run"]["host_preflight_sha256"] == "9" * 64
    assert payload["descriptive_across_training_seeds"]["L345"]["accuracy"]["n_training_seeds"] == 5
    assert payload["frozen_analysis"]["primary_contrast"] == "L345_minus_MELx3"
    assert payload["descriptive_primary_contrast_across_training_seeds"]["accuracy"]["n_training_seeds"] == 5
    for item in payload["per_seed_oof"]:
        assert item["primary_contrast_L345_minus_MELx3"]["accuracy"] == pytest.approx(
            item["metrics"]["L345"]["accuracy"]
            - item["metrics"]["MELx3"]["accuracy"]
        )
    assert "not a population" in payload["descriptive_across_training_seeds"]["L345"]["accuracy"]["interpretation"]
    second = aggregate_vocalmind_oof(run_root, output)
    assert second["fingerprint"] == first["fingerprint"]
    assert second["reused_identical_output"] is True


def test_all_gates_are_checked_before_result_loading(tmp_path: Path, monkeypatch) -> None:
    run_root = _build_complete_run(tmp_path)
    (run_root / "fold_05" / "test_gate" / "test_gate_open.json").unlink()

    def must_not_read_predictions(*args, **kwargs):
        raise AssertionError("held-out result loader ran before every gate passed")

    monkeypatch.setattr(oof_module, "_load_all_predictions", must_not_read_predictions)
    with pytest.raises(OofIntegrityError, match="fold 5 held-out gate"):
        aggregate_vocalmind_oof(run_root, tmp_path / "oof")


def test_tampered_result_fingerprint_is_rejected(tmp_path: Path) -> None:
    run_root = _build_complete_run(tmp_path)
    path = run_root / "fold_02" / "seed_4" / "result.json"
    value = json.loads(path.read_text("utf-8"))
    value["selection"]["test_used_for_selection"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(OofIntegrityError, match="fingerprint mismatch"):
        aggregate_vocalmind_oof(run_root, tmp_path / "oof")


def test_nonfixed_stored_l345_is_rejected(tmp_path: Path) -> None:
    run_root = _build_complete_run(tmp_path, corrupt_ensemble=True)
    with pytest.raises(OofIntegrityError, match="stored L345"):
        aggregate_vocalmind_oof(run_root, tmp_path / "oof")


def test_existing_output_cannot_be_replaced(tmp_path: Path) -> None:
    run_root = _build_complete_run(tmp_path)
    output = tmp_path / "oof"
    aggregate_vocalmind_oof(run_root, output)
    csv_path = output / "vocalmind_oof_metrics.csv"
    csv_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(OofIntegrityError, match="existing immutable OOF output differs"):
        aggregate_vocalmind_oof(run_root, output)


def test_aggregation_source_must_match_training_run(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = _build_complete_run(tmp_path)
    changed = _synthetic_source_identity()
    changed["git"] = {**changed["git"], "commit": "a" * 40}
    changed["fingerprint"] = fingerprint_json(
        {key: value for key, value in changed.items() if key != "fingerprint"}
    )
    monkeypatch.setattr(oof_module, "capture_source_identity", lambda: changed)
    with pytest.raises(OofIntegrityError, match="differs from the production run"):
        aggregate_vocalmind_oof(run_root, tmp_path / "oof")
