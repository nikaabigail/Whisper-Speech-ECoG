#!/usr/bin/env python3
"""Run the resumable full-neural SWPD sub-01 regression pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
from pathlib import Path
import os
import sys
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from whisper_ecog_ext.evaluation import evaluate_regression  # noqa: E402
from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
    fingerprint_json,
    read_json,
    sha256_file,
)
from whisper_ecog_ext.model import OneSecondEcogEncoder  # noqa: E402
from whisper_ecog_ext.protocol import SplitManifest, TestGate  # noqa: E402
from whisper_ecog_ext.source_identity import capture_source_identity  # noqa: E402
from whisper_ecog_ext.swpd.audio_vad import closed_event_gate_payload  # noqa: E402
from whisper_ecog_ext.swpd.matched_linear import (  # noqa: E402
    TARGET_NAMES,
    make_visual_blocks,
    regression_metrics,
)
from whisper_ecog_ext.swpd.neural_pilot import (  # noqa: E402
    FAST_SMOKE_FRAME_HZ,
    PRODUCTION_FRAME_HZ,
    SWPDNeuralPreprocessor,
    build_neural_extraction_fingerprint,
    extract_neural_target_block,
    fit_or_load_channel_standardizer,
    fit_or_load_target_reducer,
    load_neural_block_cache,
    load_usable_channels,
    make_sub01_neural_split,
    make_window_dataset,
    save_neural_block_cache,
    standardize_blocks_once,
)
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    ConfirmatoryDataLocked,
    NWBLayoutError,
    PILOT_SUBJECT,
    SWPDRecording,
    inventory_pilot,
    load_visual_word_events,
    recording_duration_seconds,
    subject_paths,
)
from whisper_ecog_ext.targets import (  # noqa: E402
    MelTargetExtractor,
    WhisperLayerTargetExtractor,
)
from whisper_ecog_ext.training import TrainingConfig, train_regression  # noqa: E402


CONFIG_PATH = HERE / "configs" / "experiments" / "swpd_sub01_neural_pilot.json"
DATASET_MANIFEST = HERE / "manifests" / "swpd_osf_nrgx6.json"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _external_directory(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if result == HERE or _inside(result, HERE):
        raise ValueError(f"{label} must be outside the source checkout")
    return result


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_prediction_result(path: Path, evaluation) -> str:
    _atomic_npz(
        path,
        sample_ids=np.asarray(evaluation.sample_ids, dtype="U96"),
        targets=evaluation.targets,
        predictions=evaluation.predictions,
    )
    return sha256_file(path)


def _model(input_channels: int) -> OneSecondEcogEncoder:
    return OneSecondEcogEncoder(
        input_channels=input_channels,
        target_dim=50,
        window_samples=1001,
        hidden_channels=30,
        temporal_stride=10,
        filtering_kernel=25,
        envelope_kernel=15,
        use_lstm=True,
    )


def _completion_matches(path: Path, *, artifact: str, run: str) -> None:
    value = read_json(path)
    stored = value.pop("fingerprint", None)
    if stored != fingerprint_json(value):
        raise RuntimeError(f"training completion receipt was modified: {path}")
    if value.get("artifact_sha256") != artifact or value.get("run_fingerprint") != run:
        raise RuntimeError(f"training completion receipt disagrees with checkpoint: {path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--subject", choices=[PILOT_SUBJECT], default=PILOT_SUBJECT)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument(
        "--single-mel-development",
        action="store_true",
        help="Train one MEL initialization; not the branch-count-matched control",
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="50 Hz/two-epoch diagnostic only; it is explicitly not a result",
    )
    parser.add_argument(
        "--audio-candidate-tsv",
        type=Path,
        help="Optional unreviewed audio-only candidate TSV to bind into the closed event gate",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    config_document = read_json(CONFIG_PATH)
    if config_document.get("status") != "development_sub01_only":
        raise RuntimeError("unexpected SWPD neural protocol status")
    cache_dir = _external_directory(args.cache_dir, "cache-dir")
    run_dir = _external_directory(args.run_dir, "run-dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    fast_smoke = bool(args.fast_smoke)
    single_mel = bool(args.single_mel_development or fast_smoke)
    frame_hz = FAST_SMOKE_FRAME_HZ if fast_smoke else PRODUCTION_FRAME_HZ
    result_status = (
        "diagnostic_only_50hz_not_a_scientific_result"
        if fast_smoke
        else (
            "development_single_mel_not_compute_matched"
            if single_mel
            else "development_compute_matched_pilot"
        )
    )
    print(f"[runtime] device={device} | mode={result_status} | subject={PILOT_SUBJECT}")

    inventory = inventory_pilot(args.data_root)
    paths = subject_paths(args.data_root, PILOT_SUBJECT)
    usable = load_usable_channels(paths["channels"], inventory.ieeg.shape[1])
    if len(usable.indices) != 127:
        raise RuntimeError(
            f"sub-01 protocol expected 127 usable channels, found {len(usable.indices)}"
        )
    preprocessor = SWPDNeuralPreprocessor(
        input_rate_hz=inventory.ieeg.rate_hz,
        usable_channels=usable,
    )
    preprocessing_provenance = preprocessor.provenance()
    print(
        "[neural] 127 usable channels | CAR=false | notch 50/100/150 | "
        "band-pass 10-200 | polyphase 1024->1000"
    )

    events = load_visual_word_events(args.data_root)
    definitions = make_visual_blocks(events, recording_duration_seconds(inventory))
    dataset_manifest_sha = sha256_file(DATASET_MANIFEST)
    source_identity = capture_source_identity(HERE)
    whisper_revision = config_document["targets"]["whisper"]["revision"]
    mel_extractor = MelTargetExtractor(n_mels=80, frame_hz=float(frame_hz))
    mel_provenance = mel_extractor.provenance()
    whisper_provenance = {
        "kind": "whisper_encoder_hidden_states",
        "model_name": "openai/whisper-base",
        "revision": whisper_revision,
        "layers": [3, 4, 5],
        "sample_rate": 16_000,
        "chunk_seconds": 30,
        "peak_normalize": True,
        "alignment": "neighboring-frame linear interpolation",
        "single_forward_per_chunk_for_all_layers": True,
    }
    extraction_fingerprints = [
        build_neural_extraction_fingerprint(
            inventory=inventory,
            events_path=paths["events"],
            dataset_manifest_sha256=dataset_manifest_sha,
            definition=definition,
            preprocessing_provenance=preprocessing_provenance,
            mel_provenance=mel_provenance,
            whisper_provenance=whisper_provenance,
            frame_hz=frame_hz,
            source_fingerprint=source_identity["fingerprint"],
        )
        for definition in definitions
    ]

    blocks = []
    missing: list[int] = []
    for definition, extraction_fingerprint in zip(
        definitions, extraction_fingerprints
    ):
        try:
            cached = load_neural_block_cache(
                cache_dir,
                definition.index,
                extraction_fingerprint=extraction_fingerprint,
            )
        except RuntimeError:
            if not args.force_cache:
                raise
            for path in cache_dir.glob(f"neural_block_{definition.index:02d}*"):
                if path.is_file():
                    path.unlink()
            cached = None
        blocks.append(cached)
        if cached is None:
            missing.append(definition.index)
        else:
            print(f"[cache] reuse visual block {definition.index}")

    if missing:
        print(
            f"[cache] build blocks {missing} at {frame_hz} Hz; "
            "Whisper L3/L4/L5 share each encoder forward"
        )
        whisper_extractor = WhisperLayerTargetExtractor(
            revision=whisper_revision,
            device=device,
        )
        actual_whisper = whisper_extractor.provenance()
        for key in (
            "kind",
            "model_name",
            "revision",
            "layers",
            "sample_rate",
            "chunk_seconds",
            "peak_normalize",
            "alignment",
        ):
            if actual_whisper[key] != whisper_provenance[key]:
                raise RuntimeError(f"Whisper provenance changed for {key}")
        with SWPDRecording(args.data_root, PILOT_SUBJECT) as recording:
            for index in missing:
                block = extract_neural_target_block(
                    recording,
                    inventory,
                    definitions[index],
                    preprocessor=preprocessor,
                    mel_extractor=mel_extractor,
                    whisper_extractor=whisper_extractor,
                    extraction_fingerprint=extraction_fingerprints[index],
                    frame_hz=frame_hz,
                )
                save_neural_block_cache(block, cache_dir)
                blocks[index] = block
                print(
                    f"[cache] saved block {index}: {len(block.sample_ids)} frames, "
                    f"{block.raw_1000hz.shape[1]} channels"
                )
    if any(block is None for block in blocks):
        raise RuntimeError("not all five SWPD neural caches are available")
    complete_blocks = tuple(blocks)

    split_plan = make_sub01_neural_split(
        complete_blocks, dataset_manifest_sha256=dataset_manifest_sha
    )
    split_path = run_dir / "split_manifest.json"
    if split_path.exists():
        if SplitManifest.load(split_path) != split_plan.manifest:
            raise RuntimeError("existing run has a different immutable split")
    else:
        split_plan.manifest.save(split_path)
    print(
        f"[split] train blocks={split_plan.train_blocks} | "
        f"validation={split_plan.validation_block} | held-out test={split_plan.test_block}"
    )

    run_protocol = {
        "schema_version": 1,
        "kind": "swpd_sub01_full_neural_regression_protocol",
        "source_config_sha256": sha256_file(CONFIG_PATH),
        "source_identity": source_identity,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "split_fingerprint": split_plan.manifest.fingerprint,
        "extraction_fingerprints": extraction_fingerprints,
        "frame_hz": frame_hz,
        "downstream_hidden_stride": 10,
        "downstream_hidden_trajectory_hz": 100 if not fast_smoke else 5,
        "result_status": result_status,
        "single_mel_development": single_mel,
        "preprocessing": preprocessing_provenance,
        "visual_events_used_only_for": "five block boundaries",
        "visual_events_are_acoustic_onsets": False,
        "asynchronous_event_gate_open": False,
    }
    run_protocol["fingerprint"] = fingerprint_json(run_protocol)
    atomic_write_json(run_dir / "run_protocol.json", run_protocol, overwrite=True)
    if args.prepare_cache_only:
        print(f"[done] five reusable neural/target blocks prepared at {cache_dir}")
        print("[gate] event/asynchronous evaluation remains CLOSED")
        return 0

    scaler = fit_or_load_channel_standardizer(
        complete_blocks,
        split_plan.train_blocks,
        run_dir / "transforms" / "neural_channel_standardizer",
    )
    train_validation_indexes = split_plan.train_blocks + (split_plan.validation_block,)
    standardized_train_validation = standardize_blocks_once(
        complete_blocks, train_validation_indexes, scaler
    )
    print("[scale] train-only channel statistics applied once per whole block")

    reducer_seed = int(config_document["training"]["base_seed"])
    reducers = {
        target: fit_or_load_target_reducer(
            complete_blocks,
            split_plan.train_blocks,
            target,
            run_dir / "transforms" / f"{target}_target_pca50",
            seed=reducer_seed,
        )
        for target in TARGET_NAMES
    }
    mel_seeds = tuple(
        int(value)
        for value in config_document["compute_matched_controls"][
            "production_mel_initialization_seeds"
        ]
    )
    if single_mel:
        mel_seeds = mel_seeds[:1]
    whisper_seed = int(
        config_document["compute_matched_controls"]["whisper_initialization_seed"]
    )
    units = tuple((f"mel80_seed_{seed}", "mel80", seed) for seed in mel_seeds) + tuple(
        (f"{target}_seed_{whisper_seed}", target, whisper_seed)
        for target in ("L3", "L4", "L5")
    )
    gate = TestGate(
        state_directory=run_dir / "regression_test_gate",
        split=split_plan.manifest,
        required_units=tuple(unit[0] for unit in units),
        protocol_fingerprint=run_protocol["fingerprint"],
    )

    base_training = config_document["training"]
    maximum_epochs = int(args.max_epochs or base_training["max_epochs"])
    if fast_smoke:
        maximum_epochs = min(maximum_epochs, 2)
    batch_size = int(args.batch_size or base_training["batch_size"])
    unit_results = {}
    for unit_name, target_name, unit_seed in units:
        target_reducer = reducers[target_name]
        train_dataset = make_window_dataset(
            complete_blocks,
            split_plan.train_blocks,
            target_name=target_name,
            target_reducer=target_reducer,
            standardized_trials=standardized_train_validation,
            split_role="train",
        )
        validation_dataset = make_window_dataset(
            complete_blocks,
            (split_plan.validation_block,),
            target_name=target_name,
            target_reducer=target_reducer,
            standardized_trials=standardized_train_validation,
            split_role="validation",
        )
        training_config = TrainingConfig(
            seed=unit_seed,
            max_epochs=maximum_epochs,
            batch_size=batch_size,
            learning_rate=float(base_training["learning_rate"]),
            weight_decay=float(base_training["weight_decay"]),
            patience=int(base_training["patience"]),
            min_delta=float(base_training["min_delta"]),
            grad_clip_norm=float(base_training["grad_clip_norm"]),
            num_workers=int(base_training["num_workers"]),
            device=device,
            strict_determinism=bool(base_training["strict_determinism"]),
        )
        model = _model(len(usable.indices))
        checkpoint = run_dir / "checkpoints" / f"{unit_name}.pt"
        while True:
            result = train_regression(
                model,
                train_dataset,
                validation_dataset,
                config=training_config,
                checkpoint_path=checkpoint,
                resume=checkpoint.exists(),
                run_context={
                    "protocol_fingerprint": run_protocol["fingerprint"],
                    "split_fingerprint": split_plan.manifest.fingerprint,
                    "unit": unit_name,
                    "target": target_name,
                    "target_reducer_train_ids_sha256": target_reducer.train_sample_ids_sha256,
                },
                max_epochs_this_call=1,
            )
            latest = result.history[-1]
            print(
                f"[{unit_name} epoch {latest['epoch']:03d}] "
                f"trainMSE={latest['train_loss']:.6f} "
                f"valMSE={latest['validation_loss']:.6f} "
                f"best={result.best_validation_loss:.6f}@{result.best_epoch}"
            )
            if result.completed:
                break
        validation = evaluate_regression(
            model,
            validation_dataset,
            batch_size=batch_size,
            device=device,
            training_config_fingerprint=result.config_fingerprint,
            evaluation_seed=unit_seed,
        )
        unit_dir = run_dir / "units" / unit_name
        validation_path = unit_dir / "validation_predictions.npz"
        validation_sha = _save_prediction_result(validation_path, validation)
        atomic_write_json(
            unit_dir / "validation_receipt.json", validation.receipt, overwrite=True
        )
        completion_path = gate.completions_directory / f"{unit_name}.json"
        if completion_path.exists():
            _completion_matches(
                completion_path,
                artifact=result.checkpoint_sha256,
                run=result.config_fingerprint,
            )
        else:
            gate.mark_completed(
                unit=unit_name,
                artifact_sha256=result.checkpoint_sha256,
                run_fingerprint=result.config_fingerprint,
            )
        training_receipt = result.receipt()
        training_receipt["checkpoint_path"] = str(training_receipt["checkpoint_path"])
        unit_results[unit_name] = {
            "target": target_name,
            "seed": unit_seed,
            "training": training_receipt,
            "validation": validation.receipt,
            "validation_predictions_sha256": validation_sha,
        }
        del model, train_dataset, validation_dataset, validation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # This gate is regression-only and independent of the still-closed event gate.
    expected_test_ids = gate.open_test()
    if expected_test_ids != split_plan.manifest.held_out_test_ids:
        raise RuntimeError("regression test gate returned unexpected sample IDs")
    authorization = gate.authorization()
    standardized_test = standardize_blocks_once(
        complete_blocks, (split_plan.test_block,), scaler
    )
    test_results = {}
    mel_evaluations = []
    for unit_name, target_name, unit_seed in units:
        target_reducer = reducers[target_name]
        train_dataset = make_window_dataset(
            complete_blocks,
            split_plan.train_blocks,
            target_name=target_name,
            target_reducer=target_reducer,
            standardized_trials=standardized_train_validation,
            split_role="train",
        )
        validation_dataset = make_window_dataset(
            complete_blocks,
            (split_plan.validation_block,),
            target_name=target_name,
            target_reducer=target_reducer,
            standardized_trials=standardized_train_validation,
            split_role="validation",
        )
        test_dataset = make_window_dataset(
            complete_blocks,
            (split_plan.test_block,),
            target_name=target_name,
            target_reducer=target_reducer,
            standardized_trials=standardized_test,
            split_role="held_out_test",
            split_fingerprint=split_plan.manifest.fingerprint,
        )
        training_config = replace(
            TrainingConfig(
                seed=unit_seed,
                max_epochs=maximum_epochs,
                batch_size=batch_size,
                learning_rate=float(base_training["learning_rate"]),
                weight_decay=float(base_training["weight_decay"]),
                patience=int(base_training["patience"]),
                min_delta=float(base_training["min_delta"]),
                grad_clip_norm=float(base_training["grad_clip_norm"]),
                num_workers=int(base_training["num_workers"]),
                device=device,
                strict_determinism=bool(base_training["strict_determinism"]),
            )
        )
        model = _model(len(usable.indices))
        result = train_regression(
            model,
            train_dataset,
            validation_dataset,
            config=training_config,
            checkpoint_path=run_dir / "checkpoints" / f"{unit_name}.pt",
            resume=True,
            run_context={
                "protocol_fingerprint": run_protocol["fingerprint"],
                "split_fingerprint": split_plan.manifest.fingerprint,
                "unit": unit_name,
                "target": target_name,
                "target_reducer_train_ids_sha256": target_reducer.train_sample_ids_sha256,
            },
        )
        evaluation = evaluate_regression(
            model,
            test_dataset,
            batch_size=batch_size,
            device=device,
            training_config_fingerprint=result.config_fingerprint,
            evaluation_seed=unit_seed,
            test_gate_authorization=authorization,
        )
        prediction_path = run_dir / "units" / unit_name / "test_predictions.npz"
        prediction_sha = _save_prediction_result(prediction_path, evaluation)
        metrics = regression_metrics(evaluation.targets, evaluation.predictions)
        test_results[unit_name] = {
            "target": target_name,
            "seed": unit_seed,
            "metrics": metrics,
            "evaluation_receipt": evaluation.receipt,
            "predictions_sha256": prediction_sha,
        }
        if target_name == "mel80":
            mel_evaluations.append(evaluation)
        correlation = metrics["fisher_z_component_correlation"]
        correlation_text = "n/a" if correlation is None else f"{correlation:.4f}"
        print(
            f"[test {unit_name}] standardized MSE={metrics['standardized_mse']:.6f} | "
            f"r={correlation_text}"
        )
        del model, train_dataset, validation_dataset, test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mel_control = None
    if len(mel_evaluations) == 3:
        reference = mel_evaluations[0]
        for evaluation in mel_evaluations[1:]:
            if evaluation.sample_ids != reference.sample_ids:
                raise RuntimeError("MEL control members have different sample ordering")
            np.testing.assert_array_equal(evaluation.targets, reference.targets)
        prediction = np.mean(
            np.stack([evaluation.predictions for evaluation in mel_evaluations], axis=0),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        mel_control_path = run_dir / "controls" / "mel80_fixed_3init_test_predictions.npz"
        _atomic_npz(
            mel_control_path,
            sample_ids=np.asarray(reference.sample_ids, dtype="U96"),
            targets=reference.targets,
            predictions=prediction,
        )
        mel_control = {
            "kind": "fixed_compute_matched_mel_three_initialization_regression_control",
            "member_seeds": list(mel_seeds),
            "selection": "none_predeclared_arithmetic_mean",
            "metrics": regression_metrics(reference.targets, prediction),
            "predictions_sha256": sha256_file(mel_control_path),
        }

    event_gate = closed_event_gate_payload(
        candidate_tsv=args.audio_candidate_tsv,
        regression_units=unit_results,
    )
    atomic_write_json(run_dir / "asynchronous_event_gate.json", event_gate, overwrite=True)
    summary = {
        "schema_version": 1,
        "kind": "swpd_sub01_full_neural_regression_pilot_result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_status": result_status,
        "subject": PILOT_SUBJECT,
        "confirmatory_subjects_read": False,
        "frame_hz": frame_hz,
        "split": {
            "train_blocks": list(split_plan.train_blocks),
            "validation_block": split_plan.validation_block,
            "test_block": split_plan.test_block,
            "fingerprint": split_plan.manifest.fingerprint,
        },
        "architecture": _model(len(usable.indices)).architecture_receipt(),
        "preprocessing": preprocessing_provenance,
        "train_only_channel_standardizer": scaler.manifest_payload(),
        "units": unit_results,
        "test": test_results,
        "compute_matched_mel_control": mel_control,
        "fixed_l345_event_ensemble": {
            "layers": [3, 4, 5],
            "rule": "arithmetic mean of probabilities",
            "evaluated": False,
            "reason": "continuous event heads and human-audited audio labels are not available",
        },
        "asynchronous_event_gate": event_gate,
    }
    atomic_write_json(run_dir / "regression_summary.json", summary, overwrite=True)
    print(f"[done] regression pilot: {run_dir / 'regression_summary.json'}")
    print("[gate] asynchronous/event final test remains CLOSED pending human audio audit")
    print("[safety] only sub-01 was accessible; sub-02..sub-10 remain code-locked")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ConfirmatoryDataLocked,
        FileExistsError,
        NWBLayoutError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
