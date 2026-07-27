from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from whisper_ecog_ext.evaluation import evaluate_regression  # noqa: E402
from whisper_ecog_ext.model import OneSecondEcogEncoder  # noqa: E402
from whisper_ecog_ext.swpd.audio_vad import (  # noqa: E402
    AudioAuditRequired,
    EnergyVadConfig,
    closed_event_gate_payload,
    detect_audio_energy_candidates,
    load_audited_audio_intervals,
    validate_audio_candidate_bundle,
    write_audio_candidates,
)
from whisper_ecog_ext.swpd.matched_linear import TARGET_NAMES, VisualBlock  # noqa: E402
from whisper_ecog_ext.swpd.neural_pilot import (  # noqa: E402
    NeuralTargetBlock,
    PRODUCTION_FRAME_HZ,
    SWPDNeuralPreprocessor,
    UsableChannels,
    fit_or_load_channel_standardizer,
    fit_or_load_target_reducer,
    load_neural_block_cache,
    make_sub01_neural_split,
    make_window_dataset,
    save_neural_block_cache,
    standardize_blocks_once,
)
from whisper_ecog_ext.training import TrainingConfig, train_regression  # noqa: E402


def _synthetic_blocks(seed: int = 9) -> tuple[NeuralTargetBlock, ...]:
    rng = np.random.default_rng(seed)
    blocks = []
    for block_index in range(5):
        frame_count = 55
        trial_start = block_index * 3.0
        raw = rng.normal(size=(2_600, 2)).astype(np.float32)
        end_indices = 1000 + np.arange(frame_count) * 10
        times = trial_start + end_indices / 1000.0
        targets = {
            "mel80": rng.normal(size=(frame_count, 80)).astype(np.float32),
            "L3": rng.normal(size=(frame_count, 512)).astype(np.float32),
            "L4": rng.normal(size=(frame_count, 512)).astype(np.float32),
            "L5": rng.normal(size=(frame_count, 512)).astype(np.float32),
        }
        blocks.append(
            NeuralTargetBlock(
                definition=VisualBlock(
                    block_index,
                    (f"trial-{block_index}",),
                    block_index,
                    block_index,
                    trial_start,
                    trial_start + 2.6,
                ),
                trial_id=f"block-{block_index}",
                trial_start_seconds=trial_start,
                raw_1000hz=raw,
                sample_ids=np.asarray(
                    [f"block-{block_index}:frame-{index}" for index in range(frame_count)]
                ),
                frame_times_seconds=times,
                targets=targets,
                extraction_fingerprint="a" * 64,
            )
        )
    return tuple(blocks)


class SWPDNeuralPreprocessingTests(unittest.TestCase):
    def test_production_regression_grid_is_exactly_1000_hz(self) -> None:
        self.assertEqual(PRODUCTION_FRAME_HZ, 1000)

    def test_fixed_preprocessing_is_representation_independent_and_has_no_car(self) -> None:
        rate = 1024
        time = np.arange(3 * rate) / rate
        raw = np.stack(
            [
                np.sin(2 * np.pi * 5 * time),
                np.sin(2 * np.pi * 80 * time),
                np.sin(2 * np.pi * 250 * time),
            ],
            axis=1,
        )
        channels = UsableChannels(
            indices=(0, 1, 2),
            names=("A", "B", "C"),
            types=("SEEG",) * 3,
            statuses=("n/a",) * 3,
        )
        preprocessor = SWPDNeuralPreprocessor(
            input_rate_hz=rate, usable_channels=channels
        )
        first = preprocessor.transform(raw)
        second = preprocessor.transform(raw)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3000, 3))
        provenance = preprocessor.provenance()
        self.assertFalse(provenance["common_average_reference"])
        self.assertFalse(provenance["whole_recording_z_score"])
        self.assertTrue(provenance["representation_independent"])
        self.assertEqual(provenance["resample_up"], 125)
        self.assertEqual(provenance["resample_down"], 128)

    def test_checksummed_block_cache_round_trip(self) -> None:
        block = _synthetic_blocks()[0]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            save_neural_block_cache(block, cache)
            loaded = load_neural_block_cache(
                cache, 0, extraction_fingerprint=block.extraction_fingerprint
            )
            self.assertIsNotNone(loaded)
            self.assertIsInstance(loaded.raw_1000hz, np.memmap)
            np.testing.assert_array_equal(loaded.sample_ids, block.sample_ids)
            for target in TARGET_NAMES:
                np.testing.assert_array_equal(loaded.targets[target], block.targets[target])
            with self.assertRaises(RuntimeError):
                load_neural_block_cache(cache, 0, extraction_fingerprint="b" * 64)
            loaded.close()


class SWPDNeuralTrainingIntegrationTests(unittest.TestCase):
    def test_common_encoder_train_only_transforms_and_causal_windows(self) -> None:
        blocks = _synthetic_blocks()
        split = make_sub01_neural_split(
            blocks, dataset_manifest_sha256="d" * 64
        )
        self.assertEqual(split.train_blocks, (1, 2, 3))
        self.assertEqual(split.validation_block, 0)
        self.assertEqual(split.test_block, 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            standardizer = fit_or_load_channel_standardizer(
                blocks, split.train_blocks, root / "neural_scaler"
            )
            reducer = fit_or_load_target_reducer(
                blocks,
                split.train_blocks,
                "mel80",
                root / "mel_reducer",
                seed=4,
            )
            standardized = standardize_blocks_once(
                blocks, split.train_blocks + (split.validation_block,), standardizer
            )
            train = make_window_dataset(
                blocks,
                split.train_blocks,
                target_name="mel80",
                target_reducer=reducer,
                standardized_trials=standardized,
                split_role="train",
            )
            l3_reducer = fit_or_load_target_reducer(
                blocks,
                split.train_blocks,
                "L3",
                root / "l3_reducer",
                seed=4,
            )
            l3_train = make_window_dataset(
                blocks,
                split.train_blocks,
                target_name="L3",
                target_reducer=l3_reducer,
                standardized_trials=standardized,
                split_role="train",
            )
            validation = make_window_dataset(
                blocks,
                (split.validation_block,),
                target_name="mel80",
                target_reducer=reducer,
                standardized_trials=standardized,
                split_role="validation",
            )
            self.assertEqual(train.records[0].end_index, 1000)
            self.assertEqual(train.records[0].start_index, 0)
            self.assertEqual(train[0]["inputs"].shape, (2, 1001))
            self.assertEqual(train.sample_ids, l3_train.sample_ids)
            np.testing.assert_array_equal(
                train[17]["inputs"].numpy(), l3_train[17]["inputs"].numpy()
            )
            model = OneSecondEcogEncoder(
                input_channels=2,
                target_dim=50,
                hidden_channels=2,
                temporal_stride=100,
                filtering_kernel=3,
                envelope_kernel=3,
                use_lstm=False,
            )
            result = train_regression(
                model,
                train,
                validation,
                config=TrainingConfig(
                    seed=4,
                    max_epochs=1,
                    batch_size=32,
                    patience=2,
                    device="cpu",
                ),
                checkpoint_path=root / "pilot.pt",
                run_context={"split_fingerprint": split.manifest.fingerprint},
            )
            self.assertTrue(result.completed)
            evaluated = evaluate_regression(
                model,
                validation,
                batch_size=32,
                training_config_fingerprint=result.config_fingerprint,
            )
            self.assertEqual(evaluated.predictions.shape, (55, 50))
            self.assertTrue(np.isfinite(evaluated.predictions).all())


class SWPDAudioAuditGateTests(unittest.TestCase):
    def test_audio_only_candidates_are_deterministic_but_cannot_open_event_gate(self) -> None:
        rate = 48_000
        time = np.arange(2 * rate) / rate
        audio = np.full(time.shape, 1e-4, dtype=np.float32)
        for onset, offset in ((0.40, 0.75), (1.20, 1.55)):
            selected = (time >= onset) & (time < offset)
            audio[selected] += (0.5 * np.sin(2 * np.pi * 220 * time[selected])).astype(
                np.float32
            )
        config = EnergyVadConfig()
        first, provenance = detect_audio_energy_candidates(audio, config=config)
        second, second_provenance = detect_audio_energy_candidates(audio, config=config)
        self.assertEqual(first, second)
        self.assertEqual(provenance["fingerprint"], second_provenance["fingerprint"])
        self.assertEqual(len(first), 2)
        self.assertFalse(provenance["visual_events_used"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidates.tsv"
            metadata = root / "candidates.json"
            write_audio_candidates(
                first,
                provenance,
                tsv_path=candidate,
                metadata_path=metadata,
                measured_nwb_rate_hz=47_999.187483,
            )
            validated = validate_audio_candidate_bundle(candidate, metadata)
            self.assertFalse(validated["event_evaluation_authorized"])
            with self.assertRaises(AudioAuditRequired):
                load_audited_audio_intervals(
                    root / "audited.tsv",
                    root / "audit_receipt.json",
                    candidate_tsv=candidate,
                )
            gate = closed_event_gate_payload(
                candidate_tsv=candidate,
                regression_units=("L3", "L4", "L5"),
            )
            self.assertFalse(gate["open"])
            self.assertFalse(gate["visual_events_are_ground_truth"])


if __name__ == "__main__":
    unittest.main()
