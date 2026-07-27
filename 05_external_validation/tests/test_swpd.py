from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import h5py
import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT / "src"))

import swpd_download  # noqa: E402
from whisper_ecog_ext.swpd.author_mel import (  # noqa: E402
    extract_author_log_mel,
    extract_features_from_pilot,
    extract_high_gamma,
    run_nonshuffled_cv,
    stack_author_context,
)
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    ConfirmatoryDataLocked,
    SWPDRecording,
    VisualWordEvent,
    assert_series_start_alignment,
    inventory_pilot,
    recording_duration_seconds,
    recording_relative_sample_bounds,
    recording_relative_to_series_time,
    subject_paths,
)
from whisper_ecog_ext.swpd.matched_linear import (  # noqa: E402
    MatchedBlock,
    VisualBlock,
    extract_one_block,
    load_block_cache,
    make_visual_blocks,
    regression_metrics,
    run_matched_folds,
    save_block_cache,
)


MANIFEST = MODULE_ROOT / "manifests" / "swpd_osf_nrgx6.json"


def _fake_manifest(archive_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": {"pilot_subject": "sub-01"},
        "archive": {
            "name": archive_path.name,
            "url": "https://osf.io/download/test/",
            "size_bytes": archive_path.stat().st_size,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "archive_root": "SingleWordProductionDutch-iBIDS",
        },
    }


def _add_series(
    acquisition: h5py.Group,
    name: str,
    data: np.ndarray,
    rate: float,
    unit: str,
    starting_time_seconds: float = 0.0,
) -> None:
    group = acquisition.create_group(name)
    dataset = group.create_dataset("data", data=data)
    dataset.attrs["unit"] = unit
    starting_time = group.create_dataset("starting_time", data=starting_time_seconds)
    starting_time.attrs["rate"] = rate


def _make_synthetic_swpd(
    parent: Path,
    seconds: float = 4.0,
    audio_metadata_rate: float = 48000.0,
    acquisition_start_seconds: float = 0.0,
) -> Path:
    root = parent / "SingleWordProductionDutch-iBIDS"
    root.mkdir()
    (root / "participants.tsv").write_text(
        "participant_id\nsub-01\n", encoding="utf-8"
    )
    ieeg_dir = root / "sub-01" / "ieeg"
    ieeg_dir.mkdir(parents=True)
    prefix = "sub-01_task-wordProduction"
    nwb_path = ieeg_dir / f"{prefix}_ieeg.nwb"
    ieeg_rate = 1024
    audio_rate = 48000
    ieeg_count = int(seconds * ieeg_rate)
    audio_count = int(seconds * audio_rate)
    time_ieeg = np.arange(ieeg_count) / ieeg_rate
    time_audio = np.arange(audio_count) / audio_rate
    neural = np.stack(
        [
            np.sin(2 * np.pi * 80 * time_ieeg),
            np.sin(2 * np.pi * 120 * time_ieeg + 0.2),
            np.sin(2 * np.pi * 160 * time_ieeg + 0.4),
        ],
        axis=1,
    ).astype(np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 220 * time_audio)).astype(np.float32)
    stimulus = np.zeros(ieeg_count, dtype=np.int16)
    with h5py.File(nwb_path, "w") as handle:
        handle.attrs["nwb_version"] = "2.6.0"
        acquisition = handle.create_group("acquisition")
        _add_series(
            acquisition, "iEEG", neural, ieeg_rate, "volts", acquisition_start_seconds
        )
        _add_series(
            acquisition,
            "Audio",
            audio,
            audio_metadata_rate,
            "a.u.",
            acquisition_start_seconds + 3e-6,
        )
        _add_series(
            acquisition,
            "Stimulus",
            stimulus,
            ieeg_rate,
            "label",
            acquisition_start_seconds,
        )
    (ieeg_dir / f"{prefix}_channels.tsv").write_text(
        "name\ttype\nA1\tSEEG\nA2\tSEEG\nA3\tSEEG\n", encoding="utf-8"
    )
    (ieeg_dir / f"{prefix}_events.tsv").write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0.0\t2.0\tword\teerste\t0\n"
        "2.0\t1.0\tfixation\t+\t2048\n"
        "3.0\t1.0\tword\ttweede\t3072\n",
        encoding="utf-8",
    )
    return root


class SWPDManifestTests(unittest.TestCase):
    def test_pinned_osf_manifest(self) -> None:
        manifest = swpd_download.load_manifest(MANIFEST)
        archive = manifest["archive"]
        self.assertEqual(archive["size_bytes"], 2_794_936_886)
        self.assertEqual(
            archive["sha256"],
            "015bc9c565c3dbdc7259c01be54f62b3346cbd7dc5cec8156eb718f64b6cbcd9",
        )
        self.assertNotEqual(
            archive["size_bytes"], archive["observed_head_content_length_bytes"]
        )
        self.assertIn("HEAD", archive["head_size_warning"])

    def test_source_checkout_is_not_a_valid_data_destination(self) -> None:
        with self.assertRaises(ValueError):
            swpd_download.require_external_destination(MODULE_ROOT / "raw")

    def test_observed_osf_redirect_hosts_are_allowlisted(self) -> None:
        swpd_download._validate_redirect("https://osf.io/download/g6q5m/")
        swpd_download._validate_redirect(
            "https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/id"
        )
        swpd_download._validate_redirect(
            "https://storage.googleapis.com/cos-osf-prod-files-de-1/object"
        )
        with self.assertRaises(RuntimeError):
            swpd_download._validate_redirect("https://example.com/object")


class SWPDSafeExtractionTests(unittest.TestCase):
    def test_archive_top_level_is_not_double_nested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "SingleWordProductionDutch-iBIDS.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(
                    "SingleWordProductionDutch-iBIDS/participants.tsv",
                    "participant_id\nsub-01\n",
                )
                handle.writestr(
                    "SingleWordProductionDutch-iBIDS/sub-01/ieeg/sample.nwb", b"fixture"
                )
            target = swpd_download.safe_extract_archive(
                archive, parent / "destination", _fake_manifest(archive)
            )
            self.assertTrue((target / "participants.tsv").is_file())
            self.assertFalse(
                (target / "SingleWordProductionDutch-iBIDS" / "participants.tsv").exists()
            )
            self.assertEqual(target.name, "SingleWordProductionDutch-iBIDS")

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "SingleWordProductionDutch-iBIDS.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", b"unsafe")
            with self.assertRaises(swpd_download.UnsafeArchiveError):
                swpd_download.safe_extract_archive(
                    archive, parent / "destination", _fake_manifest(archive)
                )
            self.assertFalse((parent / "outside.txt").exists())

    def test_unpinned_top_level_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "SingleWordProductionDutch-iBIDS.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("unexpected/participants.tsv", b"unsafe layout")
            with self.assertRaises(swpd_download.UnsafeArchiveError):
                swpd_download.safe_extract_archive(
                    archive, parent / "destination", _fake_manifest(archive)
                )

    def test_windows_alternate_stream_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "SingleWordProductionDutch-iBIDS.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(
                    "SingleWordProductionDutch-iBIDS/participants.tsv:stream", b"unsafe"
                )
            with self.assertRaises(swpd_download.UnsafeArchiveError):
                swpd_download.safe_extract_archive(
                    archive, parent / "destination", _fake_manifest(archive)
                )


class SWPDReadOnlyInventoryTests(unittest.TestCase):
    def test_confirmatory_subject_is_rejected_before_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ConfirmatoryDataLocked):
                subject_paths(Path(temporary), "sub-02")

    def test_synthetic_nwb_inventory_and_lazy_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_synthetic_swpd(Path(temporary))
            nwb_path = root / "sub-01" / "ieeg" / "sub-01_task-wordProduction_ieeg.nwb"
            state_before = (nwb_path.stat().st_size, nwb_path.stat().st_mtime_ns)
            inventory = inventory_pilot(root)
            state_after = (nwb_path.stat().st_size, nwb_path.stat().st_mtime_ns)
            self.assertEqual(state_before, state_after)
            self.assertEqual(inventory.ieeg.shape, (4096, 3))
            self.assertEqual(inventory.ieeg.rate_hz, 1024)
            self.assertEqual(inventory.audio.rate_hz, 48000)
            self.assertEqual(inventory.channels_tsv_count, 3)
            self.assertEqual(inventory.word_event_count, 2)
            self.assertEqual(inventory.fixation_event_count, 1)
            self.assertEqual(inventory.unique_prompt_count, 2)
            with SWPDRecording(root) as recording:
                neural = recording.read_ieeg(100, 120, [0, 2])
                audio = recording.read_audio(100, 130)
            self.assertEqual(neural.shape, (20, 2))
            self.assertEqual(audio.shape, (30,))

    def test_end_to_end_feature_extraction_from_synthetic_nwb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_synthetic_swpd(
                Path(temporary), seconds=4.0, audio_metadata_rate=47999.187483
            )
            features = extract_features_from_pilot(root, channel_batch_size=2)
            self.assertEqual(features.neural.shape[1], 3 * 9)
            self.assertEqual(features.mel.shape[1], 23)
            self.assertEqual(features.neural.shape[0], features.mel.shape[0])
            self.assertTrue(np.all(np.isfinite(features.neural)))
            self.assertTrue(np.all(np.isfinite(features.mel)))
            self.assertAlmostEqual(features.measured_audio_rate_hz, 47999.187483)
            self.assertEqual(features.author_processing_audio_rate_hz, 48000.0)
            self.assertEqual(features.target_audio_rate_hz, 16000.0)


class AuthorBaselineUnitTests(unittest.TestCase):
    def test_feature_shapes_and_finiteness(self) -> None:
        sample_rate = 1024
        time = np.arange(3 * sample_rate) / sample_rate
        neural = np.stack(
            [np.sin(2 * np.pi * 80 * time), np.sin(2 * np.pi * 120 * time)], axis=1
        )
        high_gamma = extract_high_gamma(neural, sample_rate)
        stacked = stack_author_context(high_gamma)
        self.assertEqual(stacked.shape[1], 18)
        self.assertEqual(stacked.shape[0], high_gamma.shape[0] - 40)
        self.assertTrue(np.all(np.isfinite(stacked)))

        audio_rate = 16000
        audio_time = np.arange(3 * audio_rate) / audio_rate
        audio = np.asarray(np.sin(2 * np.pi * 220 * audio_time) * 32767, dtype=np.int16)
        mel = extract_author_log_mel(audio, audio_rate)
        self.assertEqual(mel.shape[1], 23)
        self.assertTrue(np.all(np.isfinite(mel)))

    def test_cv_is_complete_and_handles_constant_train_column(self) -> None:
        rng = np.random.default_rng(7)
        neural = rng.normal(size=(150, 12))
        neural[:, -1] = 4.0
        weights = rng.normal(size=(12, 3))
        target = neural @ weights + rng.normal(scale=0.05, size=(150, 3))
        predictions, details = run_nonshuffled_cv(
            neural, target, folds=3, pca_components=5
        )
        self.assertEqual(predictions.shape, target.shape)
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertEqual(details["constant_neural_columns_per_fold"], [1, 1, 1])
        self.assertEqual(len(details["fold_mean_correlations"]), 3)


class _FakeMel80:
    def extract_aligned(self, audio, sample_rate, target_times):
        del audio, sample_rate
        times = np.asarray(target_times)[:, None]
        frequencies = np.arange(1, 81, dtype=np.float64)[None, :]
        return np.sin(times * frequencies).astype(np.float32)


class _FakeWhisperL345:
    def __init__(self) -> None:
        self.calls = 0

    def extract_aligned(self, audio, sample_rate, target_times):
        del audio, sample_rate
        self.calls += 1
        times = np.asarray(target_times)[:, None]
        dimensions = np.arange(1, 513, dtype=np.float64)[None, :]
        return {
            layer: np.sin(times * dimensions * (0.01 * layer)).astype(np.float32)
            for layer in (3, 4, 5)
        }


class MatchedLinearTests(unittest.TestCase):
    def test_large_absolute_acquisition_clock_is_not_mixed_with_relative_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_synthetic_swpd(
                Path(temporary),
                seconds=4.0,
                acquisition_start_seconds=9_062_565.622919,
            )
            inventory = inventory_pilot(root)
            offsets = assert_series_start_alignment(inventory)
            self.assertLess(abs(offsets["audio_minus_ieeg_seconds"]), 1e-3)
            self.assertAlmostEqual(recording_duration_seconds(inventory), 4.0)
            self.assertAlmostEqual(
                recording_relative_to_series_time(2.0, inventory.ieeg),
                inventory.ieeg.starting_time_seconds + 2.0,
            )
            first = recording_relative_sample_bounds(0.0, 2.0, inventory.audio)
            second = recording_relative_sample_bounds(2.0, 4.0, inventory.audio)
            self.assertEqual(first.stop_index, second.start_index)
            with self.assertRaisesRegex(ValueError, "absolute NWB clock"):
                recording_relative_sample_bounds(
                    0.0, inventory.audio.starting_time_seconds, inventory.audio
                )
            with SWPDRecording(root) as recording:
                block = extract_one_block(
                    recording,
                    inventory,
                    VisualBlock(
                        index=0,
                        trial_ids=("sub-01:trial-000",),
                        first_trial_index=0,
                        last_trial_index=0,
                        start_seconds=0.0,
                        stop_seconds=4.0,
                    ),
                    mel_extractor=_FakeMel80(),
                    whisper_extractor=_FakeWhisperL345(),
                    edge_guard_seconds=0.25,
                )
            self.assertGreater(len(block.sample_ids), 0)
            self.assertGreaterEqual(block.frame_times_seconds.min(), 0.25)
            self.assertLessEqual(block.frame_times_seconds.max(), 3.75)
            self.assertLess(block.frame_times_seconds.max(), 10.0)

    def test_visual_events_create_five_adjacent_twenty_trial_blocks(self) -> None:
        events = tuple(
            VisualWordEvent(
                trial_id=f"sub-01:trial-{index:03d}",
                trial_index=index,
                onset_seconds=index * 3.0,
                duration_seconds=2.0,
                prompt=f"word-{index:03d}",
            )
            for index in range(100)
        )
        blocks = make_visual_blocks(events, 300.0)
        self.assertEqual(len(blocks), 5)
        self.assertEqual([len(block.trial_ids) for block in blocks], [20] * 5)
        self.assertEqual((blocks[0].start_seconds, blocks[0].stop_seconds), (0.0, 60.0))
        self.assertEqual((blocks[-1].start_seconds, blocks[-1].stop_seconds), (240.0, 300.0))

    def test_common_grid_extracts_all_targets_from_one_whisper_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_synthetic_swpd(Path(temporary), seconds=4.0)
            with SWPDRecording(root) as recording:
                inventory = recording.inventory()
                whisper = _FakeWhisperL345()
                block = extract_one_block(
                    recording,
                    inventory,
                    VisualBlock(
                        index=0,
                        trial_ids=("sub-01:trial-000",),
                        first_trial_index=0,
                        last_trial_index=0,
                        start_seconds=0.0,
                        stop_seconds=4.0,
                    ),
                    mel_extractor=_FakeMel80(),
                    whisper_extractor=whisper,
                    edge_guard_seconds=0.25,
                )
            self.assertEqual(whisper.calls, 1)
            self.assertEqual(block.neural.shape[0], block.targets["mel80"].shape[0])
            self.assertEqual(block.targets["mel80"].shape[1], 80)
            self.assertEqual(block.targets["L3"].shape[1], 512)
            self.assertTrue(np.all(np.diff(block.frame_times_seconds) > 0))

    def test_block_cache_round_trip_is_checksummed(self) -> None:
        rng = np.random.default_rng(12)
        definition = VisualBlock(0, ("trial",), 0, 0, 0.0, 2.0)
        block = MatchedBlock(
            definition=definition,
            sample_ids=np.asarray([f"id-{i}" for i in range(6)]),
            frame_times_seconds=np.arange(6) * 0.02,
            neural=rng.normal(size=(6, 3)).astype(np.float32),
            targets={
                "mel80": rng.normal(size=(6, 80)).astype(np.float32),
                "L3": rng.normal(size=(6, 512)).astype(np.float32),
                "L4": rng.normal(size=(6, 512)).astype(np.float32),
                "L5": rng.normal(size=(6, 512)).astype(np.float32),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            save_block_cache(block, cache, extraction_fingerprint="a" * 64)
            loaded = load_block_cache(cache, 0, extraction_fingerprint="a" * 64)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded.sample_ids, block.sample_ids)
            np.testing.assert_allclose(loaded.targets["L5"], block.targets["L5"])
            with self.assertRaises(RuntimeError):
                load_block_cache(cache, 0, extraction_fingerprint="b" * 64)

    def test_metrics_and_five_fold_pipeline(self) -> None:
        rng = np.random.default_rng(33)
        blocks = []
        for block_index in range(5):
            frame_count = 24
            latent = rng.normal(size=(frame_count, 8))
            neural = (latent + rng.normal(scale=0.1, size=latent.shape)).astype(np.float32)
            targets = {}
            for name, dimension in (("mel80", 80), ("L3", 512), ("L4", 512), ("L5", 512)):
                projection = rng.normal(size=(8, dimension))
                targets[name] = (latent @ projection).astype(np.float32)
            blocks.append(
                MatchedBlock(
                    definition=VisualBlock(
                        block_index,
                        (f"trial-{block_index}",),
                        block_index,
                        block_index,
                        block_index * 2.0,
                        (block_index + 1) * 2.0,
                    ),
                    sample_ids=np.asarray(
                        [f"block-{block_index}:frame-{frame}" for frame in range(frame_count)]
                    ),
                    frame_times_seconds=block_index * 2.0 + np.arange(frame_count) * 0.02,
                    neural=neural,
                    targets=targets,
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_matched_folds(
                blocks,
                Path(temporary) / "run",
                speech_intervals=((0.0, 20.0),),
                reduced_dimension=5,
            )
            self.assertFalse(result["visual_events_are_acoustic_onsets"])
            self.assertEqual(result["target_dimension"], 5)
            self.assertEqual(len(result["folds"]), 5)
            self.assertIsNotNone(result["aggregate_test"]["L3"]["test_speech"])
            json.loads(
                (Path(temporary) / "run" / "matched_linear_summary.json").read_text(
                    encoding="utf-8"
                )
            )

    def test_regression_metrics_reject_bad_speech_mask(self) -> None:
        truth = np.ones((4, 2))
        with self.assertRaises(ValueError):
            regression_metrics(truth, truth, np.ones(3, dtype=bool))


if __name__ == "__main__":
    unittest.main()
