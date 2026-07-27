from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_ROOT))

from whisper_ecog_ext.data.vocalmind import (  # noqa: E402
    DEFAULT_VOCALMIND_CONTRACT,
    VocalMindAdapter,
    VocalMindContract,
    VocalMindDataError,
    build_primary_split_manifest,
)


SYNTHETIC_CONTRACT = VocalMindContract(
    words=("Alpha", "Beta"),
    channel_ids=(9, 10, 11),
    eeg_sample_rate_hz=4,
    eeg_samples=4,
    audio_sample_rate_hz=8,
    audio_frames=8,
    audio_channels=1,
    audio_sample_width_bytes=2,
    expected_missing=frozenset({("Beta", 6)}),
)


def write_wav(path: Path, sample_rate: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.arange(8, dtype="<i2").tobytes())


def write_csv(
    path: Path,
    *,
    nonfinite: bool = False,
    wrong_header: bool = False,
    row_count: int = SYNTHETIC_CONTRACT.eeg_samples,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = (10, 9, 11) if wrong_header else SYNTHETIC_CONTRACT.channel_ids
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(channels)
        for row_index in range(row_count):
            row = [float(row_index + column) for column in range(len(channels))]
            if nonfinite and row_index == 0:
                row[0] = float("nan")
            writer.writerow(row)


def make_dataset(root: Path) -> None:
    audio_root = root / "extracted" / "Original_Audio_Word" / "Original_Audio_Word"
    eeg_root = (
        root
        / "extracted"
        / "Original_sEEG_Vocalized_Word"
        / "Original_sEEG_Vocalized_Word"
    )
    for word, repetition in sorted(SYNTHETIC_CONTRACT.expected_keys):
        write_wav(audio_root / f"Audio_{word}_{repetition}.wav")
        write_csv(eeg_root / f"Vocalized_{word}_{repetition}.csv")


class ContractTests(unittest.TestCase):
    def test_real_release_contract_is_pinned(self) -> None:
        contract = DEFAULT_VOCALMIND_CONTRACT
        self.assertEqual(len(contract.words), 20)
        self.assertEqual(len(contract.channel_ids), 110)
        self.assertEqual(contract.eeg_sample_rate_hz, 1000)
        self.assertEqual(contract.eeg_samples, 3000)
        self.assertEqual(contract.audio_sample_rate_hz, 44_100)
        self.assertEqual(contract.audio_frames, 132_300)
        self.assertEqual(contract.audio_channels, 2)
        self.assertEqual(contract.expected_missing, frozenset({("ShuMu", 6)}))
        self.assertEqual(len(contract.expected_keys), 119)


class AdapterTests(unittest.TestCase):
    def test_read_only_arrays_and_primary_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            adapter = VocalMindAdapter(root, SYNTHETIC_CONTRACT)
            trials = adapter.discover()
            self.assertEqual(len(trials), 11)
            eeg = adapter.load_eeg(trials[0])
            audio = adapter.load_audio(trials[0])
            self.assertEqual(eeg.shape, (4, 3))
            self.assertEqual(audio.shape, (8, 1))
            self.assertFalse(eeg.flags.writeable)
            self.assertFalse(audio.flags.writeable)

            index = adapter.build_index(deep=True, hash_files=True)
            splits = build_primary_split_manifest(index, SYNTHETIC_CONTRACT)
            self.assertEqual(index["counts"]["all_paired_trials"], 11)
            self.assertEqual(index["counts"]["primary_trials_reps_1_5"], 10)
            self.assertEqual(len(splits["folds"]), 5)
            all_test_ids = []
            for fold in splits["folds"]:
                self.assertEqual(fold["counts"], {"train": 6, "validation": 2, "test": 2})
                ids = (
                    fold["train_trial_ids"]
                    + fold["validation_trial_ids"]
                    + fold["test_trial_ids"]
                )
                self.assertFalse(any("rep06" in trial_id for trial_id in ids))
                all_test_ids.extend(fold["test_trial_ids"])
            self.assertEqual(len(all_test_ids), 10)
            self.assertEqual(len(set(all_test_ids)), 10)

    def test_pairing_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            missing = next(root.rglob("Vocalized_Alpha_1.csv"))
            missing.unlink()
            with self.assertRaisesRegex(VocalMindDataError, "pairing mismatch"):
                VocalMindAdapter(root, SYNTHETIC_CONTRACT).discover()

    def test_primary_trial_missing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            next(root.rglob("Audio_Alpha_1.wav")).unlink()
            next(root.rglob("Vocalized_Alpha_1.csv")).unlink()
            with self.assertRaisesRegex(VocalMindDataError, "frozen contract"):
                VocalMindAdapter(root, SYNTHETIC_CONTRACT).discover()

    def test_channel_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            target = next(root.rglob("Vocalized_Alpha_1.csv"))
            write_csv(target, wrong_header=True)
            adapter = VocalMindAdapter(root, SYNTHETIC_CONTRACT)
            with self.assertRaisesRegex(VocalMindDataError, "Channel order mismatch"):
                adapter.build_index(deep=False, hash_files=False)

    def test_nonfinite_eeg_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            target = next(root.rglob("Vocalized_Alpha_1.csv"))
            write_csv(target, nonfinite=True)
            adapter = VocalMindAdapter(root, SYNTHETIC_CONTRACT)
            with self.assertRaisesRegex(VocalMindDataError, "Non-finite EEG"):
                adapter.build_index(deep=True, hash_files=False)

    def test_eeg_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            target = next(root.rglob("Vocalized_Alpha_1.csv"))
            write_csv(target, row_count=SYNTHETIC_CONTRACT.eeg_samples - 1)
            adapter = VocalMindAdapter(root, SYNTHETIC_CONTRACT)
            with self.assertRaisesRegex(VocalMindDataError, "EEG shape mismatch"):
                adapter.build_index(deep=True, hash_files=False)

    def test_audio_rate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            target = next(root.rglob("Audio_Alpha_1.wav"))
            write_wav(target, sample_rate=16)
            adapter = VocalMindAdapter(root, SYNTHETIC_CONTRACT)
            with self.assertRaisesRegex(VocalMindDataError, "WAV contract mismatch"):
                adapter.build_index(deep=False, hash_files=False)


if __name__ == "__main__":
    unittest.main()
