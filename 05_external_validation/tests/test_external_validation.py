from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import download_dataset as download  # noqa: E402
import validate_inventory as inventory  # noqa: E402


MANIFEST = MODULE_ROOT / "manifests" / "vocalmind_v2.json"


def md5_bytes(value: bytes) -> str:
    return hashlib.md5(value).hexdigest()


def zip_spec(path: Path, expected_suffix: str = ".npy") -> download.FileSpec:
    return download.FileSpec(
        name=path.name,
        size_bytes=path.stat().st_size,
        md5=download.calculate_md5(path),
        url="https://zenodo.org/api/records/14696348/files/test.zip/content",
        expected_suffixes=(expected_suffix,),
        role="unit test",
    )


class ChecksumTests(unittest.TestCase):
    def test_size_and_md5_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.zip"
            payload = b"pinned external dataset bytes"
            path.write_bytes(payload)
            download.validate_downloaded_file(path, len(payload), md5_bytes(payload))

    def test_md5_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.zip"
            payload = b"changed bytes"
            path.write_bytes(payload)
            with self.assertRaises(download.IntegrityError):
                download.validate_downloaded_file(path, len(payload), "0" * 32)

    def test_size_mismatch_is_rejected_before_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.zip"
            path.write_bytes(b"short")
            with self.assertRaises(download.IntegrityError):
                download.validate_downloaded_file(path, 999, md5_bytes(b"short"))


class ManifestTests(unittest.TestCase):
    def test_packaged_manifest_and_profile_totals(self) -> None:
        manifest = download.load_manifest(MANIFEST)
        primary, primary_files = download.resolve_profile(manifest, "overt_word_raw_primary")
        transfer, transfer_files = download.resolve_profile(manifest, "word_transfer_raw_optional")
        fidelity, fidelity_files = download.resolve_profile(
            manifest, "overt_word_processed_fidelity"
        )
        self.assertEqual(manifest.dataset["record_id"], 14696348)
        self.assertEqual(len(primary_files), 2)
        self.assertEqual(len(transfer_files), 4)
        self.assertEqual(len(fidelity_files), 3)
        self.assertEqual(primary.total_download_bytes, 224483575)
        self.assertEqual(primary.total_download_bytes, sum(item.size_bytes for item in primary_files))
        self.assertEqual(transfer.total_download_bytes, sum(item.size_bytes for item in transfer_files))
        self.assertEqual(fidelity.total_download_bytes, sum(item.size_bytes for item in fidelity_files))
        self.assertEqual(
            manifest.files["Original_sEEG_Vocalized_Word.zip"].md5,
            "1cf951dca48cea887eedf414fdd736e6",
        )
        self.assertEqual(
            manifest.files["Processed_sEEG_Vocalized_Word.zip"].md5,
            "31ebd68a28d6af7e89f30efe527c93af",
        )

    def test_unknown_profile_is_rejected(self) -> None:
        manifest = download.load_manifest(MANIFEST)
        with self.assertRaises(download.ManifestError):
            download.resolve_profile(manifest, "not_a_real_profile")

    def test_unknown_file_in_profile_is_rejected(self) -> None:
        raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw["profiles"]["overt_word_raw_primary"]["files"].append("unlisted.zip")
        with self.assertRaises(download.ManifestError):
            download.validate_manifest_dict(raw, MANIFEST)

    def test_non_zenodo_download_url_is_rejected(self) -> None:
        raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw["files"][0]["url"] = "https://example.com/payload.zip"
        with self.assertRaises(download.ManifestError):
            download.validate_manifest_dict(raw, MANIFEST)


class SafeExtractionTests(unittest.TestCase):
    def _assert_malicious_member_rejected(self, member_name: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "malicious.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(member_name, b"must not escape")
            extraction_root = root / "extracted"
            with self.assertRaises(download.UnsafeArchiveError):
                download.extract_zip_safely(archive_path, extraction_root, zip_spec(archive_path))
            self.assertFalse((root / "outside.npy").exists())
            if extraction_root.exists():
                self.assertEqual(list(extraction_root.iterdir()), [])

    def test_parent_traversal_is_rejected(self) -> None:
        self._assert_malicious_member_rejected("../outside.npy")

    def test_windows_backslash_traversal_is_rejected(self) -> None:
        self._assert_malicious_member_rejected("..\\outside.npy")

    def test_windows_drive_path_is_rejected(self) -> None:
        self._assert_malicious_member_rejected("C:\\outside.npy")

    def test_absolute_path_is_rejected(self) -> None:
        self._assert_malicious_member_rejected("/outside.npy")

    def test_changed_archive_is_rejected_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "changed.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("sample.npy", b"original")
            pinned = zip_spec(archive_path)
            with archive_path.open("ab") as handle:
                handle.write(b"changed after pinning")
            with self.assertRaises(download.IntegrityError):
                download.extract_zip_safely(archive_path, root / "extracted", pinned)
            self.assertFalse((root / "extracted").exists())

    def test_normal_archive_extracts_with_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "valid.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/sample.npy", b"array placeholder")
            extraction_root = root / "extracted"
            target = download.extract_zip_safely(
                archive_path,
                extraction_root,
                zip_spec(archive_path),
            )
            self.assertEqual((target / "nested" / "sample.npy").read_bytes(), b"array placeholder")
            receipt = json.loads((target / ".extraction_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["member_count"], 1)
            self.assertEqual(receipt["archive_name"], "valid.zip")
            summary = inventory.inventory_extraction(extraction_root, zip_spec(archive_path))
            self.assertEqual(summary["file_count"], 1)
            self.assertEqual(summary["suffix_counts"], {".npy": 1})


if __name__ == "__main__":
    unittest.main()
