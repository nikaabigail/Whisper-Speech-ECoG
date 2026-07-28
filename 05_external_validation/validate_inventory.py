#!/usr/bin/env python3
"""Validate downloaded VocalMind ZIPs and extracted NPY/CSV/WAV inventories."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from download_dataset import (
    DEFAULT_MANIFEST,
    DatasetManifest,
    FileSpec,
    IntegrityError,
    ManifestError,
    load_manifest,
    resolve_profile,
    safe_child,
    validate_downloaded_file,
)


RECEIPT_NAME = ".extraction_receipt.json"


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Cannot read extraction receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"Extraction receipt is not a JSON object: {path}")
    return value


def inventory_extraction(extraction_root: Path, spec: FileSpec) -> dict[str, Any]:
    directory = safe_child(extraction_root, Path(spec.name).stem)
    if not directory.is_dir():
        raise IntegrityError(f"Missing extraction directory for {spec.name}: {directory}")

    receipt = _read_receipt(directory / RECEIPT_NAME)
    if receipt.get("archive_name") != spec.name:
        raise IntegrityError(f"Receipt archive mismatch in {directory}")
    if receipt.get("archive_size_bytes") != spec.size_bytes:
        raise IntegrityError(f"Receipt size mismatch in {directory}")
    if str(receipt.get("archive_md5", "")).casefold() != spec.md5:
        raise IntegrityError(f"Receipt MD5 mismatch in {directory}")

    suffix_counts: Counter[str] = Counter()
    total_bytes = 0
    casefolded_paths: set[str] = set()
    unexpected: list[str] = []
    empty: list[str] = []
    count = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise IntegrityError(f"Symlink found in extracted data: {path}")
        if not path.is_file() or path.name == RECEIPT_NAME:
            continue
        relative = path.relative_to(directory).as_posix()
        collision_key = relative.casefold()
        if collision_key in casefolded_paths:
            raise IntegrityError(f"Case-colliding extracted path: {relative}")
        casefolded_paths.add(collision_key)
        suffix = path.suffix.casefold()
        suffix_counts[suffix] += 1
        count += 1
        size = path.stat().st_size
        total_bytes += size
        if size == 0:
            empty.append(relative)
        if suffix not in spec.expected_suffixes:
            unexpected.append(relative)

    if count == 0:
        raise IntegrityError(f"No data files found in {directory}")
    if unexpected:
        sample = ", ".join(unexpected[:5])
        raise IntegrityError(f"Unexpected file types in {directory}: {sample}")
    if empty:
        sample = ", ".join(empty[:5])
        raise IntegrityError(f"Empty data files in {directory}: {sample}")
    if receipt.get("member_count") != count:
        raise IntegrityError(
            f"Member count differs from receipt in {directory}: "
            f"expected {receipt.get('member_count')}, got {count}"
        )
    if receipt.get("total_extracted_bytes") != total_bytes:
        raise IntegrityError(
            f"Extracted byte count differs from receipt in {directory}: "
            f"expected {receipt.get('total_extracted_bytes')}, got {total_bytes}"
        )

    return {
        "archive": spec.name,
        "role": spec.role,
        "directory": str(directory),
        "file_count": count,
        "total_bytes": total_bytes,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "expected_suffixes": list(spec.expected_suffixes),
        "archive_md5": spec.md5,
        "status": "ok",
    }


def build_report(
    manifest: DatasetManifest,
    profile_name: str,
    data_root: Path,
    *,
    verify_archives: bool = True,
) -> dict[str, Any]:
    profile, file_specs = resolve_profile(manifest, profile_name)
    archives_root = data_root / "archives"
    extraction_root = data_root / "extracted"
    entries = []
    for spec in file_specs:
        if verify_archives:
            archive_path = safe_child(archives_root, spec.name)
            validate_downloaded_file(archive_path, spec.size_bytes, spec.md5)
        entries.append(inventory_extraction(extraction_root, spec))
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": manifest.dataset["name"],
            "record_id": manifest.dataset["record_id"],
            "record_version": manifest.dataset["record_version"],
            "doi": manifest.dataset["doi"],
            "license": manifest.dataset["license"],
        },
        "manifest": str(manifest.path),
        "profile": profile.name,
        "profile_download_bytes": profile.total_download_bytes,
        "archives_verified": verify_archives,
        "entries": entries,
        "summary": {
            "archive_count": len(entries),
            "file_count": sum(item["file_count"] for item in entries),
            "total_extracted_bytes": sum(item["total_bytes"] for item in entries),
            "status": "ok",
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", default="overt_word_raw_primary")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--skip-archive-checks",
        action="store_true",
        help="inspect extracted files without rehashing downloaded ZIPs",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    report = build_report(
        manifest,
        args.profile,
        args.data_root.expanduser().resolve(),
        verify_archives=not args.skip_archive_checks,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.report:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f"{report_path.name}.partial")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.write("\n")
        temporary.replace(report_path)
        print(f"[saved] {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, IntegrityError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
