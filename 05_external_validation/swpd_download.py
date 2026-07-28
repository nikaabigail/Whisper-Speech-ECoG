#!/usr/bin/env python3
"""Download, verify, and safely extract the pinned SWPD OSF archive.

The implementation intentionally uses only Python's standard library.  Raw
participant data must live outside this source checkout.  OSF currently
reports two different lengths through HEAD and ranged GET responses, so a
download is accepted only after its actual length and SHA-256 match the pinned
manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifests" / "swpd_osf_nrgx6.json"
USER_AGENT = "Whisper-Speech-ECoG-SWPD/1.0"
CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_ENTRIES = 100_000
MAX_EXTRACTED_BYTES = 12 * 1024**3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)
ALLOWED_REDIRECT_HOSTS = {
    "osf.io",
    "files.de-1.osf.io",
    "storage.googleapis.com",
}


class ManifestError(ValueError):
    """The checked-in SWPD manifest is malformed."""


class IntegrityError(RuntimeError):
    """A local or downloaded archive differs from the pinned artifact."""


class UnsafeArchiveError(RuntimeError):
    """A ZIP entry could escape the selected extraction directory."""


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read SWPD manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ManifestError("SWPD manifest schema_version must be 1")
    dataset = raw.get("dataset")
    archive = raw.get("archive")
    if not isinstance(dataset, dict) or dataset.get("pilot_subject") != "sub-01":
        raise ManifestError("dataset.pilot_subject must be sub-01")
    if not isinstance(archive, dict):
        raise ManifestError("archive must be an object")
    required = {"name", "url", "size_bytes", "sha256", "archive_root"}
    missing = sorted(required - set(archive))
    if missing:
        raise ManifestError(f"archive is missing keys: {missing}")
    if archive["name"] != "SingleWordProductionDutch-iBIDS.zip":
        raise ManifestError("Unexpected SWPD archive name")
    parsed = urlparse(str(archive["url"]))
    if parsed.scheme != "https" or parsed.hostname != "osf.io":
        raise ManifestError("archive.url must be the pinned HTTPS OSF URL")
    if not isinstance(archive["size_bytes"], int) or archive["size_bytes"] <= 0:
        raise ManifestError("archive.size_bytes must be positive")
    checksum = str(archive["sha256"]).casefold()
    if not SHA256_RE.fullmatch(checksum):
        raise ManifestError("archive.sha256 must contain 64 hexadecimal characters")
    archive["sha256"] = checksum
    if Path(str(archive["archive_root"])).name != archive["archive_root"]:
        raise ManifestError("archive.archive_root must be a safe basename")
    return raw


def calculate_sha256(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, *, size_bytes: int, sha256: str) -> None:
    if not path.is_file():
        raise IntegrityError(f"Missing archive: {path}")
    actual_size = path.stat().st_size
    if actual_size != size_bytes:
        raise IntegrityError(
            f"Size mismatch for {path.name}: expected {size_bytes}, got {actual_size}. "
            "Do not substitute the inconsistent OSF HEAD Content-Length."
        )
    actual_hash = calculate_sha256(path)
    if actual_hash.casefold() != sha256.casefold():
        raise IntegrityError(
            f"SHA-256 mismatch for {path.name}: expected {sha256}, got {actual_hash}"
        )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _safe_child(parent: Path, basename: str) -> Path:
    if not basename or Path(basename).name != basename or "\\" in basename:
        raise ValueError(f"Unsafe basename: {basename!r}")
    candidate = (parent.resolve() / basename).resolve()
    if not _is_within(candidate, parent):
        raise ValueError(f"Path escapes destination: {basename!r}")
    return candidate


def require_external_destination(destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    if destination == HERE or _is_within(destination, HERE):
        raise ValueError(
            f"Raw SWPD data must be outside the source checkout {HERE}; got {destination}"
        )
    return destination


def _validate_redirect(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or host not in ALLOWED_REDIRECT_HOSTS:
        raise RuntimeError(f"SWPD download redirected to an unapproved URL: {url}")


def _parse_content_range(value: str | None, expected_start: int, expected_total: int) -> None:
    match = CONTENT_RANGE_RE.fullmatch((value or "").strip())
    if not match or int(match.group(1)) != expected_start:
        raise RuntimeError(f"Invalid Content-Range for resumed SWPD download: {value!r}")
    total = match.group(3)
    if total != "*" and int(total) != expected_total:
        raise IntegrityError(
            f"Ranged GET reports {total} bytes, but the pinned ZIP has {expected_total}"
        )


def _progress(completed: int, total: int, final: bool = False) -> None:
    percent = completed * 100.0 / total
    text = (
        f"[SWPD] {completed / 1024**2:,.1f}/{total / 1024**2:,.1f} MiB "
        f"({percent:5.1f}%)"
    )
    print(text, end="\n" if final else "\r", flush=True)


def download_archive(
    manifest: dict[str, Any],
    destination: Path,
    *,
    force: bool = False,
    timeout_seconds: int = 60,
) -> Path:
    archive = manifest["archive"]
    archives_dir = destination / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    final_path = _safe_child(archives_dir, archive["name"])
    partial_path = _safe_child(archives_dir, f"{archive['name']}.partial")

    if final_path.exists():
        try:
            verify_archive(
                final_path,
                size_bytes=archive["size_bytes"],
                sha256=archive["sha256"],
            )
        except IntegrityError:
            if not force:
                raise
            final_path.unlink()
        else:
            print(f"[verified] {final_path}")
            return final_path

    if force and partial_path.exists():
        partial_path.unlink()
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    if offset > archive["size_bytes"]:
        raise IntegrityError("Partial SWPD archive is larger than the pinned archive")

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"[resume] SWPD at byte {offset}")
    else:
        print("[download] pinned SWPD OSF archive")

    try:
        response = urlopen(Request(archive["url"], headers=headers), timeout=timeout_seconds)
    except HTTPError as exc:
        if exc.code == 416 and offset == archive["size_bytes"]:
            verify_archive(
                partial_path,
                size_bytes=archive["size_bytes"],
                sha256=archive["sha256"],
            )
            os.replace(partial_path, final_path)
            return final_path
        raise RuntimeError(f"HTTP {exc.code} while downloading SWPD") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while downloading SWPD: {exc}") from exc

    with response:
        _validate_redirect(response.geturl())
        status = getattr(response, "status", response.getcode())
        if offset and status == 206:
            _parse_content_range(
                response.headers.get("Content-Range"), offset, archive["size_bytes"]
            )
            mode = "ab"
        elif status == 200:
            if offset:
                print("[restart] OSF ignored Range; restarting the partial download")
            offset = 0
            mode = "wb"
        else:
            raise RuntimeError(f"Unexpected SWPD HTTP status: {status}")

        completed = offset
        last_update = 0.0
        with partial_path.open(mode) as target:
            while chunk := response.read(CHUNK_BYTES):
                target.write(chunk)
                completed += len(chunk)
                if completed > archive["size_bytes"]:
                    raise IntegrityError("OSF sent more bytes than the pinned SWPD ZIP")
                now = time.monotonic()
                if now - last_update >= 1.0:
                    _progress(completed, archive["size_bytes"])
                    last_update = now
            target.flush()
            os.fsync(target.fileno())

    _progress(partial_path.stat().st_size, archive["size_bytes"], final=True)
    verify_archive(
        partial_path,
        size_bytes=archive["size_bytes"],
        sha256=archive["sha256"],
    )
    os.replace(partial_path, final_path)
    print(f"[verified] {final_path}")
    return final_path


def _normalise_zip_name(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise UnsafeArchiveError(f"Empty or NUL-containing ZIP member: {name!r}")
    unified = name.replace("\\", "/")
    if unified.startswith("/") or re.match(r"^[A-Za-z]:", unified):
        raise UnsafeArchiveError(f"Absolute ZIP member: {name!r}")
    path = PurePosixPath(unified)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveError(f"Traversal or ambiguous ZIP member: {name!r}")
    if any(":" in part for part in path.parts):
        raise UnsafeArchiveError(f"Windows drive/alternate-stream ZIP member: {name!r}")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _remove_tree_within(path: Path, parent: Path) -> None:
    if path.resolve() == parent.resolve() or not _is_within(path, parent):
        raise RuntimeError(f"Refusing to remove an unsafe path: {path}")
    if path.exists():
        shutil.rmtree(path)


def safe_extract_archive(
    archive_path: Path,
    destination: Path,
    manifest: dict[str, Any],
    *,
    force: bool = False,
    max_entries: int = MAX_ARCHIVE_ENTRIES,
    max_extracted_bytes: int = MAX_EXTRACTED_BYTES,
) -> Path:
    archive_spec = manifest["archive"]
    # Recheck at the extraction boundary as protection against a changed file
    # between download completion and ZIP opening.
    verify_archive(
        archive_path,
        size_bytes=archive_spec["size_bytes"],
        sha256=archive_spec["sha256"],
    )
    extracted_parent = destination / "extracted"
    extracted_parent.mkdir(parents=True, exist_ok=True)
    target = _safe_child(extracted_parent, archive_spec["archive_root"])
    stage = _safe_child(extracted_parent, ".swpd-extract.partial")
    receipt_name = ".swpd_extraction_receipt.json"

    if target.exists():
        try:
            receipt = json.loads((target / receipt_name).read_text(encoding="utf-8"))
            reusable = (
                receipt.get("archive_sha256", "").casefold() == archive_spec["sha256"]
                and receipt.get("archive_size_bytes") == archive_spec["size_bytes"]
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            reusable = False
        if reusable:
            print(f"[extracted] reuse {target}")
            return target
        if not force:
            raise IntegrityError(
                f"Existing extraction has no matching receipt: {target}; use --force"
            )
        _remove_tree_within(target, extracted_parent)
    if stage.exists():
        if not force:
            raise IntegrityError(f"Incomplete extraction exists: {stage}; use --force")
        _remove_tree_within(stage, extracted_parent)
    stage.mkdir()

    total_bytes = 0
    members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                if info.flag_bits & 0x1:
                    raise UnsafeArchiveError(f"Encrypted member: {info.filename}")
                if _is_symlink(info):
                    raise UnsafeArchiveError(f"Symlink member: {info.filename}")
                normal = _normalise_zip_name(info.filename.rstrip("/"))
                if normal.parts[0] != archive_spec["archive_root"]:
                    raise UnsafeArchiveError(
                        f"ZIP member is outside the pinned top-level directory: {info.filename}"
                    )
                key = normal.as_posix().casefold()
                if key in seen:
                    raise UnsafeArchiveError(f"Duplicate/case-colliding member: {info.filename}")
                seen.add(key)
                output = (stage / Path(*normal.parts)).resolve()
                if not _is_within(output, stage):
                    raise UnsafeArchiveError(f"Member escapes extraction root: {info.filename}")
                if info.is_dir():
                    continue
                members.append((info, normal))
                total_bytes += info.file_size
                if len(members) > max_entries:
                    raise UnsafeArchiveError("SWPD ZIP contains too many entries")
                if total_bytes > max_extracted_bytes:
                    raise UnsafeArchiveError("SWPD ZIP exceeds the extraction safety limit")

            for info, normal in members:
                output = stage / Path(*normal.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".partial")
                with archive.open(info, "r") as source, temporary.open("xb") as sink:
                    shutil.copyfileobj(source, sink, CHUNK_BYTES)
                    sink.flush()
                    os.fsync(sink.fileno())
                if temporary.stat().st_size != info.file_size:
                    raise IntegrityError(f"Extracted size mismatch for {info.filename}")
                os.replace(temporary, output)

        staged_root = stage / archive_spec["archive_root"]
        if not staged_root.is_dir():
            raise UnsafeArchiveError(
                f"ZIP does not contain expected root {archive_spec['archive_root']}"
            )
        receipt = {
            "schema_version": 1,
            "archive_name": archive_spec["name"],
            "archive_size_bytes": archive_spec["size_bytes"],
            "archive_sha256": archive_spec["sha256"],
            "member_count": len(members),
            "total_extracted_bytes": total_bytes,
            "extracted_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staged_root / receipt_name).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staged_root, target)
        stage.rmdir()
    except Exception:
        _remove_tree_within(stage, extracted_parent)
        raise

    print(f"[extracted] {target}")
    return target


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    manifest = load_manifest(args.manifest)
    destination = require_external_destination(args.destination)
    archive = download_archive(
        manifest,
        destination,
        force=args.force,
        timeout_seconds=args.timeout_seconds,
    )
    if not args.download_only:
        safe_extract_archive(archive, destination, manifest, force=args.force)
    print("[done] SWPD archive matches the pinned SHA-256")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; the .partial archive can resume safely.", file=sys.stderr)
        raise SystemExit(130)
    except (ManifestError, IntegrityError, UnsafeArchiveError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
