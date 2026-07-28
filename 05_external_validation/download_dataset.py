#!/usr/bin/env python3
"""Resume, verify, and safely extract a pinned external-dataset profile.

Only Python's standard library is required. Participant data must be written to
a destination outside this source repository.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
DEFAULT_MANIFEST = HERE / "manifests" / "vocalmind_v2.json"
ALLOWED_DATA_SUFFIXES = {".npy", ".csv", ".wav"}
ALLOWED_DOWNLOAD_HOSTS = {"zenodo.org", "www.zenodo.org"}
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)
USER_AGENT = "Whisper-Speech-ECoG-external-validator/1.0"
DEFAULT_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_ENTRIES = 100_000
DEFAULT_MAX_EXTRACTED_BYTES = 8 * 1024**3


class ManifestError(ValueError):
    """The pinned dataset manifest is malformed or internally inconsistent."""


class IntegrityError(RuntimeError):
    """A downloaded file does not match its pinned size or checksum."""


class UnsafeArchiveError(RuntimeError):
    """A ZIP member could escape or otherwise make extraction unsafe."""


@dataclass(frozen=True)
class FileSpec:
    name: str
    size_bytes: int
    md5: str
    url: str
    expected_suffixes: tuple[str, ...]
    role: str


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str
    files: tuple[str, ...]
    total_download_bytes: int


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    dataset: dict[str, Any]
    files: dict[str, FileSpec]
    profiles: dict[str, ProfileSpec]


def _new_md5() -> "hashlib._Hash":
    try:
        return hashlib.md5(usedforsecurity=False)
    except TypeError:  # Python builds without the usedforsecurity keyword
        return hashlib.md5()


def calculate_md5(path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = _new_md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_downloaded_file(
    path: Path,
    expected_size: int,
    expected_md5: str,
) -> None:
    if not path.is_file():
        raise IntegrityError(f"Missing file: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise IntegrityError(
            f"Size mismatch for {path.name}: expected {expected_size}, got {actual_size}"
        )
    actual_md5 = calculate_md5(path)
    if actual_md5.casefold() != expected_md5.casefold():
        raise IntegrityError(
            f"MD5 mismatch for {path.name}: expected {expected_md5}, got {actual_md5}"
        )


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be a JSON array")
    return value


def _validate_https_zenodo_url(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{label} must be a string")
    parsed = urlparse(value)
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() not in ALLOWED_DOWNLOAD_HOSTS:
        raise ManifestError(f"{label} must be an HTTPS Zenodo URL")
    return value


def validate_manifest_dict(raw: dict[str, Any], source: Path) -> DatasetManifest:
    if raw.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")

    dataset = _require_dict(raw.get("dataset"), "dataset")
    record_id = dataset.get("record_id")
    if not isinstance(record_id, int) or record_id <= 0:
        raise ManifestError("dataset.record_id must be a positive integer")
    _validate_https_zenodo_url(dataset.get("landing_page"), "dataset.landing_page")
    if not isinstance(dataset.get("license"), str) or not dataset["license"]:
        raise ManifestError("dataset.license must be a non-empty string")

    file_specs: dict[str, FileSpec] = {}
    for index, item_value in enumerate(_require_list(raw.get("files"), "files")):
        item = _require_dict(item_value, f"files[{index}]")
        name = item.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name:
            raise ManifestError(f"files[{index}].name must be a safe basename")
        if not name.casefold().endswith(".zip"):
            raise ManifestError(f"Only ZIP archives are supported: {name}")
        if name.casefold() in {key.casefold() for key in file_specs}:
            raise ManifestError(f"Duplicate archive name: {name}")

        size_bytes = item.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ManifestError(f"{name}: size_bytes must be a positive integer")
        md5 = item.get("md5")
        if not isinstance(md5, str) or not MD5_RE.fullmatch(md5.casefold()):
            raise ManifestError(f"{name}: md5 must contain 32 hexadecimal characters")
        url = _validate_https_zenodo_url(item.get("url"), f"{name}.url")
        expected_url_path = f"/api/records/{record_id}/files/{name}/content"
        if urlparse(url).path != expected_url_path:
            raise ManifestError(
                f"{name}: URL path must be {expected_url_path!r} for the pinned record"
            )
        suffix_values = _require_list(item.get("expected_suffixes"), f"{name}.expected_suffixes")
        suffixes = tuple(str(value).casefold() for value in suffix_values)
        if not suffixes or any(value not in ALLOWED_DATA_SUFFIXES for value in suffixes):
            raise ManifestError(
                f"{name}: expected_suffixes must use {sorted(ALLOWED_DATA_SUFFIXES)}"
            )
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ManifestError(f"{name}: role must be a non-empty string")
        file_specs[name] = FileSpec(name, size_bytes, md5.casefold(), url, suffixes, role)

    if not file_specs:
        raise ManifestError("files must not be empty")

    profiles_value = _require_dict(raw.get("profiles"), "profiles")
    profiles: dict[str, ProfileSpec] = {}
    for name, item_value in profiles_value.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ManifestError(f"Unsafe profile name: {name!r}")
        item = _require_dict(item_value, f"profiles.{name}")
        file_names_value = _require_list(item.get("files"), f"profiles.{name}.files")
        file_names = tuple(str(value) for value in file_names_value)
        if not file_names or len(set(file_names)) != len(file_names):
            raise ManifestError(f"profiles.{name}.files must be non-empty and unique")
        unknown = sorted(set(file_names) - set(file_specs))
        if unknown:
            raise ManifestError(f"profiles.{name} references unknown files: {unknown}")
        total = item.get("total_download_bytes")
        calculated_total = sum(file_specs[file_name].size_bytes for file_name in file_names)
        if total != calculated_total:
            raise ManifestError(
                f"profiles.{name}.total_download_bytes is {total}, expected {calculated_total}"
            )
        description = item.get("description")
        if not isinstance(description, str) or not description:
            raise ManifestError(f"profiles.{name}.description must be non-empty")
        profiles[name] = ProfileSpec(name, description, file_names, total)

    if not profiles:
        raise ManifestError("profiles must not be empty")
    return DatasetManifest(source, dataset, file_specs, profiles)


def load_manifest(path: Path) -> DatasetManifest:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read manifest {path}: {exc}") from exc
    return validate_manifest_dict(_require_dict(raw, "manifest root"), path.resolve())


def resolve_profile(manifest: DatasetManifest, name: str) -> tuple[ProfileSpec, list[FileSpec]]:
    try:
        profile = manifest.profiles[name]
    except KeyError as exc:
        raise ManifestError(
            f"Unknown profile {name!r}; choose from {', '.join(sorted(manifest.profiles))}"
        ) from exc
    return profile, [manifest.files[file_name] for file_name in profile.files]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def safe_child(parent: Path, basename: str) -> Path:
    if not basename or Path(basename).name != basename or "\\" in basename:
        raise ValueError(f"Unsafe basename: {basename!r}")
    parent_resolved = parent.resolve()
    candidate = (parent_resolved / basename).resolve()
    if not _is_within(candidate, parent_resolved):
        raise ValueError(f"Path escapes destination: {basename!r}")
    return candidate


def _validate_response_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() not in ALLOWED_DOWNLOAD_HOSTS:
        raise RuntimeError(f"Download redirected outside HTTPS Zenodo: {url}")


def _progress(name: str, completed: int, total: int, *, final: bool = False) -> None:
    percent = 100.0 * completed / total if total else 0.0
    line = f"[{name}] {completed / 1024**2:,.1f}/{total / 1024**2:,.1f} MiB ({percent:5.1f}%)"
    print(line, end="\n" if final else "\r", flush=True)


def _parse_content_range(value: str | None, expected_start: int, expected_total: int) -> None:
    match = CONTENT_RANGE_RE.fullmatch((value or "").strip())
    if not match or int(match.group(1)) != expected_start:
        raise RuntimeError(
            f"Server returned an invalid Content-Range for resume: {value!r}"
        )
    reported_total = match.group(3)
    if reported_total != "*" and int(reported_total) != expected_total:
        raise RuntimeError(
            f"Server Content-Range total differs from the manifest: {value!r}"
        )


def download_file(
    spec: FileSpec,
    archives_dir: Path,
    *,
    force: bool = False,
    timeout_seconds: int = 60,
) -> Path:
    archives_dir.mkdir(parents=True, exist_ok=True)
    final_path = safe_child(archives_dir, spec.name)
    partial_path = safe_child(archives_dir, f"{spec.name}.partial")

    if final_path.exists():
        try:
            validate_downloaded_file(final_path, spec.size_bytes, spec.md5)
        except IntegrityError:
            if not force:
                raise IntegrityError(
                    f"Existing archive is invalid: {final_path}. Use --force to replace it."
                )
            final_path.unlink()
        else:
            print(f"[verified] {final_path}")
            return final_path

    if force and partial_path.exists():
        partial_path.unlink()
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    if offset > spec.size_bytes:
        raise IntegrityError(
            f"Partial file is larger than the manifest: {partial_path}. Use --force to restart."
        )

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"[resume] {spec.name} from byte {offset}")
    else:
        print(f"[download] {spec.name}")
    request = Request(spec.url, headers=headers)

    try:
        response = urlopen(request, timeout=timeout_seconds)
    except HTTPError as exc:
        if exc.code == 416 and offset == spec.size_bytes:
            validate_downloaded_file(partial_path, spec.size_bytes, spec.md5)
            os.replace(partial_path, final_path)
            print(f"[verified] {final_path}")
            return final_path
        raise RuntimeError(f"HTTP {exc.code} while downloading {spec.name}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while downloading {spec.name}: {exc}") from exc

    with response:
        _validate_response_url(response.geturl())
        status = getattr(response, "status", response.getcode())
        if offset and status == 206:
            _parse_content_range(
                response.headers.get("Content-Range"),
                offset,
                spec.size_bytes,
            )
            mode = "ab"
        elif status == 200:
            if offset:
                print(f"[restart] Server ignored Range for {spec.name}")
            offset = 0
            mode = "wb"
        else:
            raise RuntimeError(f"Unexpected HTTP status {status} for {spec.name}")

        completed = offset
        last_update = 0.0
        with partial_path.open(mode) as target:
            while chunk := response.read(DEFAULT_CHUNK_BYTES):
                target.write(chunk)
                completed += len(chunk)
                if completed > spec.size_bytes:
                    raise IntegrityError(
                        f"Server sent more bytes than pinned for {spec.name}"
                    )
                now = time.monotonic()
                if now - last_update >= 1.0:
                    _progress(spec.name, completed, spec.size_bytes)
                    last_update = now
            target.flush()
            os.fsync(target.fileno())

    _progress(spec.name, partial_path.stat().st_size, spec.size_bytes, final=True)
    validate_downloaded_file(partial_path, spec.size_bytes, spec.md5)
    os.replace(partial_path, final_path)
    print(f"[verified] {final_path}")
    return final_path


def _normalise_member_name(raw_name: str) -> PurePosixPath:
    if not raw_name or "\x00" in raw_name:
        raise UnsafeArchiveError(f"Empty or NUL-containing ZIP member: {raw_name!r}")
    unified = raw_name.replace("\\", "/")
    if unified.startswith("/") or re.match(r"^[A-Za-z]:", unified):
        raise UnsafeArchiveError(f"Absolute or drive ZIP path: {raw_name!r}")
    path = PurePosixPath(unified)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveError(f"Traversal or ambiguous ZIP path: {raw_name!r}")
    return path


def _member_is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(unix_mode)


def _member_is_unsupported_special_file(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    return bool(file_type) and not (
        stat.S_ISREG(unix_mode) or stat.S_ISDIR(unix_mode) or stat.S_ISLNK(unix_mode)
    )


def _safe_remove_tree(path: Path, parent: Path) -> None:
    path_resolved = path.resolve()
    parent_resolved = parent.resolve()
    if path_resolved == parent_resolved or not _is_within(path_resolved, parent_resolved):
        raise RuntimeError(f"Refusing recursive removal outside extraction root: {path}")
    if path.exists():
        shutil.rmtree(path)


def _preflight_zip(
    archive: zipfile.ZipFile,
    stage_dir: Path,
    *,
    expected_suffixes: tuple[str, ...],
    max_entries: int,
    max_extracted_bytes: int,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    selected: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen_casefolded: set[str] = set()
    total_bytes = 0
    for info in archive.infolist():
        if info.flag_bits & 0x1:
            raise UnsafeArchiveError(f"Encrypted ZIP member is not supported: {info.filename}")
        if _member_is_symlink(info):
            raise UnsafeArchiveError(f"Symlink ZIP member is not allowed: {info.filename}")
        if _member_is_unsupported_special_file(info):
            raise UnsafeArchiveError(f"Special-file ZIP member is not allowed: {info.filename}")
        member = _normalise_member_name(info.filename.rstrip("/"))
        relative = member.as_posix()
        collision_key = relative.casefold()
        if collision_key in seen_casefolded:
            raise UnsafeArchiveError(f"Duplicate/case-colliding ZIP member: {info.filename}")
        seen_casefolded.add(collision_key)
        destination = (stage_dir / Path(*member.parts)).resolve()
        if not _is_within(destination, stage_dir.resolve()):
            raise UnsafeArchiveError(f"ZIP member escapes extraction root: {info.filename}")
        if info.is_dir():
            continue
        suffix = Path(member.name).suffix.casefold()
        if suffix not in expected_suffixes:
            raise UnsafeArchiveError(
                f"Unexpected {suffix or '<no suffix>'} member in {info.filename}; "
                f"expected {list(expected_suffixes)}"
            )
        selected.append((info, member))
        total_bytes += info.file_size
        if len(selected) > max_entries:
            raise UnsafeArchiveError(f"ZIP has more than {max_entries} files")
        if total_bytes > max_extracted_bytes:
            raise UnsafeArchiveError(
                f"ZIP expands beyond the {max_extracted_bytes / 1024**3:.1f} GiB safety limit"
            )
    if not selected:
        raise UnsafeArchiveError("ZIP contains no regular files")
    return selected


def extract_zip_safely(
    archive_path: Path,
    extraction_root: Path,
    spec: FileSpec,
    *,
    force: bool = False,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
) -> Path:
    validate_downloaded_file(archive_path, spec.size_bytes, spec.md5)
    extraction_root.mkdir(parents=True, exist_ok=True)
    target_name = Path(spec.name).stem
    target_dir = safe_child(extraction_root, target_name)
    stage_dir = safe_child(extraction_root, f"{target_name}.partial-extract")
    receipt_name = ".extraction_receipt.json"

    if target_dir.exists():
        receipt_path = target_dir / receipt_name
        try:
            with receipt_path.open("r", encoding="utf-8") as handle:
                receipt = json.load(handle)
            reusable = (
                receipt.get("archive_name") == spec.name
                and receipt.get("archive_size_bytes") == spec.size_bytes
                and receipt.get("archive_md5", "").casefold() == spec.md5
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            reusable = False
        if reusable:
            print(f"[extracted] reuse {target_dir}")
            return target_dir
        if not force:
            raise IntegrityError(
                f"Extraction directory has no matching receipt: {target_dir}. Use --force to replace it."
            )
        _safe_remove_tree(target_dir, extraction_root)

    if stage_dir.exists():
        if not force:
            raise IntegrityError(
                f"Incomplete extraction staging directory exists: {stage_dir}. Use --force to replace it."
            )
        _safe_remove_tree(stage_dir, extraction_root)
    stage_dir.mkdir()

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _preflight_zip(
                archive,
                stage_dir,
                expected_suffixes=spec.expected_suffixes,
                max_entries=max_entries,
                max_extracted_bytes=max_extracted_bytes,
            )
            total_bytes = 0
            for info, member in members:
                destination = stage_dir / Path(*member.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f"{destination.name}.partial")
                with archive.open(info, "r") as source, temporary.open("xb") as target:
                    copied = shutil.copyfileobj(source, target, length=DEFAULT_CHUNK_BYTES)
                    del copied  # copyfileobj returns None; make the intentional streaming explicit
                if temporary.stat().st_size != info.file_size:
                    raise IntegrityError(
                        f"Extracted size mismatch for {info.filename}: "
                        f"expected {info.file_size}, got {temporary.stat().st_size}"
                    )
                os.replace(temporary, destination)
                total_bytes += info.file_size

        receipt = {
            "schema_version": 1,
            "archive_name": spec.name,
            "archive_size_bytes": spec.size_bytes,
            "archive_md5": spec.md5,
            "expected_suffixes": list(spec.expected_suffixes),
            "member_count": len(members),
            "total_extracted_bytes": total_bytes,
            "extracted_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (stage_dir / receipt_name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        stage_dir.replace(target_dir)
    except Exception:
        _safe_remove_tree(stage_dir, extraction_root)
        raise

    print(f"[extracted] {target_dir}")
    return target_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", default="overt_word_raw_primary")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true", help="verify ZIPs without extraction")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace invalid/incomplete items, but only inside --destination",
    )
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    parser.add_argument("--max-extracted-gib", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0 or args.max_entries <= 0 or args.max_extracted_gib <= 0:
        raise SystemExit("Timeout and extraction limits must be positive")
    manifest = load_manifest(args.manifest)
    profile, file_specs = resolve_profile(manifest, args.profile)
    destination = args.destination.expanduser().resolve()
    repository_root = HERE.parent.resolve()
    if destination == repository_root or _is_within(destination, repository_root):
        raise ManifestError(
            f"Participant data destination must be outside the source repository: {destination}"
        )
    archives_dir = destination / "archives"
    extraction_root = destination / "extracted"
    print(f"Dataset: {manifest.dataset['short_name']} record {manifest.dataset['record_id']}")
    print(f"Profile: {profile.name} ({profile.total_download_bytes / 1024**2:.1f} MiB)")
    print(f"Destination: {destination}")

    for spec in file_specs:
        archive_path = download_file(
            spec,
            archives_dir,
            force=args.force,
            timeout_seconds=args.timeout_seconds,
        )
        if not args.download_only:
            extract_zip_safely(
                archive_path,
                extraction_root,
                spec,
                force=args.force,
                max_entries=args.max_entries,
                max_extracted_bytes=int(args.max_extracted_gib * 1024**3),
            )
    print("[done] Every selected archive matches the pinned manifest.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Verified archives remain intact; .partial downloads can resume.", file=sys.stderr)
        raise SystemExit(130)
    except (ManifestError, IntegrityError, UnsafeArchiveError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
