"""Small deterministic integrity helpers used by external-validation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject NaN/Infinity."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def atomic_write_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    """Write UTF-8 JSON through a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    try:
        temporary.write_text(rendered + "\n", encoding="utf-8", newline="\n")
        if not overwrite and path.exists():
            raise FileExistsError(f"Immutable artifact already exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
