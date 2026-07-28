"""Bind caches and checkpoints to the executable source and runtime."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess
import sys
from typing import Any

from .integrity import fingerprint_json, sha256_file


RUNTIME_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "h5py",
    "pynwb",
    "librosa",
    "soundfile",
    "transformers",
)


def _source_files(root: Path) -> tuple[Path, ...]:
    patterns = (
        "*.py",
        "pyproject.toml",
        "src/**/*.py",
        "configs/**/*.json",
        "manifests/**/*.json",
        "requirements/*.txt",
        "scripts/*.ps1",
    )
    selected: set[Path] = set()
    for pattern in patterns:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    if not selected:
        raise RuntimeError(f"no external-validation source files found below {root}")
    return tuple(sorted(selected, key=lambda path: path.relative_to(root).as_posix()))


def _git_receipt(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    try:
        commit = run("rev-parse", "HEAD")
    except OSError:
        return {"available": False, "commit": None, "dirty": None, "status": []}
    if commit.returncode != 0:
        return {"available": False, "commit": None, "dirty": None, "status": []}
    status = run("status", "--porcelain=v1", "--untracked-files=all", "--", ".")
    if status.returncode != 0:
        raise RuntimeError(f"git status failed while capturing source identity: {status.stderr}")
    lines = tuple(line for line in status.stdout.splitlines() if line.strip())
    return {
        "available": True,
        "commit": commit.stdout.strip().lower(),
        "dirty": bool(lines),
        "status": list(lines),
    }


def capture_source_identity(root: Path | None = None) -> dict[str, Any]:
    """Return a deterministic receipt for every executable protocol file.

    The content hash protects resumable artifacts even during development, while
    the Git receipt lets confirmatory runners require one clean frozen commit.
    """

    source_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    files = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in _source_files(source_root)
    }
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip().lower().replace("_", "-")
        if name:
            packages[name] = str(distribution.version)
    packages = dict(sorted(packages.items()))
    required_packages = {
        name: packages.get(name.lower().replace("_", "-"), "MISSING")
        for name in RUNTIME_DISTRIBUTIONS
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "external_validation_source_identity",
        "root_name": source_root.name,
        "files": files,
        "files_fingerprint": fingerprint_json(files),
        "git": _git_receipt(source_root),
        "python": sys.version,
        "runtime_distributions": packages,
        "required_runtime_distributions": required_packages,
    }
    receipt["fingerprint"] = fingerprint_json(receipt)
    return receipt


def require_clean_frozen_source(receipt: dict[str, Any]) -> None:
    git = receipt.get("git")
    if not isinstance(git, dict) or not git.get("available") or not git.get("commit"):
        raise RuntimeError("confirmatory production requires a Git checkout with a commit")
    if git.get("dirty"):
        raise RuntimeError(
            "confirmatory production requires a clean source checkout; commit or remove "
            "the listed changes before starting"
        )
    missing = [
        name
        for name, version in dict(receipt.get("required_runtime_distributions", {})).items()
        if version == "MISSING"
    ]
    if missing:
        raise RuntimeError(f"confirmatory runtime is missing packages: {', '.join(missing)}")
