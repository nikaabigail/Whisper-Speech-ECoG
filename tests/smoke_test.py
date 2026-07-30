#!/usr/bin/env python3
"""Offline release smoke checks that do not require participant data.

Checks syntax, selected imports, CLI help, curated metrics/figures, accidental
private paths or credentials, and files that should never enter the code repo.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Callable, Iterable
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
MAX_RELEASE_FILE_BYTES = 25 * 1024 * 1024
FORBIDDEN_DATA_SUFFIXES = {
    ".ckpt",
    ".csv",
    ".flac",
    ".eeg",
    ".edf",
    ".bdf",
    ".fif",
    ".h5",
    ".hdf5",
    ".joblib",
    ".mat",
    ".npy",
    ".nwb",
    ".npz",
    ".set",
    ".vhdr",
    ".vmrk",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".textgrid",
    ".tsv",
    ".wav",
    ".xdf",
    ".zip",
}
ALLOWED_CURATED_TABLES = {
    "05_external_validation/swpd_matched_pca50/table_01_system_performance.csv",
    "05_external_validation/swpd_matched_pca50/table_02_whisper_vs_mel_contrasts.csv",
    "05_external_validation/swpd_matched_pca50/table_03_patient_level_metrics.csv",
    "05_external_validation/swpd_contextual_frozen/results/frozen_subject_metrics.csv",
    "05_external_validation/swpd_contextual_neural_population/results/authors_figure4a_digitized.csv",
    "results/async/paper_style_async_pr_multiseed.csv",
    "results/reported_metrics.csv",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".gitignore",
    ".gitattributes",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def release_files() -> Iterable[Path]:
    """Yield only files Git would publish, with a filesystem fallback."""

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        completed = None
    if completed is not None and completed.returncode == 0:
        for raw_path in completed.stdout.split(b"\0"):
            if not raw_path:
                continue
            path = ROOT / os.fsdecode(raw_path)
            if path.is_file():
                yield path
        return

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_markdown_links() -> None:
    """Require every local Markdown link/image target to exist inside the repo."""
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    findings: list[str] = []
    for markdown in (path for path in release_files() if path.suffix.lower() == ".md"):
        text = markdown.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<") and raw_target.endswith(">"):
                raw_target = raw_target[1:-1]
            if not raw_target or raw_target.startswith("#"):
                continue
            if raw_target.lower().startswith(("http://", "https://", "mailto:")):
                continue
            local_target = unquote(raw_target.split("#", 1)[0])
            resolved = (markdown.parent / local_target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                findings.append(
                    f"{relative(markdown)}: target escapes repository: {raw_target}"
                )
                continue
            if not resolved.exists():
                findings.append(
                    f"{relative(markdown)}: missing target: {raw_target}"
                )
    if findings:
        raise AssertionError("Markdown link scan failed:\n" + "\n".join(findings))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_python_syntax() -> None:
    python_files = sorted(path for path in release_files() if path.suffix == ".py")
    if not python_files:
        raise AssertionError("No Python files found")
    failures = []
    for path in python_files:
        try:
            ast.parse(read_text(path), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{relative(path)}:{exc.lineno}: {exc.msg}")
    if failures:
        raise AssertionError("Python syntax failures:\n" + "\n".join(failures))


def check_metrics() -> None:
    json_path = ROOT / "results" / "reported_metrics.json"
    csv_path = ROOT / "results" / "reported_metrics.csv"
    with json_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    sync = metrics["synchronous"]["single_seed_accuracy"]
    if sync["L3+L4+L5"] != 0.8135 or max(sync, key=sync.get) != "L3+L4+L5":
        raise AssertionError("Unexpected synchronous summary")
    chance = metrics["synchronous"]["chance_accuracy"]
    if abs(chance - 1.0 / 27.0) > 5e-4:
        raise AssertionError("Synchronous chance level is inconsistent with 27 classes")
    async_models = metrics["asynchronous"]["continuous_head_multiseed"]["f1_by_model"]
    ensemble = async_models["L3+L4+L5"]
    if not ensemble["ci95"][0] < ensemble["mean"] < ensemble["ci95"][1]:
        raise AssertionError("Asynchronous ensemble mean is outside its CI")
    if not all(ensemble["mean"] > async_models[layer]["mean"] for layer in ("L3", "L4", "L5")):
        raise AssertionError("Asynchronous ensemble should exceed every reported single layer")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12:
        raise AssertionError(f"Expected 12 curated CSV rows, found {len(rows)}")


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise AssertionError(f"Not a valid PNG header: {relative(path)}")
    return struct.unpack(">II", data[16:24])


def check_figures() -> None:
    with (ROOT / "results" / "reported_metrics.json").open(encoding="utf-8") as handle:
        figure_map = json.load(handle)["figures"]
    for name, rel_path in figure_map.items():
        path = ROOT / "results" / rel_path
        if not path.is_file() or path.stat().st_size < 10_000:
            raise AssertionError(f"Missing or unexpectedly small figure {name}: {rel_path}")
        width, height = png_dimensions(path)
        if width < 1000 or height < 600:
            raise AssertionError(f"Figure is too small for publication preview: {rel_path} ({width}x{height})")


def check_no_private_paths_or_secrets() -> None:
    backslash = "\\"
    slash = "/"
    private_literals = (
        "C:" + backslash + "Users" + backslash,
        "C:" + slash + "Users" + slash,
        "file:" + slash * 3 + "C:" + slash + "Users" + slash,
        "D:" + backslash + "Osadchi",
        "D:" + slash + "Osadchi",
        "C:" + backslash + "ossadtchi",
        "C:" + slash + "ossadtchi",
    )
    private_patterns = {
        "Windows user profile path": re.compile(
            r"(?i)\b[A-Z]:[\\/]Users[\\/][^<>:\"/\\|?*\s]+"
        ),
        "credential-bearing URL": re.compile(
            r"(?i)https?://[^\s/@:]+:[^\s/@]+@[^\s/]+"
        ),
    }
    secret_patterns = {
        "AWS access-key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
        "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "GitLab token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
        "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
        "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "literal credential assignment": re.compile(
            r"(?i)\b(?:password|passwd|api[_-]?key|client[_-]?secret)\s*[:=]\s*"
            r"[\"'][^\"'\r\n]{8,}[\"']"
        ),
        "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    }
    findings = []
    for path in release_files():
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
            continue
        text = read_text(path)
        for literal in private_literals:
            if literal.casefold() in text.casefold():
                findings.append(f"{relative(path)}: private absolute path")
                break
        for label, pattern in private_patterns.items():
            if pattern.search(text):
                findings.append(f"{relative(path)}: possible {label}")
        for label, pattern in secret_patterns.items():
            if pattern.search(text):
                findings.append(f"{relative(path)}: possible {label}")
    if findings:
        raise AssertionError("Sensitive material scan failed:\n" + "\n".join(sorted(set(findings))))


def check_repository_payload() -> None:
    findings = []
    for path in release_files():
        rel = relative(path)
        if path.stat().st_size > MAX_RELEASE_FILE_BYTES:
            findings.append(f"{rel}: {path.stat().st_size / 1024**2:.1f} MiB exceeds 25 MiB")
        if (
            path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES
            and rel not in ALLOWED_CURATED_TABLES
        ):
            findings.append(f"{rel}: model/data artifact extension {path.suffix}")
        if path.name.casefold() == "patients.json":
            findings.append(f"{rel}: private patient configuration name")
    if findings:
        raise AssertionError("Repository payload scan failed:\n" + "\n".join(findings))


def run_python(cwd: Path, args: list[str], label: str, timeout: int = 90) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(
        [sys.executable, "-B", *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-25:])
        raise AssertionError(f"{label} failed with exit {completed.returncode}:\n{tail}")


def check_imports() -> None:
    run_python(
        ROOT / "02_whisper_sync",
        [
            "-c",
            "import library.runtime, library.models_regression, "
            "library.models_classification, library.whisper_target",
        ],
        "synchronous-library import",
    )
    run_python(
        ROOT / "03_whisper_ensemble",
        ["-c", "import ensemble_layers"],
        "ensemble import",
    )
    run_python(
        ROOT / "04_whisper_async",
        [
            "-c",
            "import async_replay, continuous_common, build_continuous_cache, "
            "train_continuous_heads, summarize_continuous_multiseed",
        ],
        "asynchronous-library import",
    )
    run_python(
        ROOT / "04_whisper_async" / "paper_pr_curve",
        ["-c", "import build_paper_style_pr, summarize_recall40"],
        "reporting-library import",
    )


def check_model_contracts() -> None:
    code = (
        "import torch; "
        "from library.models_regression import SimpleNet; "
        "from library.models_classification import Mel2WordHidden; "
        "encoder=SimpleNet(8,50,1000,0).eval(); "
        "assert encoder.final_out_features==3030; "
        "x=torch.zeros(2,8,1001); "
        "assert tuple(encoder(x,return_hidden=True).shape)==(2,3030); "
        "assert tuple(encoder(x).shape)==(2,50); "
        "head=Mel2WordHidden(3030,27).eval(); "
        "assert tuple(head(torch.zeros(2,3030,52)).shape)==(2,27)"
    )
    run_python(
        ROOT / "02_whisper_sync",
        ["-c", code],
        "3030D/50D/27-class model contracts",
    )


def check_cli_help() -> None:
    entries = (
        ("02_whisper_sync", "train_sync.py"),
        ("02_whisper_sync", "multiseed.py"),
        ("03_whisper_ensemble", "ensemble_layers.py"),
        ("04_whisper_async", "async_replay.py"),
        ("04_whisper_async", "build_continuous_cache.py"),
        ("04_whisper_async", "train_continuous_heads.py"),
        ("04_whisper_async", "summarize_continuous_multiseed.py"),
        ("04_whisper_async/paper_pr_curve", "build_paper_style_pr.py"),
        ("04_whisper_async/paper_pr_curve", "summarize_recall40.py"),
        ("results", "plot_reported_metrics.py"),
    )
    for directory, script in entries:
        run_python(ROOT / directory, [script, "--help"], f"{directory}/{script} --help")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-imports", action="store_true")
    parser.add_argument("--skip-help", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[tuple[str, Callable[[], None]]] = [
        ("Python AST", check_python_syntax),
        ("reported metrics", check_metrics),
        ("PNG figures", check_figures),
        ("Markdown links", check_markdown_links),
        ("private paths and secrets", check_no_private_paths_or_secrets),
        ("large/private payloads", check_repository_payload),
    ]
    if not args.skip_imports:
        checks.append(("selected imports", check_imports))
        checks.append(("model shape contracts", check_model_contracts))
    if not args.skip_help:
        checks.append(("CLI --help", check_cli_help))

    failures = []
    for label, check in checks:
        try:
            check()
        except Exception as exc:  # report every independent release check
            failures.append((label, exc))
            print(f"[FAIL] {label}: {exc}")
        else:
            print(f"[ OK ] {label}")
    if failures:
        print(f"\nRelease smoke test failed: {len(failures)} check(s).")
        return 1
    print(f"\nRelease smoke test passed: {len(checks)} check(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
