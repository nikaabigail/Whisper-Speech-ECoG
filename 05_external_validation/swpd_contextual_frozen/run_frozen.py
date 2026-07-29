#!/usr/bin/env python3
"""Run the frozen contextual MEL80 versus Whisper-L4 comparison on sub-02..09."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


MODULE_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(EXTERNAL_ROOT))
sys.path.insert(0, str(EXTERNAL_ROOT / "src"))

from swpd_contextual_frozen.core import build_or_load_block, evaluate_subject  # noqa: E402
from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
    fingerprint_json,
    read_json,
    sha256_file,
)
from whisper_ecog_ext.swpd.matched_linear import make_visual_blocks  # noqa: E402
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    NWBLayoutError,
    SWPDRecording,
    inventory_subject,
    load_visual_word_events_subject,
    recording_duration_seconds,
)
from whisper_ecog_ext.targets import MelTargetExtractor, WhisperLayerTargetExtractor  # noqa: E402


DEFAULT_PROTOCOL = EXTERNAL_ROOT / "configs" / "experiments" / "swpd_contextual_l4_frozen_v1.json"
DEFAULT_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
SUBJECTS = tuple(f"sub-{number:02d}" for number in range(2, 10))
EXCLUDED = ("sub-10",)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _external(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if result == EXTERNAL_ROOT or _inside(result, EXTERNAL_ROOT):
        raise ValueError(f"{label} must be outside the source checkout")
    return result


def validate_protocol(payload: Mapping[str, Any], protocol_path: Path) -> None:
    if payload.get("schema_version") != 1 or payload.get("status") != "frozen_contextual_extension_after_sub01_development":
        raise ValueError("Protocol is not the frozen contextual extension")
    if tuple(payload.get("primary_confirmatory_subjects", ())) != SUBJECTS:
        raise ValueError("Primary cohort must remain sub-02 through sub-09")
    if tuple(payload.get("excluded_subjects", ())) != EXCLUDED or payload.get("development_subject") != "sub-01":
        raise ValueError("Development/exclusion cohort changed")
    extraction = payload.get("extraction", {})
    expected_extraction = {
        "high_gamma_hz": [70, 170], "notch_hz": [[98, 102], [148, 152]],
        "window_ms": 50, "base_grid_ms": 10,
        "neural_context_ms": [-200, -150, -100, -50, 0, 50, 100, 150, 200],
        "output_grid_ms": 20, "edge_guard_ms": 1000,
        "whisper_model": "openai/whisper-base", "whisper_revision": DEFAULT_REVISION,
        "whisper_layer": 4, "mel_bins": 80,
    }
    if extraction != expected_extraction:
        raise ValueError("Frozen extraction settings changed")
    comparison = payload.get("comparison", {})
    expected = {
        "systems": ["direct_mel80", "whisper_l4_pca50"],
        "neural_transform": "fold-train StandardScaler plus PCA50, whiten false",
        "l4_transform": "fold-train StandardScaler plus PCA50, whiten true",
        "decoder": "ordinary least squares",
        "splits": "five visual blocks; test i, validation i+1 cyclic, remaining three train",
        "primary_metric": "subject mean of fold mean Pearson r on common standardized MEL80",
        "secondary_metric": "same metric on lower 20 MEL bins",
        "primary_contrast": "whisper_l4_pca50 minus direct_mel80",
    }
    if comparison != expected:
        raise ValueError("Frozen comparison settings changed")
    if payload.get("population_inference", {}).get("n") != len(SUBJECTS):
        raise ValueError("Frozen population size changed")
    if payload.get("fixed_before_contextual_confirmatory_extraction") is not True:
        raise ValueError("Protocol is not marked fixed before extraction")
    for key in ("sub10_qc_amendment", "development_result"):
        reference = payload[key]
        path = Path(reference["path"])
        if not path.is_absolute():
            path = EXTERNAL_ROOT / path
        path = path.resolve()
        if path.is_file():
            if sha256_file(path) != reference["sha256"]:
                raise RuntimeError(f"Frozen reference changed: {key}")
            continue
        if key == "development_result":
            portable_gate = read_json(MODULE_ROOT / "results" / "development_gate.json")
            if (
                portable_gate.get("source_summary_sha256") == reference["sha256"]
                and portable_gate.get("confirmatory_subjects_read") is False
                and portable_gate.get("selected_system") == reference["selected_system"]
                and portable_gate.get("direct_mel80_r") == reference["direct_mel80_r"]
                and portable_gate.get("l4_r") == reference["l4_r"]
            ):
                continue
        raise RuntimeError(f"Frozen reference missing: {key}")


def load_protocol(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = read_json(resolved)
    validate_protocol(payload, resolved)
    return payload


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _describe(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    n = int(data.size)
    mean = float(data.mean())
    sd = float(data.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n else float("nan")
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "sem": float(sem),
            "ci95_t": [float(mean - critical * sem), float(mean + critical * sem)]}


def _receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "fingerprint"}


def load_completions(run_root: Path, contract_fingerprint: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        path = run_root / "subjects" / subject / "subject_complete.json"
        if not path.is_file():
            continue
        receipt = read_json(path)
        if receipt.get("fingerprint") != fingerprint_json(_receipt_payload(receipt)):
            raise RuntimeError(f"Invalid completion fingerprint: {subject}")
        if receipt.get("run_contract_fingerprint") != contract_fingerprint:
            raise RuntimeError(f"Completion belongs to another frozen contract: {subject}")
        for path_key, hash_key in (("summary_path", "summary_sha256"), ("predictions_path", "predictions_sha256")):
            artifact = Path(receipt[path_key])
            if not artifact.is_file() or sha256_file(artifact) != receipt[hash_key]:
                raise RuntimeError(f"Completed artifact changed: {subject} {path_key}")
        result[subject] = receipt
    return result


def write_population(run_root: Path, completions: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]) -> Path:
    rows = []
    for subject in SUBJECTS:
        if subject not in completions:
            continue
        summary = read_json(Path(completions[subject]["summary_path"]))
        direct = float(summary["aggregate"]["direct_mel80"]["all_bins"]["mean"])
        l4 = float(summary["aggregate"]["whisper_l4_pca50"]["all_bins"]["mean"])
        rows.append({
            "subject": subject, "direct_mel80_r": direct, "whisper_l4_pca50_r": l4,
            "delta_l4_minus_mel80": l4 - direct,
            "direct_mel80_low20_r": float(summary["aggregate"]["direct_mel80"]["lower_20_bins"]["mean"]),
            "whisper_l4_pca50_low20_r": float(summary["aggregate"]["whisper_l4_pca50"]["lower_20_bins"]["mean"]),
        })
    deltas = [row["delta_l4_minus_mel80"] for row in rows]
    inference: dict[str, Any] = {"available": len(rows) == len(SUBJECTS)}
    if rows:
        inference.update({
            "direct_mel80": _describe([row["direct_mel80_r"] for row in rows]),
            "whisper_l4_pca50": _describe([row["whisper_l4_pca50_r"] for row in rows]),
            "delta_l4_minus_mel80": _describe(deltas),
            "wins": int(sum(value > 0 for value in deltas)),
        })
    if len(rows) == len(SUBJECTS):
        inference["two_sided_paired_t_p"] = float(stats.ttest_1samp(deltas, 0.0).pvalue)
        inference["two_sided_exact_sign_p"] = float(stats.binomtest(sum(value > 0 for value in deltas), len(deltas), 0.5).pvalue)
    payload = {
        "schema_version": 1, "kind": "swpd_contextual_l4_frozen_population_summary",
        "updated_utc": _now(), "primary_subjects": list(SUBJECTS),
        "completed_subjects": [row["subject"] for row in rows],
        "missing_subjects": [subject for subject in SUBJECTS if subject not in completions],
        "development_subject_excluded": "sub-01", "excluded_by_frozen_qc": list(EXCLUDED),
        "historical_access_disclosure": protocol["historical_access_disclosure"],
        "subject_rows": rows, "primary_inference": inference,
    }
    path = run_root / "summary" / "population_summary.json"
    atomic_write_json(path, payload)
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--whisper-revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--channel-batch-size", type=int, default=16)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol_config.expanduser().resolve()
    protocol = load_protocol(protocol_path)
    if args.whisper_revision != DEFAULT_REVISION:
        raise ValueError("CLI Whisper revision differs from frozen protocol")
    if args.channel_batch_size < 1:
        raise ValueError("channel-batch-size must be positive")
    cache_root = _external(args.cache_root, "cache-root")
    run_root = _external(args.run_root, "run-root")
    if args.plan_only:
        print(f"PLAN | frozen contextual cohort={list(SUBJECTS)} | excluded={list(EXCLUDED)}")
        print("PLAN | systems=direct MEL80, Whisper L4 train-only PCA50 | no dataset opened")
        return 0

    # This immutable contract is written before any confirmatory inventory, events or NWB are read.
    run_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1, "kind": "swpd_contextual_l4_frozen_run_contract",
        "protocol_path": str(protocol_path), "protocol_sha256": sha256_file(protocol_path),
        "protocol": protocol, "subjects": list(SUBJECTS), "excluded_subjects": list(EXCLUDED),
        "device": args.device, "channel_batch_size": args.channel_batch_size,
        "whisper_revision": args.whisper_revision,
        "source_sha256": {
            "runner": sha256_file(Path(__file__)), "core": sha256_file(MODULE_ROOT / "core.py"),
            "nwb": sha256_file(EXTERNAL_ROOT / "src" / "whisper_ecog_ext" / "swpd" / "nwb.py"),
            "author_mel": sha256_file(EXTERNAL_ROOT / "src" / "whisper_ecog_ext" / "swpd" / "author_mel.py"),
            "targets": sha256_file(EXTERNAL_ROOT / "src" / "whisper_ecog_ext" / "targets.py"),
        },
    }
    contract["fingerprint"] = fingerprint_json(contract)
    contract_path = run_root / "run_contract.json"
    if contract_path.is_file():
        if read_json(contract_path) != contract:
            raise RuntimeError("Existing run contract differs from current frozen source/protocol")
    else:
        atomic_write_json(contract_path, contract, overwrite=False)
    contract_fp = str(contract["fingerprint"])

    mel = MelTargetExtractor(n_mels=80, frame_hz=50.0)
    whisper: WhisperLayerTargetExtractor | None = None
    completions = load_completions(run_root, contract_fp)
    for subject in SUBJECTS:
        if subject in completions:
            print(f"[skip validated] {subject}", flush=True)
            continue
        atomic_write_json(run_root / "queue_state.json", {
            "status": "running", "current_subject": subject,
            "completed_subjects": [item for item in SUBJECTS if item in completions],
            "remaining_subjects": [item for item in SUBJECTS if item not in completions], "updated_utc": _now(),
        })
        print(f"===== {subject} | frozen contextual MEL80 vs L4 PCA50 =====", flush=True)
        inventory = inventory_subject(args.data_root, subject, allow_confirmatory=True)
        events = load_visual_word_events_subject(args.data_root, subject, allow_confirmatory=True)
        definitions = make_visual_blocks(events, recording_duration_seconds(inventory))
        if whisper is None:
            whisper = WhisperLayerTargetExtractor(revision=args.whisper_revision, device=args.device)
        blocks = []
        block_contracts = []
        with SWPDRecording(args.data_root, subject, allow_confirmatory=True) as recording:
            for definition in definitions:
                block, block_contract = build_or_load_block(
                    recording, subject, inventory, definition, cache_root / subject,
                    mel, whisper, args.channel_batch_size,
                )
                blocks.append(block)
                block_contracts.append(fingerprint_json(block_contract))
                print(f"[{subject}] block {definition.index}: {len(block.sample_ids)} frames ready", flush=True)
        summary, predictions = evaluate_subject(blocks, subject)
        subject_root = run_root / "subjects" / subject
        subject_root.mkdir(parents=True, exist_ok=True)
        summary.update({
            "schema_version": 1, "kind": "swpd_contextual_l4_frozen_subject_summary",
            "run_contract_fingerprint": contract_fp, "block_contract_fingerprints": block_contracts,
            "inventory": inventory.to_dict(), "created_utc": _now(),
        })
        predictions_path = subject_root / "predictions.npz"
        summary_path = subject_root / "summary.json"
        _atomic_npz(predictions_path, predictions)
        atomic_write_json(summary_path, summary, overwrite=False)
        receipt: dict[str, Any] = {
            "schema_version": 1, "kind": "swpd_contextual_l4_frozen_subject_completion",
            "subject": subject, "run_contract_fingerprint": contract_fp,
            "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
            "predictions_path": str(predictions_path), "predictions_sha256": sha256_file(predictions_path),
            "completed_utc": _now(),
        }
        receipt["fingerprint"] = fingerprint_json(receipt)
        atomic_write_json(subject_root / "subject_complete.json", receipt, overwrite=False)
        completions[subject] = receipt
        population = write_population(run_root, completions, protocol)
        print(f"[completed] {subject} | delta={summary['delta_l4_minus_mel80']:+.4f} | {population}", flush=True)

    population = write_population(run_root, completions, protocol)
    atomic_write_json(run_root / "queue_state.json", {
        "status": "completed", "current_subject": None, "completed_subjects": list(SUBJECTS),
        "remaining_subjects": [], "population_summary": str(population), "updated_utc": _now(),
    })
    print(f"ALL FROZEN SUBJECTS COMPLETE | {population}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NWBLayoutError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
