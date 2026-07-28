#!/usr/bin/env python3
"""Run the frozen matched PCA50 comparison sequentially for all SWPD subjects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
    fingerprint_json,
    read_json,
    sha256_file,
)
from whisper_ecog_ext.swpd.matched_linear import (  # noqa: E402
    AUTHOR_AUDIO_PROCESSING_RATE,
    BLOCK_COUNT,
    EDGE_GUARD_SECONDS,
    FRAME_SHIFT_SECONDS,
    REDUCED_DIMENSION,
    TARGET_NAMES,
    TRIALS_PER_BLOCK,
    build_extraction_fingerprint,
    extract_one_block,
    load_block_cache,
    make_visual_blocks,
    run_matched_folds,
    save_block_cache,
)
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    ALL_SUBJECTS,
    LOCKED_CONFIRMATORY_SUBJECTS,
    NWBLayoutError,
    PILOT_SUBJECT,
    SWPDRecording,
    inventory_subject,
    load_visual_word_events_subject,
    recording_duration_seconds,
    subject_paths_frozen,
)
from whisper_ecog_ext.targets import (  # noqa: E402
    MelTargetExtractor,
    WhisperLayerTargetExtractor,
)


DEFAULT_WHISPER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
DEFAULT_PROTOCOL = HERE / "configs" / "experiments" / "swpd_all_matched_pca50_v1.json"
PINNED_DATASET_MANIFEST = HERE / "manifests" / "swpd_osf_nrgx6.json"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _external_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == HERE or _inside(resolved, HERE):
        raise ValueError(f"{label} must be outside the source checkout")
    return resolved


def validate_production_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Production protocol schema_version must be 1")
    if payload.get("status") != "frozen_confirmatory_after_sub01_development":
        raise ValueError("Production protocol is not frozen for confirmatory access")
    if tuple(payload.get("all_subjects", ())) != ALL_SUBJECTS:
        raise ValueError("Production protocol must contain SWPD sub-01 through sub-10")
    if tuple(payload.get("development_subjects", ())) != (PILOT_SUBJECT,):
        raise ValueError("sub-01 must remain the sole development subject")
    if tuple(payload.get("primary_confirmatory_subjects", ())) != LOCKED_CONFIRMATORY_SUBJECTS:
        raise ValueError("Primary confirmatory cohort must be sub-02 through sub-10")
    extraction = payload.get("target_extraction")
    if not isinstance(extraction, Mapping) or (
        extraction.get("whisper_model"),
        extraction.get("whisper_revision"),
        extraction.get("whisper_layers"),
        extraction.get("whisper_chunk_seconds"),
        extraction.get("mel_bins"),
        extraction.get("target_frame_hz"),
        extraction.get("audio_processing_rate_hz"),
        extraction.get("reducer_seed"),
    ) != (
        "openai/whisper-base",
        DEFAULT_WHISPER_REVISION,
        [3, 4, 5],
        30,
        80,
        50,
        AUTHOR_AUDIO_PROCESSING_RATE,
        42,
    ):
        raise ValueError("Frozen target extraction configuration changed")
    comparison = payload.get("matched_comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("Production protocol has no matched_comparison")
    expected = {
        "targets": list(TARGET_NAMES),
        "common_target_dimension": REDUCED_DIMENSION,
        "frame_grid_ms": int(round(FRAME_SHIFT_SECONDS * 1000)),
        "high_gamma_window_ms": 50,
        "ensemble_status": "not_defined_because_layer_specific_pca_bases_are_not_aligned",
    }
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise ValueError(f"Frozen production field changed: {key}")
    target_transform = comparison.get("target_transform")
    neural_transform = comparison.get("neural_transform")
    decoder = comparison.get("decoder")
    splits = comparison.get("splits")
    if not isinstance(target_transform, Mapping) or (
        target_transform.get("fit_scope"),
        target_transform.get("standardize"),
        target_transform.get("reducer"),
        target_transform.get("whiten"),
    ) != ("outer_train_only", True, "PCA", True):
        raise ValueError("Targets must use train-only standardized PCA whitening")
    if not isinstance(neural_transform, Mapping) or (
        neural_transform.get("shared_across_targets_within_subject_fold"),
        neural_transform.get("fit_scope"),
        neural_transform.get("components"),
        neural_transform.get("whiten"),
    ) != (True, "outer_train_only", REDUCED_DIMENSION, True):
        raise ValueError("Neural transform must be shared train-only PCA50 whitening")
    if not isinstance(decoder, Mapping) or (
        decoder.get("kind"), decoder.get("hyperparameters_identical_across_targets")
    ) != ("ordinary_least_squares", True):
        raise ValueError("All targets must use identical ordinary least squares")
    if not isinstance(splits, Mapping) or (
        splits.get("block_count"),
        splits.get("trials_per_block"),
        splits.get("visual_events_define_blocks_only"),
        splits.get("visual_events_are_not_acoustic_onsets"),
    ) != (BLOCK_COUNT, TRIALS_PER_BLOCK, True, True):
        raise ValueError("Split contract changed")
    population = payload.get("population_inference")
    if not isinstance(population, Mapping) or (
        population.get("unit"),
        population.get("primary_n"),
        population.get("exclude_development_subject_from_primary_inference"),
    ) != ("patient", 9, True):
        raise ValueError("Population inference must use confirmatory patients sub-02..sub-10")
    baseline = payload.get("author_mel_baseline")
    if not isinstance(baseline, Mapping) or baseline.get("included_in_matched_fit") is not False:
        raise ValueError("Author baseline must remain a separate reproduction control")


def load_production_protocol(path: Path, *, verify_baseline: bool = True) -> dict[str, Any]:
    payload = read_json(path.expanduser().resolve())
    validate_production_protocol(payload)
    if verify_baseline:
        baseline = payload["author_mel_baseline"]
        baseline_path = Path(str(baseline["path"])).expanduser().resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Frozen author baseline is missing: {baseline_path}")
        if sha256_file(baseline_path) != baseline["sha256"]:
            raise RuntimeError("Frozen author baseline checksum changed")
    return payload


def _metric(summary: Mapping[str, Any], target: str, metric: str) -> float:
    return float(summary["aggregate_test"][target]["test_all"][metric]["mean"])


def _describe(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    n = int(data.size)
    if n == 0:
        return {"mean": None, "sd": None, "sem": None, "ci95_t": None, "n": 0}
    mean = float(np.mean(data))
    sd = float(np.std(data, ddof=1)) if n > 1 else 0.0
    sem = float(sd / np.sqrt(n)) if n else float("nan")
    if n > 1:
        critical = float(stats.t.ppf(0.975, df=n - 1))
        ci = [float(mean - critical * sem), float(mean + critical * sem)]
    else:
        ci = [mean, mean]
    return {"mean": mean, "sd": sd, "sem": sem, "ci95_t": ci, "n": n}


def aggregate_subject_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    cohort: Sequence[str],
) -> dict[str, Any]:
    subjects = [subject for subject in cohort if subject in summaries]
    rows = []
    for subject in subjects:
        summary = summaries[subject]
        row: dict[str, Any] = {"subject": subject}
        for target in TARGET_NAMES:
            row[target] = {
                "fisher_r": _metric(summary, target, "fisher_z_component_correlation"),
                "standardized_mse": _metric(summary, target, "standardized_mse"),
            }
        rows.append(row)
    systems: dict[str, Any] = {}
    for target in TARGET_NAMES:
        systems[target] = {
            "fisher_r": _describe([row[target]["fisher_r"] for row in rows]),
            "standardized_mse": _describe(
                [row[target]["standardized_mse"] for row in rows]
            ),
        }
    contrasts: dict[str, Any] = {}
    raw_p_values: dict[str, float | None] = {}
    for target in ("L3", "L4", "L5"):
        differences = [
            row[target]["fisher_r"] - row["mel80"]["fisher_r"] for row in rows
        ]
        described = _describe(differences)
        described["wins"] = int(sum(value > 0 for value in differences))
        raw_p = None
        if len(differences) > 1:
            candidate = float(stats.ttest_1samp(differences, popmean=0.0).pvalue)
            if np.isfinite(candidate):
                raw_p = candidate
        name = f"{target}_minus_mel80"
        described["two_sided_paired_t_p_raw"] = raw_p
        described["two_sided_paired_t_p_holm"] = None
        contrasts[name] = described
        raw_p_values[name] = raw_p
    finite = sorted(
        ((name, value) for name, value in raw_p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    total = len(finite)
    for rank, (name, value) in enumerate(finite):
        adjusted = min(1.0, (total - rank) * value)
        running = max(running, adjusted)
        contrasts[name]["two_sided_paired_t_p_holm"] = running
    return {"subjects": subjects, "subject_rows": rows, "systems": systems, "contrasts": contrasts}


def _next_attempt(subject_root: Path) -> Path:
    attempts = subject_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in attempts.glob("attempt_*"))
    return attempts / f"attempt_{len(existing) + 1:03d}"


def _load_completions(run_root: Path, protocol_sha256: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for subject in ALL_SUBJECTS:
        receipt_path = run_root / "subjects" / subject / "subject_complete.json"
        if not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path)
        if receipt.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Completion protocol mismatch for {subject}")
        summary_path = Path(receipt["summary_path"])
        if not summary_path.is_file() or sha256_file(summary_path) != receipt.get("summary_sha256"):
            raise RuntimeError(f"Completion artifact mismatch for {subject}")
        completed[subject] = receipt
    return completed


def _write_aggregate(run_root: Path, completions: Mapping[str, Mapping[str, Any]]) -> Path:
    summaries = {
        subject: read_json(Path(receipt["summary_path"]))
        for subject, receipt in completions.items()
    }
    payload = {
        "schema_version": 1,
        "kind": "swpd_matched_pca50_all_subjects_summary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_subjects": [subject for subject in ALL_SUBJECTS if subject in summaries],
        "missing_subjects": [subject for subject in ALL_SUBJECTS if subject not in summaries],
        "primary_confirmatory": aggregate_subject_summaries(
            summaries, cohort=LOCKED_CONFIRMATORY_SUBJECTS
        ),
        "all_ten_secondary_descriptive": aggregate_subject_summaries(
            summaries, cohort=ALL_SUBJECTS
        ),
        "development_subject_excluded_from_primary": PILOT_SUBJECT,
    }
    path = run_root / "summary" / "swpd_matched_pca50_all_subjects_summary.json"
    atomic_write_json(path, payload)
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--whisper-revision", default=DEFAULT_WHISPER_REVISION)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--reducer-seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol_config.expanduser().resolve()
    protocol = load_production_protocol(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    extraction = protocol["target_extraction"]
    if args.whisper_revision != extraction["whisper_revision"]:
        raise ValueError("CLI Whisper revision differs from the frozen protocol")
    if args.reducer_seed != extraction["reducer_seed"]:
        raise ValueError("CLI reducer seed differs from the frozen protocol")
    cache_root = _external_directory(args.cache_root, "cache-root")
    run_root = _external_directory(args.run_root, "run-root")
    if args.plan_only:
        print(f"PLAN | subjects={list(ALL_SUBJECTS)} | protocol={protocol_sha256}")
        print("PLAN | primary population inference excludes development subject sub-01")
        return 0
    cache_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    root_contract = {
        "schema_version": 1,
        "kind": "swpd_all_matched_pca50_run_contract",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "dataset_manifest": str(PINNED_DATASET_MANIFEST),
        "dataset_manifest_sha256": sha256_file(PINNED_DATASET_MANIFEST),
        "runner_sha256": sha256_file(Path(__file__)),
        "matched_implementation_sha256": sha256_file(
            HERE / "src" / "whisper_ecog_ext" / "swpd" / "matched_linear.py"
        ),
        "nwb_adapter_sha256": sha256_file(
            HERE / "src" / "whisper_ecog_ext" / "swpd" / "nwb.py"
        ),
        "author_mel_implementation_sha256": sha256_file(
            HERE / "src" / "whisper_ecog_ext" / "swpd" / "author_mel.py"
        ),
        "target_extractor_sha256": sha256_file(
            HERE / "src" / "whisper_ecog_ext" / "targets.py"
        ),
        "reducer_implementation_sha256": sha256_file(
            HERE / "src" / "whisper_ecog_ext" / "reducer.py"
        ),
        "whisper_revision": args.whisper_revision,
        "reducer_seed": args.reducer_seed,
        "subjects": list(ALL_SUBJECTS),
    }
    root_contract["fingerprint"] = fingerprint_json(root_contract)
    contract_path = run_root / "run_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != root_contract:
            raise RuntimeError("Existing all-subject run contract differs from current source/protocol")
    else:
        atomic_write_json(contract_path, root_contract, overwrite=False)

    mel_extractor = MelTargetExtractor(n_mels=80, frame_hz=50.0)
    mel_provenance = mel_extractor.provenance()
    whisper_provenance = {
        "kind": "whisper_encoder_hidden_states",
        "model_name": "openai/whisper-base",
        "revision": args.whisper_revision,
        "layers": [3, 4, 5],
        "sample_rate": 16000,
        "chunk_seconds": 30,
        "alignment": "neighboring-frame linear interpolation",
        "single_forward_per_chunk_for_all_layers": True,
    }
    whisper_extractor: WhisperLayerTargetExtractor | None = None
    completions = _load_completions(run_root, protocol_sha256)
    for subject in ALL_SUBJECTS:
        if subject in completions:
            print(f"[skip validated] {subject}", flush=True)
            continue
        queue_state = {
            "schema_version": 1,
            "status": "running",
            "current_subject": subject,
            "completed_subjects": [item for item in ALL_SUBJECTS if item in completions],
            "remaining_subjects": [item for item in ALL_SUBJECTS if item not in completions],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(run_root / "queue_state.json", queue_state)
        print(f"===== {subject} matched PCA50 =====", flush=True)
        inventory = inventory_subject(args.data_root, subject, allow_confirmatory=True)
        events = load_visual_word_events_subject(
            args.data_root, subject, allow_confirmatory=True
        )
        recording_stop = recording_duration_seconds(inventory)
        block_definitions = make_visual_blocks(events, recording_stop)
        paths = subject_paths_frozen(args.data_root, subject)
        subject_cache = cache_root / subject
        subject_cache.mkdir(parents=True, exist_ok=True)
        blocks: list[Any] = []
        missing: list[int] = []
        fingerprints: list[str] = []
        for definition in block_definitions:
            fingerprint = build_extraction_fingerprint(
                inventory=inventory,
                events_path=paths["events"],
                block=definition,
                mel_provenance=mel_provenance,
                whisper_provenance=whisper_provenance,
            )
            fingerprints.append(fingerprint)
            cached = load_block_cache(
                subject_cache,
                definition.index,
                extraction_fingerprint=fingerprint,
            )
            blocks.append(cached)
            if cached is None:
                missing.append(definition.index)
        if missing:
            if whisper_extractor is None:
                whisper_extractor = WhisperLayerTargetExtractor(
                    revision=args.whisper_revision, device=args.device
                )
                actual = whisper_extractor.provenance()
                for key in (
                    "kind", "model_name", "revision", "layers", "sample_rate",
                    "chunk_seconds", "alignment",
                ):
                    if actual[key] != whisper_provenance[key]:
                        raise RuntimeError(f"Whisper provenance changed for {key}")
            with SWPDRecording(
                args.data_root, subject, allow_confirmatory=True
            ) as recording:
                for index in missing:
                    block = extract_one_block(
                        recording,
                        inventory,
                        block_definitions[index],
                        mel_extractor=mel_extractor,
                        whisper_extractor=whisper_extractor,
                        edge_guard_seconds=EDGE_GUARD_SECONDS,
                    )
                    save_block_cache(
                        block,
                        subject_cache,
                        extraction_fingerprint=fingerprints[index],
                    )
                    blocks[index] = block
                    print(
                        f"[{subject}] cached block {index} ({len(block.sample_ids)} frames)",
                        flush=True,
                    )
        if any(block is None for block in blocks):
            raise RuntimeError(f"Incomplete block cache for {subject}")
        subject_root = run_root / "subjects" / subject
        attempt = _next_attempt(subject_root)
        attempt.mkdir(parents=True, exist_ok=False)
        summary = run_matched_folds(
            tuple(blocks),
            attempt,
            reducer_seed=args.reducer_seed,
            reduced_dimension=REDUCED_DIMENSION,
            subject=subject,
        )
        subject_manifest = {
            "schema_version": 1,
            "kind": "swpd_matched_pca50_subject_run",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "subject": subject,
            "protocol_sha256": protocol_sha256,
            "inventory": inventory.to_dict(),
            "block_cache_fingerprints": fingerprints,
            "mel_provenance": mel_provenance,
            "whisper_provenance": whisper_provenance,
            "audio_nwb_measured_rate_hz": inventory.audio.rate_hz,
            "audio_processing_rate_hz": AUTHOR_AUDIO_PROCESSING_RATE,
            "summary": summary,
        }
        atomic_write_json(attempt / "subject_run_manifest.json", subject_manifest, overwrite=False)
        summary_path = attempt / "matched_linear_summary.json"
        completion = {
            "schema_version": 1,
            "kind": "swpd_matched_pca50_subject_completion",
            "subject": subject,
            "protocol_sha256": protocol_sha256,
            "attempt_directory": str(attempt),
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        completion["fingerprint"] = fingerprint_json(completion)
        atomic_write_json(subject_root / "subject_complete.json", completion, overwrite=False)
        completions[subject] = completion
        aggregate_path = _write_aggregate(run_root, completions)
        print(f"[completed] {subject} | aggregate={aggregate_path}", flush=True)

    final_aggregate = _write_aggregate(run_root, completions)
    atomic_write_json(
        run_root / "queue_state.json",
        {
            "schema_version": 1,
            "status": "completed",
            "current_subject": None,
            "completed_subjects": list(ALL_SUBJECTS),
            "remaining_subjects": [],
            "aggregate": str(final_aggregate),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"ALL 10 SUBJECTS COMPLETE | {final_aggregate}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NWBLayoutError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
