#!/usr/bin/env python3
"""Run the development matched-linear MEL80/Whisper analysis on SWPD sub-01."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
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
    load_audited_speech_intervals,
    load_block_cache,
    make_visual_blocks,
    run_matched_folds,
    save_block_cache,
)
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    ConfirmatoryDataLocked,
    NWBLayoutError,
    PILOT_SUBJECT,
    SWPDRecording,
    inventory_pilot,
    load_visual_word_events,
    recording_duration_seconds,
    subject_paths,
)
from whisper_ecog_ext.targets import (  # noqa: E402
    MelTargetExtractor,
    WhisperLayerTargetExtractor,
)


DEFAULT_WHISPER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
PINNED_DATASET_MANIFEST = HERE / "manifests" / "swpd_osf_nrgx6.json"
DEFAULT_PROTOCOL_CONFIG = (
    HERE / "configs" / "experiments" / "swpd_sub01_matched_pca50_v1.json"
)


def validate_protocol_payload(payload: dict[str, object]) -> None:
    """Fail closed when the frozen matched-comparison contract changes."""

    if payload.get("schema_version") != 1:
        raise ValueError("Matched protocol schema_version must be 1")
    if payload.get("status") != "frozen_development_sub01":
        raise ValueError("Matched protocol must be frozen_development_sub01")
    if payload.get("subject") != PILOT_SUBJECT:
        raise ValueError("Matched protocol is restricted to sub-01")
    comparison = payload.get("matched_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("Matched protocol has no matched_comparison object")
    expected = {
        "targets": list(TARGET_NAMES),
        "common_target_dimension": REDUCED_DIMENSION,
        "frame_grid_ms": int(round(FRAME_SHIFT_SECONDS * 1000)),
        "high_gamma_window_ms": 50,
        "primary_metric": "fold_mean_fisher_z_component_correlation",
        "test_scope": "all_frames",
    }
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise ValueError(f"Matched protocol changed {key}: {comparison.get(key)!r}")
    target_transform = comparison.get("target_transform")
    neural_transform = comparison.get("neural_transform")
    decoder = comparison.get("decoder")
    splits = comparison.get("splits")
    if not isinstance(target_transform, dict) or (
        target_transform.get("fit_scope"),
        target_transform.get("standardize"),
        target_transform.get("reducer"),
        target_transform.get("whiten"),
    ) != ("outer_train_only", True, "PCA", True):
        raise ValueError("Target transform must be train-only standardized PCA whitening")
    if not isinstance(neural_transform, dict) or (
        neural_transform.get("shared_across_targets"),
        neural_transform.get("fit_scope"),
        neural_transform.get("components"),
        neural_transform.get("whiten"),
    ) != (True, "outer_train_only", REDUCED_DIMENSION, True):
        raise ValueError("Neural transform must be shared train-only PCA50 whitening")
    if not isinstance(decoder, dict) or (
        decoder.get("kind"), decoder.get("hyperparameters_identical_across_targets")
    ) != ("ordinary_least_squares", True):
        raise ValueError("Decoder must be identical ordinary least squares")
    if not isinstance(splits, dict) or (
        splits.get("block_count"),
        splits.get("trials_per_block"),
        splits.get("visual_events_define_blocks_only"),
        splits.get("visual_events_are_not_acoustic_onsets"),
    ) != (BLOCK_COUNT, TRIALS_PER_BLOCK, True, True):
        raise ValueError("Split contract must remain five adjacent 20-trial blocks")
    baseline = payload.get("author_mel_baseline")
    if not isinstance(baseline, dict) or baseline.get("included_in_matched_fit") is not False:
        raise ValueError("Author MEL baseline must remain a separate frozen control")


def load_protocol_config(path: Path, *, verify_baseline: bool = True) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    payload = read_json(resolved)
    validate_protocol_payload(payload)
    if verify_baseline:
        baseline = payload["author_mel_baseline"]
        assert isinstance(baseline, dict)
        baseline_path = Path(str(baseline["path"])).expanduser().resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Frozen author MEL baseline is missing: {baseline_path}")
        if sha256_file(baseline_path) != baseline["sha256"]:
            raise RuntimeError("Frozen author MEL baseline checksum changed")
    return payload


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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-config", type=Path, default=DEFAULT_PROTOCOL_CONFIG
    )
    parser.add_argument("--subject", choices=[PILOT_SUBJECT], default=PILOT_SUBJECT)
    parser.add_argument("--whisper-revision", default=DEFAULT_WHISPER_REVISION)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Default: CUDA if available")
    parser.add_argument(
        "--speech-mask-tsv",
        type=Path,
        help="Optional independently audited audio onset/offset TSV; visual events are rejected as speech labels",
    )
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--reducer-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol_config.expanduser().resolve()
    protocol = load_protocol_config(protocol_path)
    cache_dir = _external_directory(args.cache_dir, "cache-dir")
    output_dir = _external_directory(args.output_dir, "output-dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output-dir must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_pilot(args.data_root)
    events = load_visual_word_events(args.data_root)
    recording_stop = recording_duration_seconds(inventory)
    block_definitions = make_visual_blocks(events, recording_stop)
    paths = subject_paths(args.data_root, PILOT_SUBJECT)

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

    blocks = []
    fingerprints = []
    missing = []
    for definition in block_definitions:
        fingerprint = build_extraction_fingerprint(
            inventory=inventory,
            events_path=paths["events"],
            block=definition,
            mel_provenance=mel_provenance,
            whisper_provenance=whisper_provenance,
        )
        fingerprints.append(fingerprint)
        try:
            cached = load_block_cache(
                cache_dir, definition.index, extraction_fingerprint=fingerprint
            )
        except RuntimeError:
            if not args.force_cache:
                raise
            arrays_path = cache_dir / f"block_{definition.index:02d}.npz"
            manifest_path = cache_dir / f"block_{definition.index:02d}.json"
            for path in (arrays_path, manifest_path):
                if path.exists():
                    path.unlink()
            cached = None
        if cached is None:
            missing.append(definition.index)
            blocks.append(None)
        else:
            print(f"[cache] reuse block {definition.index}")
            blocks.append(cached)

    if missing:
        print(
            f"[targets] extracting blocks {missing}; Whisper L3/L4/L5 share one forward per 30 s chunk"
        )
        whisper_extractor = WhisperLayerTargetExtractor(
            revision=args.whisper_revision,
            device=args.device,
        )
        actual_provenance = whisper_extractor.provenance()
        for key in ("kind", "model_name", "revision", "layers", "sample_rate", "chunk_seconds", "alignment"):
            if actual_provenance[key] != whisper_provenance[key]:
                raise RuntimeError(f"Whisper provenance changed for {key}")
        with SWPDRecording(args.data_root, PILOT_SUBJECT) as recording:
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
                    cache_dir,
                    extraction_fingerprint=fingerprints[index],
                )
                blocks[index] = block
                print(f"[cache] saved block {index} ({len(block.sample_ids)} frames)")

    if any(block is None for block in blocks):
        raise RuntimeError("Not all five matched-linear blocks are available")
    complete_blocks = tuple(blocks)
    speech_intervals = load_audited_speech_intervals(args.speech_mask_tsv)
    summary = run_matched_folds(
        complete_blocks,
        output_dir,
        speech_intervals=speech_intervals,
        reducer_seed=args.reducer_seed,
    )
    run_manifest = {
        "schema_version": 1,
        "kind": "swpd_sub01_matched_linear_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": PILOT_SUBJECT,
        "scope": "development_only",
        "confirmatory_subjects_read": False,
        "dataset_manifest": str(PINNED_DATASET_MANIFEST),
        "dataset_manifest_sha256": sha256_file(PINNED_DATASET_MANIFEST),
        "protocol_config": str(protocol_path),
        "protocol_config_sha256": sha256_file(protocol_path),
        "protocol": protocol,
        "inventory": inventory.to_dict(),
        "block_definitions": [definition.__dict__ for definition in block_definitions],
        "block_cache_fingerprints": fingerprints,
        "mel_provenance": mel_provenance,
        "whisper_provenance": whisper_provenance,
        "audio_nwb_measured_rate_hz": inventory.audio.rate_hz,
        "audio_processing_rate_hz": AUTHOR_AUDIO_PROCESSING_RATE,
        "speech_mask_tsv": str(args.speech_mask_tsv.resolve()) if args.speech_mask_tsv else None,
        "speech_mask_tsv_sha256": sha256_file(args.speech_mask_tsv.resolve())
        if args.speech_mask_tsv
        else None,
        "offline_noncausal_analysis": True,
        "summary_file": str(output_dir / "matched_linear_summary.json"),
        "aggregate_test": summary["aggregate_test"],
    }
    atomic_write_json(output_dir / "run_manifest.json", run_manifest, overwrite=False)
    print(f"[done] development matched-linear result: {output_dir}")
    print("[safety] only sub-01 was accessible; sub-02..sub-10 remain code-locked")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ConfirmatoryDataLocked,
        FileNotFoundError,
        NWBLayoutError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
