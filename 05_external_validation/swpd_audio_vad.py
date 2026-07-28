#!/usr/bin/env python3
"""Create deterministic audio-only SWPD speech candidates for manual audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from whisper_ecog_ext.swpd.audio_vad import (  # noqa: E402
    EnergyVadConfig,
    detect_audio_energy_candidates,
    validate_audio_candidate_bundle,
    write_audio_candidates,
)
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    PILOT_SUBJECT,
    SWPDRecording,
    inventory_pilot,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", choices=[PILOT_SUBJECT], default=PILOT_SUBJECT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_tsv = output / "audio_vad_candidates_unreviewed.tsv"
    metadata = output / "audio_vad_candidates_metadata.json"
    if candidate_tsv.exists() and metadata.exists():
        validate_audio_candidate_bundle(candidate_tsv, metadata)
        print(f"[vad] immutable candidates already exist: {candidate_tsv}")
        return 0
    if candidate_tsv.exists() or metadata.exists():
        raise RuntimeError("partial VAD candidate output exists; inspect it before retrying")
    inventory = inventory_pilot(args.data_root)
    with SWPDRecording(args.data_root, PILOT_SUBJECT) as recording:
        audio = recording.read_audio(0, inventory.audio.shape[0])
    config = EnergyVadConfig(processing_input_rate_hz=48_000)
    intervals, provenance = detect_audio_energy_candidates(audio, config=config)
    write_audio_candidates(
        intervals,
        provenance,
        tsv_path=candidate_tsv,
        metadata_path=metadata,
        measured_nwb_rate_hz=inventory.audio.rate_hz,
    )
    print(f"[vad] {len(intervals)} unreviewed audio-only candidates: {candidate_tsv}")
    print("[gate] CLOSED: listen, edit, and approve with a bound audit receipt first")
    print("[safety] visual cue events were not read by this command")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
