#!/usr/bin/env python3
"""Run the pinned authors' MEL reconstruction on SWPD sub-01 only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from whisper_ecog_ext.swpd.author_mel import (  # noqa: E402
    CV_FOLDS,
    PCA_COMPONENTS,
    circular_shift_null,
    extract_features_from_pilot,
    run_nonshuffled_cv,
)
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    ConfirmatoryDataLocked,
    NWBLayoutError,
    PILOT_SUBJECT,
    inventory_pilot,
)


OFFICIAL_COMMIT = "cfb563696a8d44207532e3777ba6c5aabaf68805"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", choices=[PILOT_SUBJECT], default=PILOT_SUBJECT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomizations", type=int, default=1000)
    parser.add_argument("--channel-batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.randomizations <= 0 or args.channel_batch_size <= 0:
        raise ValueError("randomizations and channel-batch-size must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == HERE or _inside(output_dir, HERE):
        raise ValueError("Baseline outputs must be outside the source checkout")
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_pilot(args.data_root)
    if (
        inventory.word_event_count != 100
        or inventory.fixation_event_count != 99
        or inventory.unique_prompt_count != 100
    ):
        raise NWBLayoutError(
            "sub-01 event inventory differs from the pinned 100-word/99-fixation contract"
        )

    print("[sub-01] extracting exact author high-gamma and 23-bin log-MEL features")
    features = extract_features_from_pilot(
        args.data_root, channel_batch_size=args.channel_batch_size
    )
    print(
        f"[features] neural={features.neural.shape} mel={features.mel.shape}; "
        "all transforms remain fold-train-only"
    )
    predictions, cv = run_nonshuffled_cv(
        features.neural,
        features.mel,
        folds=CV_FOLDS,
        pca_components=PCA_COMPONENTS,
    )
    null = circular_shift_null(
        features.mel, rounds=args.randomizations, seed=args.seed
    )
    observed = cv["mean_correlation"]
    null_means = np.asarray(null["mean_correlations"])
    empirical_p = float((1 + np.sum(null_means >= observed)) / (1 + len(null_means)))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    prediction_path = output_dir / f"swpd_sub01_author_mel_predictions_{timestamp}.npy"
    result_path = output_dir / f"swpd_sub01_author_mel_result_{timestamp}.json"
    temporary_prediction = prediction_path.with_name(prediction_path.name + ".partial")
    with temporary_prediction.open("wb") as handle:
        np.save(handle, predictions)
        handle.flush()
    temporary_prediction.replace(prediction_path)

    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "SWPD / OSF nrgx6",
        "subject": PILOT_SUBJECT,
        "scope": "development_subject_only",
        "confirmatory_subjects_read": False,
        "exact_reproduction": True,
        "official_code_commit": OFFICIAL_COMMIT,
        "protocol": {
            "high_gamma_hz": [70, 170],
            "bandstop_hz": [[98, 102], [148, 152]],
            "window_ms": 50,
            "frame_shift_ms": 10,
            "neural_context_ms": [-200, -150, -100, -50, 0, 50, 100, 150, 200],
            "mel_bins": 23,
            "pca_components": PCA_COMPONENTS,
            "cv": "10-fold non-shuffled",
            "regression": "ordinary least squares",
            "normalization_and_pca_fit": "fold train only",
            "null": "author circular shift with pinned deterministic seed",
        },
        "modernization_notes": [
            "np.float replaced by built-in float",
            "removed scipy.hanning replaced by scipy.signal.windows.hann equivalent",
            "measured NWB audio clock retained in provenance while processing uses the authors' explicit 48000 Hz assumption and factor-3 decimation",
            "zero-variance neural columns receive scale 1 within each training fold",
            "waveform/Griffin-Lim synthesis omitted because it does not affect MEL reconstruction metrics",
        ],
        "inventory": inventory.to_dict(),
        "feature_shapes": {
            "neural": list(features.neural.shape),
            "mel": list(features.mel.shape),
        },
        "sampling_hz": {
            "ieeg": features.ieeg_rate_hz,
            "audio_nwb_measured": features.measured_audio_rate_hz,
            "audio_author_processing_assumption": features.author_processing_audio_rate_hz,
            "audio_target": features.target_audio_rate_hz,
        },
        "constant_audio": features.constant_audio,
        "cv_results": cv,
        "circular_shift_null": null,
        "empirical_p_mean_correlation": empirical_p,
        "prediction_file": str(prediction_path),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    temporary_result = result_path.with_name(result_path.name + ".partial")
    temporary_result.write_text(rendered, encoding="utf-8")
    temporary_result.replace(result_path)
    print(f"[done] mean Pearson r={observed:.4f}; empirical p={empirical_p:.6f}")
    print(f"[saved] {result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ConfirmatoryDataLocked,
        NWBLayoutError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
