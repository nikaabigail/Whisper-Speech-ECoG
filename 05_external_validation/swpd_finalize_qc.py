#!/usr/bin/env python3
"""Finalize the SWPD matched run after the auditable sub-10 data-QC exclusion."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from swpd_matched_all import (  # noqa: E402
    ALL_SUBJECTS,
    LOCKED_CONFIRMATORY_SUBJECTS,
    TARGET_NAMES,
    _load_completions,
    aggregate_subject_summaries,
)
from whisper_ecog_ext.integrity import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
)
from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    inventory_subject,
    subject_paths_frozen,
)


QC_SUBJECT = "sub-10"
ANALYZABLE_SUBJECTS = ALL_SUBJECTS[:-1]
PRIMARY_SUBJECTS = LOCKED_CONFIRMATORY_SUBJECTS[:-1]
DEFAULT_AMENDMENT = (
    HERE
    / "configs"
    / "experiments"
    / "swpd_all_matched_pca50_v1_qc_amendment_sub10.json"
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_sub10_qc(
    rows: Sequence[Mapping[str, str]], *, ieeg_sample_count: int
) -> dict[str, Any]:
    """Verify the exact official-file defect before allowing the exclusion."""

    words = [row for row in rows if row.get("trial_type") == "word"]
    if len(words) != 100:
        raise ValueError(f"Expected 100 sub-10 word rows, found {len(words)}")
    durations = [float(row["duration"]) for row in words]
    valid = [row for row, duration in zip(words, durations) if duration > 0]
    invalid = [row for row, duration in zip(words, durations) if duration <= 0]
    if len(valid) != 95 or len(invalid) != 5 or invalid != words[-5:]:
        raise ValueError("sub-10 no longer has the audited 95-valid/5-final-invalid layout")
    invalid_onsets = {float(row["onset"]) for row in invalid}
    invalid_samples = {int(row["sample"]) for row in invalid}
    expected_final_sample = int(ieeg_sample_count) - 1
    if len(invalid_onsets) != 1 or invalid_samples != {expected_final_sample}:
        raise ValueError(
            "sub-10 invalid trials do not all point to the final recorded sample"
        )
    return {
        "word_event_count": len(words),
        "positive_duration_word_event_count": len(valid),
        "zero_duration_final_word_event_count": len(invalid),
        "final_placeholder_onset_seconds": next(iter(invalid_onsets)),
        "final_placeholder_sample": expected_final_sample,
        "ieeg_sample_count": int(ieeg_sample_count),
    }


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    fieldnames = list(rows[0]) if rows else ["subject"]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_rows(primary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in primary["subject_rows"]:
        row: dict[str, Any] = {"subject": source["subject"]}
        for target in TARGET_NAMES:
            row[f"{target}_fisher_r"] = source[target]["fisher_r"]
            row[f"{target}_standardized_mse"] = source[target]["standardized_mse"]
        for target in ("L3", "L4", "L5"):
            row[f"{target}_minus_mel80_fisher_r"] = (
                source[target]["fisher_r"] - source["mel80"]["fisher_r"]
            )
        rows.append(row)
    return rows


def finalize(
    *, data_root: Path, run_root: Path, amendment_path: Path
) -> tuple[Path, Path]:
    run_root = run_root.expanduser().resolve()
    contract_path = run_root / "run_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Run contract is missing: {contract_path}")
    contract = read_json(contract_path)
    protocol_sha256 = str(contract["protocol_sha256"])
    completions = _load_completions(run_root, protocol_sha256)
    missing = [subject for subject in ANALYZABLE_SUBJECTS if subject not in completions]
    if missing:
        raise RuntimeError(f"Analyzable subject completions are missing: {missing}")
    if QC_SUBJECT in completions:
        raise RuntimeError("sub-10 unexpectedly has a completion receipt; audit manually")

    amendment_path = amendment_path.expanduser().resolve()
    amendment = read_json(amendment_path)
    if amendment.get("excluded_subject") != QC_SUBJECT:
        raise ValueError("QC amendment does not identify sub-10")
    paths = subject_paths_frozen(data_root, QC_SUBJECT)
    inventory = inventory_subject(data_root, QC_SUBJECT, allow_confirmatory=True)
    observed_qc = validate_sub10_qc(
        _read_tsv(paths["events"]), ieeg_sample_count=inventory.ieeg.shape[0]
    )
    expected_qc = {
        "word_event_count": amendment["observed_word_event_count"],
        "positive_duration_word_event_count": amendment[
            "observed_positive_duration_word_event_count"
        ],
        "zero_duration_final_word_event_count": amendment[
            "observed_zero_duration_final_word_event_count"
        ],
        "final_placeholder_sample": amendment["final_placeholder_sample"],
        "ieeg_sample_count": amendment["ieeg_sample_count"],
    }
    for key, expected in expected_qc.items():
        if observed_qc[key] != expected:
            raise RuntimeError(
                f"Observed sub-10 QC differs from amendment for {key}: "
                f"{observed_qc[key]} != {expected}"
            )

    summaries = {
        subject: read_json(Path(completions[subject]["summary_path"]))
        for subject in ANALYZABLE_SUBJECTS
    }
    primary = aggregate_subject_summaries(summaries, cohort=PRIMARY_SUBJECTS)
    secondary = aggregate_subject_summaries(summaries, cohort=ANALYZABLE_SUBJECTS)
    if len(primary["subjects"]) != 8 or len(secondary["subjects"]) != 9:
        raise RuntimeError("QC-adjusted cohort sizes are not 8 primary / 9 secondary")

    created = datetime.now(timezone.utc).isoformat()
    summary_dir = run_root / "summary"
    exclusion_path = summary_dir / "sub-10_qc_exclusion.json"
    exclusion = {
        "schema_version": 1,
        "kind": "swpd_subject_qc_exclusion",
        "created_utc": created,
        "subject": QC_SUBJECT,
        "decision": amendment["decision"],
        "reason_code": amendment["reason_code"],
        "rationale": amendment["rationale"],
        "observed_qc": observed_qc,
        "events_path": str(paths["events"]),
        "events_sha256": sha256_file(paths["events"]),
        "nwb_path": str(paths["nwb"]),
        "nwb_sha256": sha256_file(paths["nwb"]),
        "amendment_path": str(amendment_path),
        "amendment_sha256": sha256_file(amendment_path),
        "parent_protocol_sha256": protocol_sha256,
    }
    atomic_write_json(exclusion_path, exclusion, overwrite=False)

    final_path = summary_dir / "swpd_matched_pca50_qc_final_summary.json"
    payload = {
        "schema_version": 1,
        "kind": "swpd_matched_pca50_qc_final_summary",
        "created_utc": created,
        "source_run_root": str(run_root),
        "source_run_contract": str(contract_path),
        "source_run_contract_sha256": sha256_file(contract_path),
        "parent_protocol_sha256": protocol_sha256,
        "qc_amendment": amendment,
        "qc_amendment_path": str(amendment_path),
        "qc_amendment_sha256": sha256_file(amendment_path),
        "qc_exclusion_path": str(exclusion_path),
        "qc_exclusion_sha256": sha256_file(exclusion_path),
        "planned_subjects": list(ALL_SUBJECTS),
        "completed_analyzable_subjects": list(ANALYZABLE_SUBJECTS),
        "excluded_subjects": [QC_SUBJECT],
        "primary_confirmatory_after_qc": primary,
        "secondary_all_analyzable_after_qc": secondary,
        "development_subject_excluded_from_primary": "sub-01",
        "test_results_were_not_imputed": True,
    }
    atomic_write_json(final_path, payload, overwrite=False)
    csv_path = summary_dir / "swpd_matched_pca50_qc_primary_subjects.csv"
    _atomic_write_csv(csv_path, _csv_rows(primary))
    atomic_write_json(
        run_root / "queue_state.json",
        {
            "schema_version": 1,
            "status": "completed_with_data_qc_exclusion",
            "current_subject": None,
            "completed_subjects": list(ANALYZABLE_SUBJECTS),
            "excluded_subjects": [QC_SUBJECT],
            "remaining_subjects": [],
            "primary_n": 8,
            "secondary_n": 9,
            "final_summary": str(final_path),
            "primary_csv": str(csv_path),
            "updated_utc": created,
        },
    )
    return final_path, csv_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    final_path, csv_path = finalize(
        data_root=args.data_root,
        run_root=args.run_root,
        amendment_path=args.amendment,
    )
    print(f"[done] {final_path}")
    print(f"[csv]  {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
