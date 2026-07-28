from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from swpd_finalize_qc import validate_sub10_qc  # noqa: E402


def _rows() -> list[dict[str, str]]:
    rows = []
    for index in range(100):
        invalid = index >= 95
        rows.append(
            {
                "trial_type": "word",
                "onset": "285.0576171875" if invalid else str(index * 3.0),
                "duration": "0.0" if invalid else "2.0",
                "sample": "291899" if invalid else str(index * 3072),
            }
        )
    return rows


def test_exact_sub10_qc_layout_is_accepted() -> None:
    observed = validate_sub10_qc(_rows(), ieeg_sample_count=291900)
    assert observed["positive_duration_word_event_count"] == 95
    assert observed["zero_duration_final_word_event_count"] == 5
    assert observed["final_placeholder_sample"] == 291899


def test_qc_rejects_imputation_or_changed_source_layout() -> None:
    rows = _rows()
    rows[-1]["duration"] = "2.0"
    with pytest.raises(ValueError, match="95-valid/5-final-invalid"):
        validate_sub10_qc(rows, ieeg_sample_count=291900)

    rows = _rows()
    rows[-1]["sample"] = "291898"
    with pytest.raises(ValueError, match="final recorded sample"):
        validate_sub10_qc(rows, ieeg_sample_count=291900)
