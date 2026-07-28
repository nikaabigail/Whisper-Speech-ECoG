from __future__ import annotations

import copy
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from swpd_matched_all import (  # noqa: E402
    ALL_SUBJECTS,
    DEFAULT_PROTOCOL,
    aggregate_subject_summaries,
    load_production_protocol,
    validate_production_protocol,
)


def _summary(offset: float) -> dict:
    targets = {}
    for index, target in enumerate(("mel80", "L3", "L4", "L5")):
        correlation = offset + index * 0.01 + offset * index * 0.001
        targets[target] = {
            "test_all": {
                "fisher_z_component_correlation": {
                    "mean": correlation,
                    "sd": 0.001,
                    "fold_count": 5,
                },
                "standardized_mse": {
                    "mean": 1.0 - correlation,
                    "sd": 0.01,
                    "fold_count": 5,
                },
            }
        }
    return {"aggregate_test": targets}


def test_frozen_all_subject_protocol() -> None:
    protocol = load_production_protocol(DEFAULT_PROTOCOL, verify_baseline=False)
    assert tuple(protocol["all_subjects"]) == ALL_SUBJECTS
    assert protocol["primary_confirmatory_subjects"] == list(ALL_SUBJECTS[1:])
    mutated = copy.deepcopy(protocol)
    mutated["matched_comparison"]["common_target_dimension"] = 49
    try:
        validate_production_protocol(mutated)
    except ValueError as exc:
        assert "common_target_dimension" in str(exc)
    else:
        raise AssertionError("Changed PCA dimension was accepted")


def test_subject_aggregation_excludes_development_subject() -> None:
    summaries = {
        "sub-01": _summary(0.0),
        "sub-02": _summary(0.1),
        "sub-03": _summary(0.2),
    }
    result = aggregate_subject_summaries(summaries, cohort=ALL_SUBJECTS[1:])
    assert result["subjects"] == ["sub-02", "sub-03"]
    assert result["systems"]["mel80"]["fisher_r"]["n"] == 2
    assert result["contrasts"]["L3_minus_mel80"]["wins"] == 2
    assert result["contrasts"]["L3_minus_mel80"]["two_sided_paired_t_p_holm"] is not None


def test_empty_confirmatory_aggregate_is_json_safe() -> None:
    result = aggregate_subject_summaries(
        {"sub-01": _summary(0.0)}, cohort=ALL_SUBJECTS[1:]
    )
    assert result["subjects"] == []
    assert result["systems"]["mel80"]["fisher_r"] == {
        "mean": None,
        "sd": None,
        "sem": None,
        "ci95_t": None,
        "n": 0,
    }
