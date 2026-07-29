from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from swpd_contextual_neural_e2e.core import ContextualResidualDecoder  # noqa: E402
from swpd_contextual_neural_population.run_population import (  # noqa: E402
    CONTEXT_STEPS, OUTPUT_DIM, SUBJECTS, SEEDS, FOLDS,
)


def test_population_contract_geometry_is_frozen() -> None:
    assert SUBJECTS == tuple(f"sub-{number:02d}" for number in range(2, 10))
    assert SEEDS == (1, 2, 3, 4, 42)
    assert FOLDS == (0, 1, 2, 3, 4)
    assert CONTEXT_STEPS == 9
    assert OUTPUT_DIM == 50


def test_population_model_accepts_subject_specific_channel_counts() -> None:
    for channels in (54, 60, 115, 117, 127):
        model = ContextualResidualDecoder(CONTEXT_STEPS, channels, OUTPUT_DIM)
        output = model(torch.zeros(2, CONTEXT_STEPS, channels))
        assert output.shape == (2, OUTPUT_DIM)
        assert model.architecture_receipt()["dropout"] == 0.0


def test_population_scripts_keep_fit_and_test_separate() -> None:
    root = ROOT / "swpd_contextual_neural_population"
    source = (root / "run_population.py").read_text(encoding="utf-8")
    fit_script = (root / "scripts" / "run_fit.ps1").read_text(encoding="utf-8")
    evaluate_script = (root / "scripts" / "run_evaluate.ps1").read_text(encoding="utf-8")
    assert 'parser.add_argument("stage", choices=("fit", "evaluate"))' in source
    assert "200/200 selections frozen" in source
    assert "--diagnostic" in fit_script
    assert " evaluate " in evaluate_script
