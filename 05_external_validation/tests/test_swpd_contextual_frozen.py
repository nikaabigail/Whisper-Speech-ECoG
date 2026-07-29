from __future__ import annotations

import sys
from pathlib import Path
import json
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swpd_contextual_frozen.core import FrozenBlock, evaluate_subject  # noqa: E402
from swpd_contextual_frozen.run_frozen import DEFAULT_PROTOCOL, validate_protocol  # noqa: E402


class ContextualFrozenTests(unittest.TestCase):
    def test_five_block_evaluation_is_complete_and_finite(self) -> None:
        rng = np.random.default_rng(20260729)
        weights = rng.normal(size=(55, 80))
        blocks = []
        for block_index in range(5):
            rows = 64
            latent = rng.normal(size=(rows, 55))
            neural = latent + rng.normal(scale=0.15, size=(rows, 55))
            mel = latent @ weights + rng.normal(scale=0.5, size=(rows, 80))
            l4 = latent + rng.normal(scale=0.1, size=(rows, 55))
            sample_ids = np.asarray(
                [f"synthetic:{block_index}:{row}" for row in range(rows)]
            )
            blocks.append(
                FrozenBlock(
                    block_index,
                    sample_ids,
                    np.arange(rows) / 50 + block_index * 10,
                    neural.astype(np.float32),
                    mel.astype(np.float32),
                    l4.astype(np.float32),
                )
            )
        summary, predictions = evaluate_subject(blocks, "synthetic")
        self.assertEqual(len(summary["folds"]), 5)
        self.assertEqual(predictions["truth_mel80_z"].shape, (320, 80))
        self.assertTrue(np.isfinite(summary["delta_l4_minus_mel80"]))
        for fold in summary["folds"]:
            self.assertNotIn(fold["test_block"], fold["train_blocks"])
            self.assertNotIn(fold["validation_block"], fold["train_blocks"])

    def test_requires_exactly_five_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "five blocks"):
            evaluate_subject([], "synthetic")

    def test_portable_development_gate_validates_frozen_selection(self) -> None:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        protocol["development_result"]["path"] = "Z:/missing/development_summary.json"
        validate_protocol(protocol, DEFAULT_PROTOCOL)
        protocol["development_result"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "development_result"):
            validate_protocol(protocol, DEFAULT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
