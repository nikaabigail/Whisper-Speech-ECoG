#!/usr/bin/env python3
"""File-based runtime preflight for the contextual neural E2E experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


MODULE_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = MODULE_ROOT.parent
sys.path[:0] = [str(EXTERNAL_ROOT), str(EXTERNAL_ROOT / "src")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import scipy
    import sklearn
    import torch

    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cache = args.cache_dir.expanduser().resolve()
    reference = args.reference_summary.expanduser().resolve()
    required = [reference, MODULE_ROOT / "core.py"]
    required.extend(cache / f"block_{index:02d}.json" for index in range(5))
    required.extend(cache / f"block_{index:02d}.npz" for index in range(5))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    from swpd_contextual_neural_e2e.core import ContextualResidualDecoder

    model = ContextualResidualDecoder(
        context_steps=9,
        channels=127,
        output_dim=50,
    )
    receipt = model.architecture_receipt()
    if receipt.get("dropout", 0) != 0 or receipt.get("batch_norm", False):
        raise SystemExit("The paired deterministic model must not use dropout/BatchNorm")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")
    print("Contextual neural E2E dependencies and frozen inputs: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
