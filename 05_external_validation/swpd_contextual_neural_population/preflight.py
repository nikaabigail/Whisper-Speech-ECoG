#!/usr/bin/env python3
"""Runtime and cache preflight for the frozen neural population follow-up."""

from pathlib import Path
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args()
    import numpy as np
    import sklearn
    import torch
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    missing = []
    for number in range(2, 10):
        subject = f"sub-{number:02d}"
        for fold in range(5):
            for suffix in ("json", "npz"):
                path = args.cache_root / subject / f"block_{fold:02d}.{suffix}"
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise SystemExit("Missing frozen caches:\n" + "\n".join(missing))
    print("Frozen population inputs: 8 subjects x 5 blocks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
