#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if args.device == "cuda":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
