#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
args = parser.parse_args()

import numpy
import scipy
import sklearn
import librosa
import h5py
import torch
import transformers

print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if args.device == "cuda":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"NumPy: {numpy.__version__} | SciPy: {scipy.__version__} | scikit-learn: {sklearn.__version__}")
print(f"librosa: {librosa.__version__} | h5py: {h5py.__version__} | transformers: {transformers.__version__}")
