#!/usr/bin/env python3
"""Dependency preflight for the CPU-only contextual alternating experiment."""

import numpy as np
import scipy
import sklearn

print("Contextual covariance-alternating dependencies: OK")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print("Runtime: CPU (deterministic PCA/OLS/SVD; CUDA is not used)")
