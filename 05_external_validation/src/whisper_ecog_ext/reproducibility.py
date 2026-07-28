"""Deterministic random-state setup with an auditable runtime receipt."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np


def set_deterministic_seed(seed: int, *, strict_torch: bool = False) -> dict[str, Any]:
    """Seed Python, NumPy and PyTorch and return the effective settings.

    ``PYTHONHASHSEED`` only affects child interpreters when set after Python has
    started, so it is recorded but never presented as retroactive determinism.
    """

    if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    receipt: dict[str, Any] = {
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "pythonhashseed_for_child_processes": str(seed),
        "numpy": np.__version__,
        "torch_available": False,
        "strict_torch_algorithms": bool(strict_torch),
    }
    try:
        import torch
    except ImportError:
        return receipt

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(strict_torch)
    receipt.update(
        {
            "torch_available": True,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
    )
    return receipt


def make_torch_generator(seed: int, *, device: str = "cpu"):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker from its framework-provided seed."""

    del worker_id
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
