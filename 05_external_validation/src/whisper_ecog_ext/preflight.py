from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGES = (
    "torch",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "h5py",
    "pynwb",
    "librosa",
    "transformers",
)

MIN_FREE_GIB = {
    "swpd": 50.0,
    "vocalmind": 30.0,
    "all": 80.0,
}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def _cuda_probe(require_cuda: bool) -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    result: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": available,
    }
    if not available:
        if require_cuda:
            raise RuntimeError(
                "CUDA is unavailable. Update the NVIDIA driver and install the cu128 "
                "PyTorch wheel, or use --allow-cpu only for a smoke test."
            )
        return result

    props = torch.cuda.get_device_properties(0)
    result.update(
        {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gib": round(props.total_memory / (1024**3), 3),
            "compute_capability": f"{props.major}.{props.minor}",
            "cudnn_version": torch.backends.cudnn.version(),
        }
    )
    try:
        query = subprocess.run(
            (
                "nvidia-smi.exe",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        result["nvidia_driver"] = (
            query.stdout.splitlines()[0].strip()
            if query.returncode == 0 and query.stdout.splitlines()
            else "UNAVAILABLE"
        )
    except (OSError, subprocess.SubprocessError):
        result["nvidia_driver"] = "UNAVAILABLE"

    # A real forward/backward probe catches driver/wheel mismatches that a plain
    # torch.cuda.is_available() check can miss.
    x = torch.randn(32, 64, device="cuda", requires_grad=True)
    w = torch.randn(64, 16, device="cuda", requires_grad=True)
    loss = (x @ w).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA forward/backward probe produced non-finite values")
    result["cuda_forward_backward"] = "ok"
    return result


def _storage_report(path: Path, *, required_gib: float, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(path).free / (1024**3)
    result = {
        "path": str(path),
        "ascii_only": str(path).isascii(),
        "free_gib": round(free_gib, 3),
        "required_gib": float(required_gib),
        "ok": free_gib >= required_gib,
    }
    if not result["ok"]:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB is free at the {label} path {path}; "
            f"at least {required_gib:.1f} GiB is required."
        )
    return result


def build_report(
    dataset: str,
    data_root: Path,
    require_cuda: bool,
    *,
    cache_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_root)
    free_gib = usage.free / (1024**3)
    required_gib = MIN_FREE_GIB[dataset]
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_profile": dataset,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "data_root": str(data_root),
        "data_root_ascii_only": str(data_root).isascii(),
        "disk_free_gib": round(free_gib, 3),
        "disk_required_gib": required_gib,
        "disk_ok": free_gib >= required_gib,
        "package_versions": _package_versions(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
        },
    }
    missing = [name for name, version in report["package_versions"].items() if version == "MISSING"]
    if missing:
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")
    if not report["disk_ok"]:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB is free at {data_root}; "
            f"the {dataset} profile requires at least {required_gib:.1f} GiB."
        )
    if cache_root is not None:
        report["model_cache_storage"] = _storage_report(
            cache_root, required_gib=5.0, label="model cache"
        )
    if output_root is not None:
        output_required = 50.0 if dataset in {"swpd", "all"} else 20.0
        report["output_storage"] = _storage_report(
            output_root, required_gib=output_required, label="output"
        )
    report["accelerator"] = _cuda_probe(require_cuda=require_cuda)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a Windows host before external validation")
    parser.add_argument("--dataset", choices=sorted(MIN_FREE_GIB), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-cpu", action="store_true", help="allow CPU only for smoke tests")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    report = build_report(
        args.dataset,
        args.data_root,
        require_cuda=not args.allow_cpu,
        cache_root=args.cache_root,
        output_root=args.output_root,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(rendered + "\n", encoding="utf-8")
        tmp.replace(args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
