#!/usr/bin/env python3
"""Engineering-only real-data CUDA smoke using one VocalMind repetition-6 trial."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from whisper_ecog_ext.classifier import HiddenSequenceClassifier  # noqa: E402
from whisper_ecog_ext.data.vocalmind import VocalMindAdapter  # noqa: E402
from whisper_ecog_ext.integrity import atomic_write_json, sha256_file  # noqa: E402
from whisper_ecog_ext.model import OneSecondEcogEncoder  # noqa: E402
from whisper_ecog_ext.source_identity import capture_source_identity  # noqa: E402
from whisper_ecog_ext.vocalmind_neural import VocalMindNeuralPreprocessor  # noqa: E402
from whisper_ecog_ext.vocalmind_primary import (  # noqa: E402
    DefaultAcousticTargetProvider,
    load_primary_config,
    stereo_to_mono,
)
from whisper_ecog_ext.vocalmind_targets import (  # noqa: E402
    VocalMindAuthorMelTargetExtractor,
)


CONFIG_PATH = HERE / "configs" / "experiments" / "vocalmind_primary_production.json"


def _external_output(path: Path, data_root: Path) -> Path:
    result = path.expanduser().resolve()
    for forbidden, label in ((HERE, "source checkout"), (data_root, "dataset")):
        try:
            result.relative_to(forbidden.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(f"output directory must be outside the {label}")
    result.mkdir(parents=True, exist_ok=True)
    return result


def _finite_gradient_norm(module: torch.nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().double().cpu()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("smoke produced a non-finite gradient")
        squared += gradient.square().sum()
    value = float(torch.sqrt(squared).item())
    if not np.isfinite(value) or value <= 0:
        raise FloatingPointError("smoke gradient norm is not finite and positive")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--trial-id",
        help="Optional rep6 trial ID; defaults to the first pinned rep6 trial",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = _external_output(args.output_dir, data_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config = load_primary_config(CONFIG_PATH)
    adapter = VocalMindAdapter(data_root)
    rep6 = tuple(trial for trial in adapter.discover() if trial.repetition == 6)
    if len(rep6) != 19 or any(trial.repetition != 6 for trial in rep6):
        raise RuntimeError("engineering smoke must see exactly the 19 released rep6 trials")
    trial = adapter.trial(args.trial_id) if args.trial_id else rep6[0]
    if trial.repetition != 6:
        raise ValueError("numeric smoke access is restricted to repetition 6")

    seed = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    preprocess = VocalMindNeuralPreprocessor.from_mapping(
        dict(config.payload["neural_preprocessing"]),
        expected_channel_count=len(adapter.contract.channel_ids),
    )
    started = time.perf_counter()
    eeg = preprocess.transform(adapter.load_eeg(trial))
    audio = stereo_to_mono(adapter.load_audio(trial))
    endpoint_samples = np.arange(1000, 3000, 50, dtype=np.int64)
    target_times_s = endpoint_samples.astype(np.float64) / 1000.0

    provider = DefaultAcousticTargetProvider(config, device=args.device)
    targets = provider.extract(audio, 44_100, target_times_s)
    expected_dimensions = {"mel": 80, "L3": 512, "L4": 512, "L5": 512}
    for name, dimension in expected_dimensions.items():
        value = np.asarray(targets[name])
        if value.shape != (len(endpoint_samples), dimension) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {name} smoke target: {value.shape}")

    # The official no-peak waveform policy is retained only as a fidelity check;
    # it is never substituted for the shared-normalization primary target.
    mel_spec = config.payload["targets"]["mel"]
    author_no_peak = VocalMindAuthorMelTargetExtractor(
        sample_rate_hz=int(mel_spec["sample_rate_hz"]),
        n_fft=int(mel_spec["n_fft"]),
        hop_length=int(mel_spec["hop_length"]),
        win_length=int(mel_spec["win_length"]),
        window=str(mel_spec["window"]),
        n_mels=int(mel_spec["n_mels"]),
        fmin_hz=float(mel_spec["fmin_hz"]),
        fmax_hz=float(mel_spec["fmax_hz"]),
        epsilon=float(mel_spec["epsilon"]),
        peak_normalize=False,
    ).extract_aligned(audio, 44_100, target_times_s)

    mean = eeg.mean(axis=0, dtype=np.float64)
    scale = eeg.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = np.asarray((eeg - mean) / scale, dtype=np.float32)
    windows = np.stack(
        [normalized[end - 1000 : end + 1].T for end in endpoint_samples],
        axis=0,
    )
    inputs = torch.from_numpy(windows).to(args.device)
    l3_target = torch.from_numpy(np.asarray(targets["L3"][:, :50], dtype=np.float32)).to(
        args.device
    )

    encoder = OneSecondEcogEncoder(input_channels=110, target_dim=50).to(args.device)
    encoder.train()
    prediction = encoder(inputs)
    regression_loss = F.mse_loss(prediction, l3_target)
    if not torch.isfinite(regression_loss):
        raise FloatingPointError("non-finite regression smoke loss")
    regression_loss.backward()
    regression_gradient_norm = _finite_gradient_norm(encoder)

    encoder.zero_grad(set_to_none=True)
    with torch.no_grad():
        hidden = encoder(inputs, return_hidden=True).detach()
    classifier = HiddenSequenceClassifier(input_features=3030, num_classes=20).to(args.device)
    classifier.train()
    logits = classifier(hidden.T.unsqueeze(0))
    class_index = adapter.contract.words.index(trial.word)
    classification_loss = F.cross_entropy(
        logits, torch.tensor([class_index], dtype=torch.long, device=args.device)
    )
    if not torch.isfinite(classification_loss):
        raise FloatingPointError("non-finite classifier smoke loss")
    classification_loss.backward()
    classifier_gradient_norm = _finite_gradient_norm(classifier)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    matched_mel = np.asarray(targets["mel"], dtype=np.float64).reshape(-1)
    fidelity_mel = np.asarray(author_no_peak, dtype=np.float64).reshape(-1)
    mel_correlation = float(np.corrcoef(matched_mel, fidelity_mel)[0, 1])
    receipt = {
        "schema_version": 1,
        "kind": "vocalmind_rep6_engineering_smoke",
        "scientific_result": False,
        "classification_metric_reported": False,
        "numeric_primary_repetitions_accessed": [],
        "numeric_development_repetition_accessed": 6,
        "trial_id": trial.trial_id,
        "eeg_sha256": sha256_file(trial.eeg_path),
        "audio_sha256": sha256_file(trial.audio_path),
        "source_identity": capture_source_identity(HERE),
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "endpoint_count": int(len(endpoint_samples)),
        "window_shape": list(windows.shape),
        "target_shapes": {name: list(np.asarray(value).shape) for name, value in targets.items()},
        "regression_loss_diagnostic": float(regression_loss.detach().cpu()),
        "regression_gradient_norm": regression_gradient_norm,
        "classifier_loss_diagnostic": float(classification_loss.detach().cpu()),
        "classifier_gradient_norm": classifier_gradient_norm,
        "shared_peak_vs_author_no_peak_mel_correlation": mel_correlation,
        "elapsed_seconds": float(elapsed),
        "cuda_peak_memory_mib": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if args.device == "cuda"
            else None
        ),
        "target_provenance": provider.provenance(),
        "neural_preprocessing": preprocess.provenance(),
    }
    output_path = output_dir / "vocalmind_rep6_smoke_receipt.json"
    atomic_write_json(output_path, receipt, overwrite=False)
    print(f"[done] rep6 engineering smoke passed in {elapsed:.2f} s")
    print(f"[saved] {output_path}")
    print("[safety] no repetition 1-5 numeric file was loaded; no metric is reportable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
