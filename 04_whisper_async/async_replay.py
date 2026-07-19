#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style asynchronous replay for the frozen Whisper-guided decoder.

The script deliberately does not modify the synchronous source project. It
imports model definitions from the adjacent ``02_whisper_sync`` directory,
reads checkpoints and HDF5 files in read-only mode, and writes artifacts next
to this script.

Protocol: Petrosyan et al., JNE 19 (2022) 066016, sections 3.3.2, 3.5, 4.5.
The already trained synchronous word head is evaluated on trailing, overlapping
windows of continuous hidden activity.  Word boundaries are used only after
inference for event-level scoring.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = SCRIPT_DIR.parent / "02_whisper_sync"
OUTPUT_ROOT = SCRIPT_DIR.parent / "artifacts" / "frozen_replay"
ARTICLE_DOI = "10.1088/1741-2552/aca1e1"


@dataclass(frozen=True)
class ModelSpec:
    layer: int
    model_name: str
    date: str
    seed: int
    max_words_length: int
    regression_path: Path
    classifier_path: Path
    result_path: Path
    synchronous_accuracy: Optional[float]


@dataclass
class ModelBundle:
    spec: ModelSpec
    regression: object
    classifier: object


@dataclass(frozen=True)
class GroundTruth:
    file_index: int
    event_index: int
    start_s: float
    end_s: float
    class_index: int
    word: str


@dataclass(frozen=True)
class Candidate:
    file_index: int
    time_s: float
    class_index: int
    score: float
    gt_event_index: Optional[int]
    gt_class_index: Optional[int]


@dataclass
class FileProfile:
    file_index: int
    filename: str
    duration_s: float
    times_s: "np.ndarray"
    probabilities: Dict[str, "np.ndarray"]
    smoothed: Dict[str, "np.ndarray"]
    ground_truth: List[GroundTruth]
    diagnostics: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boundary-free continuous replay of frozen Whisper L3/L4/L5 word decoders."
    )
    parser.add_argument("--base-project", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--patient", default="ivanova", choices=("ivanova", "procenko"))
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--step-frames", type=int, default=1,
                        help="Stride on the ~100 Hz hidden timeline; 1 is paper-style dense replay.")
    parser.add_argument("--smooth-ms", type=float, default=200.0)
    parser.add_argument("--smoothing", choices=("centered", "causal"), default="centered")
    parser.add_argument("--threshold-points", type=int, default=201)
    parser.add_argument(
        "--null-permutations", type=int, default=50,
        help=(
            "Monte-Carlo label-null repetitions for a transparent chance PR curve; "
            "0 disables it. The paper shows chance but does not publish its implementation."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--match-pre-ms", type=float, default=0.0,
                        help="Secondary tolerance before acoustic onset; paper-exact default is 0.")
    parser.add_argument("--match-post-ms", type=float, default=0.0,
                        help="Secondary tolerance after acoustic offset; paper-exact default is 0.")
    parser.add_argument("--limit-seconds", type=float, default=0.0,
                        help="Crop each file for a smoke-test; 0 means the full recording.")
    parser.add_argument("--debug", action="store_true",
                        help="Use only the first file of each requested split.")
    parser.add_argument("--preflight", action="store_true",
                        help="Resolve data/checkpoint paths and exit before loading models.")
    parser.add_argument("--no-save-timelines", action="store_true")
    parser.add_argument("--allow-nonzero-lead", action="store_true",
                        help="By default abort if a checkpoint architecture has ECOG_LEAD_MS != 0.")
    return parser.parse_args()


def load_base_api(base: Path, seed: int) -> SimpleNamespace:
    base = base.resolve()
    if not (base / "library" / "patients.json").is_file():
        raise FileNotFoundError(f"Base project not found or incomplete: {base}")
    os.environ["BENCH_SEED"] = str(seed)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.path.insert(0, str(base))

    import h5py
    import numpy as np
    import sklearn.preprocessing
    import torch
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    from library import bench_models_regression as bmr
    from library import models_classification as mcls
    from library.runner_classification import (
        HIDDEN_STRIDE,
        get_words_filepath,
        load_words_info,
        predict_regression_hidden,
    )
    from library.runner_common import WORDS_REMAP
    from library.runtime import DEVICE, device_str, make_split, set_seed, shift_ecog_lead

    return SimpleNamespace(
        base=base,
        h5py=h5py,
        np=np,
        sklearn=sklearn,
        torch=torch,
        uniform_filter1d=uniform_filter1d,
        find_peaks=find_peaks,
        bmr=bmr,
        mcls=mcls,
        HIDDEN_STRIDE=HIDDEN_STRIDE,
        get_words_filepath=get_words_filepath,
        load_words_info=load_words_info,
        predict_regression_hidden=predict_regression_hidden,
        WORDS_REMAP=WORDS_REMAP,
        DEVICE=DEVICE,
        device_str=device_str,
        make_split=make_split,
        set_seed=set_seed,
        shift_ecog_lead=shift_ecog_lead,
    )


def load_patient(base: Path, patient_name: str) -> dict:
    patients = json.loads((base / "library" / "patients.json").read_text(encoding="utf-8"))
    for patient in patients:
        if patient["name"] == patient_name:
            return patient
    raise KeyError(f"Patient not found: {patient_name}")


def model_name(patient_name: str, layer: int) -> str:
    channels = "8_16" if patient_name == "ivanova" else "6_12"
    return (
        f"SimpleNetBase_WithLSTM__CNANNELS_{channels}__LAG_1000_0"
        f"__WHISPER_BASE_L{layer}"
    )


def safe_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def resolve_model_spec(base: Path, patient_name: str, layer: int, seed: int) -> ModelSpec:
    name = model_name(patient_name, layer)
    pattern = str(base / "results" / f"classification_hidden___{patient_name}___{name}___*.json")
    candidates: List[Tuple[str, ModelSpec]] = []
    for filename in glob.glob(pattern):
        result_path = Path(filename)
        result = safe_json(result_path)
        if not result:
            continue
        config = result.get("config", {})
        if config.get("seed") != seed:
            continue
        if config.get("regression_model") != name:
            continue
        if config.get("control", "none") != "none" or config.get("augment", "none") != "none":
            continue
        date = result_path.stem.rsplit("___", 1)[-1]
        regression_path = base / "model_dumps" / f"regression___{patient_name}___{name}___{date}.pth"
        classifier_path = base / "model_dumps" / f"classification_hidden___{patient_name}___{name}___{date}.pth"
        if not regression_path.is_file() or not classifier_path.is_file():
            continue
        max_words_length = int(config.get("max_words_length") or 0)
        if max_words_length < 10:
            continue
        sync_acc = result.get("test_accuracy_full")
        if sync_acc is None:
            sync_acc = result.get("test_accuracy")
        spec = ModelSpec(
            layer=layer,
            model_name=name,
            date=date,
            seed=seed,
            max_words_length=max_words_length,
            regression_path=regression_path,
            classifier_path=classifier_path,
            result_path=result_path,
            synchronous_accuracy=float(sync_acc) if sync_acc is not None else None,
        )
        # ISO-like timestamps in filenames are a more stable provenance key
        # than filesystem mtime, which changes when a checkpoint is copied.
        candidates.append((date, spec))
    if not candidates:
        raise FileNotFoundError(
            f"No complete regression+hidden pair for patient={patient_name}, L{layer}, seed={seed}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def torch_load_weights(torch_module, path: Path, device):
    try:
        return torch_module.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location=device)


def load_bundle(api: SimpleNamespace, patient: dict, spec: ModelSpec,
                allow_nonzero_lead: bool) -> ModelBundle:
    # The upstream benchmark wrapper creates an optimizer and TensorBoard
    # SummaryWriter in its constructor. Neither is needed for frozen inference.
    # Temporarily replace those hooks so even a direct invocation from the base
    # directory cannot write a runs/ tree there. The source files stay untouched.
    wrapper_base = api.bmr.BenchModelRegressionBase
    original_optimizer = wrapper_base.create_optimizer
    original_logger = wrapper_base.create_logger
    try:
        wrapper_base.create_optimizer = lambda self: setattr(self, "optimizer", None)
        wrapper_base.create_logger = lambda self: setattr(self, "logger", None)
        regression = getattr(api.bmr, spec.model_name)(patient)
    finally:
        wrapper_base.create_optimizer = original_optimizer
        wrapper_base.create_logger = original_logger
    lead_ms = float(getattr(regression, "ECOG_LEAD_MS", 0.0))
    if lead_ms != 0.0 and not allow_nonzero_lead:
        raise RuntimeError(
            f"{spec.model_name} has ECOG_LEAD_MS={lead_ms}; this experiment requires no shift."
        )
    regression.model.load_state_dict(torch_load_weights(api.torch, spec.regression_path, api.DEVICE))
    regression.model.eval()
    hidden_dim = int(regression.model.final_out_features)
    classifier = api.mcls.Mel2WordHidden(hidden_dim, len(api.WORDS_REMAP)).to(api.DEVICE)
    classifier.load_state_dict(torch_load_weights(api.torch, spec.classifier_path, api.DEVICE))
    classifier.eval()
    return ModelBundle(spec=spec, regression=regression, classifier=classifier)


def validate_checkpoint_splits(api: SimpleNamespace, patient: dict,
                               specs: Sequence[ModelSpec]) -> None:
    train_idx, val_idx, test_idx = api.make_split(
        len(patient["files_list"]), patient["test_start_file_classification_index"]
    )
    expected = {
        "train_files": list(train_idx),
        "val_files": list(val_idx),
        "test_files": list(test_idx),
    }
    for spec in specs:
        result = safe_json(spec.result_path) or {}
        actual = (result.get("config") or {}).get("split")
        if actual is None:
            raise RuntimeError(f"Checkpoint result has no split provenance: {spec.result_path}")
        normalized = {key: [int(item) for item in actual.get(key, [])] for key in expected}
        if normalized != expected:
            raise RuntimeError(
                f"Checkpoint split mismatch for L{spec.layer}: {normalized} != {expected}"
            )


def preflight(api: SimpleNamespace, patient: dict, specs: Sequence[ModelSpec], splits: Sequence[str]) -> None:
    train_idx, val_idx, test_idx = api.make_split(
        len(patient["files_list"]), patient["test_start_file_classification_index"]
    )
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    print(f"[runtime] {api.device_str()}")
    print(f"[base] {api.base}")
    print(f"[patient] {patient['name']} | raw_fs={patient['sampling_rate']} Hz")
    for split in splits:
        print(f"[{split}] files={split_map[split]}")
        for index in split_map[split]:
            path = Path(patient["files_list"][index])
            words_path = Path(str(path.with_suffix("")) + "_words.txt")
            print(f"  {index}: data={path.is_file()} words={words_path.is_file()} {path.name}")
    for spec in specs:
        print(
            f"[L{spec.layer}] seed={spec.seed} date={spec.date} mwl={spec.max_words_length} "
            f"reg={spec.regression_path.stat().st_size / 2**20:.1f}MiB "
            f"clf={spec.classifier_path.stat().st_size / 2**20:.1f}MiB"
        )


def scale_hidden_like_training(api: SimpleNamespace, hidden) -> Tuple[object, dict]:
    np = api.np
    std64 = np.std(hidden.astype(np.float64, copy=False), axis=0)
    diag = {
        "hidden_shape": [int(v) for v in hidden.shape],
        "std_min": float(std64.min(initial=np.inf)),
        "std_median": float(np.median(std64)),
        "near_constant_lt_1e-8": int((std64 < 1e-8).sum()),
        "near_constant_lt_1e-6": int((std64 < 1e-6).sum()),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaled = api.sklearn.preprocessing.scale(hidden, copy=False)
    diag["sklearn_warnings"] = [str(item.message) for item in caught]
    diag["finite_after_scale"] = bool(np.isfinite(scaled).all())
    if not diag["finite_after_scale"]:
        raise FloatingPointError("Non-finite values appeared while scaling hidden features")
    return scaled, diag


def ground_truth_for_file(api: SimpleNamespace, words_info: Sequence[Sequence[object]],
                          file_index: int, raw_fs: float, raw_samples: int,
                          match_pre_ms: float, match_post_ms: float) -> List[GroundTruth]:
    gt: List[GroundTruth] = []
    pre_s = match_pre_ms / 1000.0
    post_s = match_post_ms / 1000.0
    for event_index, (start, end, word) in enumerate(words_info):
        if int(end) > raw_samples:
            continue
        gt.append(
            GroundTruth(
                file_index=file_index,
                event_index=event_index,
                start_s=max(0.0, int(start) / raw_fs - pre_s),
                end_s=min(raw_samples / raw_fs, int(end) / raw_fs + post_s),
                class_index=int(api.WORDS_REMAP[word]),
                word=str(word),
            )
        )
    return gt


def infer_sliding_probabilities(api: SimpleNamespace, hidden, bundle: ModelBundle,
                                batch_size: int, step_frames: int) -> Tuple[object, object, dict]:
    np = api.np
    torch = api.torch
    mwl = bundle.spec.max_words_length
    first_valid_hidden = int(math.ceil(bundle.regression.LAG_BACKWARD / api.HIDDEN_STRIDE))
    first_end = first_valid_hidden + mwl - 1
    endpoints = np.arange(first_end, hidden.shape[0], step_frames, dtype=np.int64)
    if endpoints.size == 0:
        raise RuntimeError(
            f"Recording is too short: hidden={hidden.shape[0]}, first async endpoint={first_end}"
        )
    probabilities = np.empty((len(endpoints), len(api.WORDS_REMAP)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(endpoints), batch_size):
            current = endpoints[start:start + batch_size]
            windows = np.empty(
                (len(current), hidden.shape[1], mwl), dtype=np.float32
            )
            for row, endpoint in enumerate(current):
                windows[row] = hidden[endpoint - mwl + 1:endpoint + 1].T
            logits = bundle.classifier(torch.from_numpy(windows).to(api.DEVICE))
            probabilities[start:start + len(current)] = torch.softmax(logits, dim=1).cpu().numpy()
    return endpoints, probabilities, {
        "max_words_length_frames": int(mwl),
        "first_valid_hidden_frame": int(first_valid_hidden),
        "first_async_endpoint_frame": int(first_end),
        "n_async_windows": int(len(endpoints)),
    }


def smooth_probabilities(api: SimpleNamespace, probabilities, width: int, mode: str):
    np = api.np
    width = max(1, int(width))
    if width == 1:
        return probabilities.copy()
    if mode == "centered":
        return api.uniform_filter1d(probabilities, size=width, axis=0, mode="nearest")
    # Strictly trailing boxcar for an optional causal comparison.
    cumulative = np.vstack(
        [np.zeros((1, probabilities.shape[1]), dtype=np.float64),
         np.cumsum(probabilities, axis=0, dtype=np.float64)]
    )
    result = np.empty_like(probabilities)
    for index in range(len(probabilities)):
        left = max(0, index - width + 1)
        result[index] = (cumulative[index + 1] - cumulative[left]) / (index - left + 1)
    return result


def locate_gt(time_s: float, ground_truth: Sequence[GroundTruth]) -> Optional[GroundTruth]:
    if not ground_truth:
        return None
    starts = [item.start_s for item in ground_truth]
    index = bisect.bisect_right(starts, time_s) - 1
    if index >= 0 and time_s <= ground_truth[index].end_s:
        return ground_truth[index]
    return None


def extract_candidates(api: SimpleNamespace, profile: FileProfile, system: str) -> List[Candidate]:
    np = api.np
    smoothed = profile.smoothed[system]
    winners = smoothed.argmax(axis=1)
    candidates: List[Candidate] = []
    # Class zero is silence/background and suppresses word emission.
    for class_index in range(1, smoothed.shape[1]):
        peaks, _ = api.find_peaks(smoothed[:, class_index])
        for peak in peaks:
            if winners[peak] != class_index:
                continue
            time_s = float(profile.times_s[peak])
            target = locate_gt(time_s, profile.ground_truth)
            candidates.append(
                Candidate(
                    file_index=profile.file_index,
                    time_s=time_s,
                    class_index=class_index,
                    score=float(smoothed[peak, class_index]),
                    gt_event_index=target.event_index if target else None,
                    gt_class_index=target.class_index if target else None,
                )
            )
    candidates.sort(key=lambda item: (item.file_index, item.time_s, -item.score))
    return candidates


def score_at_threshold(candidates: Sequence[Candidate], profiles: Sequence[FileProfile],
                       threshold: float) -> dict:
    selected = [item for item in candidates if item.score >= threshold]
    matched: set[Tuple[int, int]] = set()
    gt_lookup = {
        (gt.file_index, gt.event_index): gt
        for profile in profiles for gt in profile.ground_truth
    }
    tp = fp = substitutions = duplicates = insertions = 0
    latencies_ms: List[float] = []
    for candidate in selected:
        if candidate.gt_event_index is None:
            fp += 1
            insertions += 1
            continue
        key = (candidate.file_index, candidate.gt_event_index)
        if candidate.gt_class_index != candidate.class_index:
            fp += 1
            substitutions += 1
            continue
        if key in matched:
            fp += 1
            duplicates += 1
            continue
        matched.add(key)
        tp += 1
        latencies_ms.append((candidate.time_s - gt_lookup[key].start_s) * 1000.0)

    n_events = len(gt_lookup)
    fn = n_events - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / n_events if n_events else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_minutes = sum(profile.duration_s for profile in profiles) / 60.0
    word_seconds = sum(
        max(0.0, gt.end_s - gt.start_s) for profile in profiles for gt in profile.ground_truth
    )
    background_minutes = max(0.0, sum(p.duration_s for p in profiles) - word_seconds) / 60.0
    latency = {
        "n": len(latencies_ms),
        "median_ms": float("nan"),
        "q25_ms": float("nan"),
        "q75_ms": float("nan"),
        "p90_ms": float("nan"),
    }
    if latencies_ms:
        import numpy as np
        latency.update({
            "median_ms": float(np.median(latencies_ms)),
            "q25_ms": float(np.percentile(latencies_ms, 25)),
            "q75_ms": float(np.percentile(latencies_ms, 75)),
            "p90_ms": float(np.percentile(latencies_ms, 90)),
        })
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "substitutions": int(substitutions),
        "duplicate_detections": int(duplicates),
        "background_insertions": int(insertions),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_events_per_min": float(fp / total_minutes) if total_minutes else float("nan"),
        "background_insertions_per_min": (
            float(insertions / background_minutes) if background_minutes else float("nan")
        ),
        "event_error_rate": float((fp + fn) / n_events) if n_events else float("nan"),
        "latency": latency,
    }


def label_null_pr_curve(api: SimpleNamespace, candidates: Sequence[Candidate],
                        profiles: Sequence[FileProfile], thresholds,
                        permutations: int, random_seed: int) -> Optional[dict]:
    """Chance control with observed peak times/scores and random word labels.

    Figure 13 of the paper shows a chance curve but neither the article nor its
    public repository specifies how it was generated. This explicit null keeps
    the detector's event rate, timing and confidence distribution fixed, while
    replacing every emitted word with an independent uniform draw from the 26
    non-silence classes. It therefore tests lexical information at the observed
    temporal candidate locations without pretending to reproduce unpublished
    code.
    """
    if permutations <= 0:
        return None
    np = api.np
    n_events = int(sum(len(item.ground_truth) for item in profiles))
    rng = np.random.default_rng(int(random_seed))
    scores = np.asarray([item.score for item in candidates], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    n_classes = len(api.WORDS_REMAP) - 1
    precision_runs = np.empty((permutations, len(thresholds)), dtype=np.float64)
    recall_runs = np.empty_like(precision_runs)

    for permutation in range(permutations):
        random_classes = rng.integers(1, n_classes + 1, size=len(candidates))
        matched: set[Tuple[int, int]] = set()
        tp = fp = cursor = 0
        for threshold_index, threshold in enumerate(thresholds):
            while cursor < len(order) and scores[order[cursor]] >= threshold:
                candidate_index = int(order[cursor])
                candidate = candidates[candidate_index]
                predicted_class = int(random_classes[candidate_index])
                if candidate.gt_event_index is None:
                    fp += 1
                elif candidate.gt_class_index != predicted_class:
                    fp += 1
                else:
                    key = (candidate.file_index, candidate.gt_event_index)
                    if key in matched:
                        fp += 1
                    else:
                        matched.add(key)
                        tp += 1
                cursor += 1
            precision_runs[permutation, threshold_index] = tp / (tp + fp) if tp + fp else 1.0
            recall_runs[permutation, threshold_index] = tp / n_events if n_events else 0.0

    return {
        "method": (
            "Observed candidate peak times and scores are fixed; each candidate word label "
            "is redrawn uniformly from the 26 non-silence classes."
        ),
        "paper_exact_implementation_known": False,
        "permutations": int(permutations),
        "random_seed": int(random_seed),
        "threshold": [float(item) for item in thresholds],
        "precision_mean": [float(item) for item in precision_runs.mean(axis=0)],
        "precision_p05": [float(item) for item in np.quantile(precision_runs, 0.05, axis=0)],
        "precision_p95": [float(item) for item in np.quantile(precision_runs, 0.95, axis=0)],
        "recall_mean": [float(item) for item in recall_runs.mean(axis=0)],
    }


def event_pr_curve(api: SimpleNamespace, profiles: Sequence[FileProfile], system: str,
                   threshold_points: int, null_permutations: int = 0,
                   null_seed: int = 0) -> dict:
    np = api.np
    candidates = []
    for profile in profiles:
        candidates.extend(extract_candidates(api, profile, system))
    thresholds = np.linspace(1.0, 0.0, threshold_points)
    points = [score_at_threshold(candidates, profiles, float(value)) for value in thresholds]
    best = max(points, key=lambda point: (point["f1"], point["threshold"]))

    # Descriptive area under the event PR envelope. The complete curve remains
    # the primary paper-compatible result.
    by_recall: Dict[float, float] = {}
    for point in points:
        recall = float(point["recall"])
        by_recall[recall] = max(by_recall.get(recall, 0.0), float(point["precision"]))
    recall_values = np.array(sorted(by_recall), dtype=np.float64)
    precision_values = np.array([by_recall[value] for value in recall_values], dtype=np.float64)
    if len(precision_values):
        precision_values = np.maximum.accumulate(precision_values[::-1])[::-1]
    pr_auc = float(np.trapz(precision_values, recall_values)) if len(recall_values) > 1 else 0.0
    return {
        "system": system,
        "n_candidates_before_threshold": int(len(candidates)),
        "n_ground_truth_events": int(sum(len(item.ground_truth) for item in profiles)),
        "pr_auc_envelope": pr_auc,
        "best_f1_posthoc": best,
        "curve": {
            "threshold": [float(point["threshold"]) for point in points],
            "precision": [float(point["precision"]) for point in points],
            "recall": [float(point["recall"]) for point in points],
            "f1": [float(point["f1"]) for point in points],
            "fp_per_min": [float(point["false_events_per_min"]) for point in points],
        },
        "chance_label_null": label_null_pr_curve(
            api, candidates, profiles, thresholds, null_permutations, null_seed
        ),
        "_candidates": candidates,
    }


def strip_private(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def save_timeline(api: SimpleNamespace, split: str, profile: FileProfile, seed: int,
                  layers: Sequence[int], timestamp: str, is_full_run: bool) -> Path:
    output_dir = OUTPUT_ROOT / "results"
    if not is_full_run:
        output_dir = output_dir / "smoke"
    output_dir = output_dir / "timelines" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "times_s": profile.times_s.astype("float32"),
        "gt_start_s": api.np.array([item.start_s for item in profile.ground_truth], dtype="float32"),
        "gt_end_s": api.np.array([item.end_s for item in profile.ground_truth], dtype="float32"),
        "gt_class": api.np.array([item.class_index for item in profile.ground_truth], dtype="int16"),
    }
    for system, values in profile.probabilities.items():
        payload[f"prob_{system}"] = values.astype("float32")
        payload[f"smooth_{system}"] = profile.smoothed[system].astype("float32")
    layer_tag = "_".join(str(item) for item in layers)
    path = output_dir / (
        f"timeline_{split}_{profile.file_index:02d}_seed{seed}_L{layer_tag}.npz"
    )
    api.np.savez_compressed(path, **payload)
    return path


def process_file(api: SimpleNamespace, patient: dict, bundles: Sequence[ModelBundle],
                 file_index: int, args: argparse.Namespace) -> FileProfile:
    np = api.np
    filepath = Path(patient["files_list"][file_index])
    raw_fs = float(patient["sampling_rate"])
    with api.h5py.File(filepath, "r") as handle:
        samples = handle["RawData"]["Samples"]
        raw_samples = int(samples.shape[0])
        if args.limit_seconds > 0:
            raw_samples = min(raw_samples, int(round(args.limit_seconds * raw_fs)))
        data = samples[:raw_samples]
    ecog = data[:, patient["ecog_channels"]].astype("double")
    words_info = api.load_words_info(api.get_words_filepath(str(filepath)))
    ground_truth = ground_truth_for_file(
        api, words_info, file_index, raw_fs, raw_samples,
        args.match_pre_ms, args.match_post_ms,
    )

    raw_probabilities: Dict[str, object] = {}
    times_s = None
    diagnostics: Dict[str, object] = {"layers": {}}
    expected_endpoints = None
    frame_hz = None
    for bundle in bundles:
        print(f"  [file {file_index}] L{bundle.spec.layer}: preprocess -> hidden -> sliding windows")
        regression = bundle.regression
        preprocessed = regression.preprocess_ecog(ecog, raw_fs).astype("float32")
        preprocessed = api.shift_ecog_lead(
            preprocessed,
            getattr(regression, "ECOG_LEAD_MS", 0),
            raw_fs / regression.downsampling_coef,
        )
        hidden = api.predict_regression_hidden(regression, preprocessed, api.HIDDEN_STRIDE)
        hidden, scale_diag = scale_hidden_like_training(api, hidden)
        endpoints, probabilities, window_diag = infer_sliding_probabilities(
            api, hidden, bundle, args.batch_size, args.step_frames
        )
        current_hz = raw_fs / regression.downsampling_coef / api.HIDDEN_STRIDE
        current_times = endpoints.astype(np.float64) / current_hz
        if expected_endpoints is None:
            expected_endpoints = endpoints
            times_s = current_times
            frame_hz = current_hz
        else:
            if not np.array_equal(expected_endpoints, endpoints):
                raise RuntimeError("Layer timelines are not aligned")
            if not np.allclose(times_s, current_times):
                raise RuntimeError("Layer timestamps are not aligned")
        raw_probabilities[f"L{bundle.spec.layer}"] = probabilities
        diagnostics["layers"][f"L{bundle.spec.layer}"] = {
            "scale": scale_diag,
            "window": window_diag,
            "hidden_frame_hz": float(current_hz),
        }

    assert times_s is not None and frame_hz is not None
    raw_probabilities["L3+L4+L5" if [b.spec.layer for b in bundles] == [3, 4, 5]
                      else "ensemble"] = np.mean(
        np.stack([raw_probabilities[f"L{bundle.spec.layer}"] for bundle in bundles], axis=0),
        axis=0,
    ).astype(np.float32)
    output_hz = frame_hz / args.step_frames
    smooth_width = max(1, int(round(args.smooth_ms / 1000.0 * output_hz)))
    smoothed = {
        name: smooth_probabilities(api, values, smooth_width, args.smoothing)
        for name, values in raw_probabilities.items()
    }
    diagnostics.update({
        "raw_samples": raw_samples,
        "duration_s": raw_samples / raw_fs,
        "output_hz": float(output_hz),
        "smoothing_frames": int(smooth_width),
        "n_ground_truth_words": len(ground_truth),
    })
    return FileProfile(
        file_index=file_index,
        filename=str(filepath),
        duration_s=raw_samples / raw_fs,
        times_s=times_s,
        probabilities=raw_probabilities,
        smoothed=smoothed,
        ground_truth=ground_truth,
        diagnostics=diagnostics,
    )


def plot_pr(api: SimpleNamespace, test_results: Dict[str, dict], val_threshold_points: Dict[str, dict],
            patient_name: str, seed: int, layers: Sequence[int], timestamp: str,
            smooth_ms: float, smoothing: str, is_full_run: bool) -> Optional[Path]:
    if not test_results:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = OUTPUT_ROOT / "plots"
    if not is_full_run:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ensemble_name = "L3+L4+L5" if list(layers) == [3, 4, 5] else "ensemble"
    systems = [f"L{layer}" for layer in layers] + [ensemble_name]
    for system in systems:
        result = test_results[system]
        curve = result["curve"]
        linewidth = 2.8 if system == ensemble_name else 1.4
        ax.plot(curve["recall"], curve["precision"], label=system, linewidth=linewidth)
        if system in val_threshold_points:
            point = val_threshold_points[system]
            ax.scatter([point["recall"]], [point["precision"]], s=48, zorder=5)
    chance = test_results.get(ensemble_name, {}).get("chance_label_null")
    if chance:
        ax.plot(
            chance["recall_mean"], chance["precision_mean"],
            color="black", linestyle="--", linewidth=1.6,
            label=f"label-null ({chance['permutations']}x)",
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Event recall", ylabel="Event precision")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(
        f"Offline asynchronous replay | {patient_name} | seed={seed}\n"
        f"trailing window, {smooth_ms:g} ms {smoothing} smoothing, no boundary cues at inference"
    )
    fig.tight_layout()
    layer_tag = "_".join(str(item) for item in layers)
    output = output_dir / f"async_pr_{patient_name}_seed{seed}_L{layer_tag}_{timestamp}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def synchronous_reference(base: Path, patient_name: str, seed: int,
                          layers: Sequence[int]) -> Optional[dict]:
    layer_tag = "_".join(str(item) for item in layers)
    pattern = str(base / "results" / f"ensemble___{patient_name}___seed_{seed}___L{layer_tag}___*.json")
    files = [Path(item) for item in glob.glob(pattern)]
    for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True):
        result = safe_json(path)
        if not result or result.get("n_test") != 445 and patient_name == "ivanova":
            continue
        fixed_name = "+".join(f"L{layer}" for layer in layers)
        fixed = next((item for item in result.get("ensembles", []) if item.get("layers") == fixed_name), None)
        if fixed:
            return {"accuracy": fixed.get("acc"), "n_test": result.get("n_test"), "file": str(path)}
    return None


def json_ready(value):
    """Convert numpy/path values and replace non-finite floats with null."""
    import numpy as np
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    if (args.step_frames < 1 or args.batch_size < 1 or args.threshold_points < 2
            or args.null_permutations < 0):
        raise ValueError(
            "step-frames and batch-size must be positive; threshold-points must be >= 2; "
            "null-permutations must be >= 0"
        )
    layers = sorted(set(int(item) for item in args.layers))
    if any(layer not in (3, 4, 5) for layer in layers):
        raise ValueError("This fixed experiment supports Whisper layers 3, 4 and 5")

    base = args.base_project.resolve()
    api = load_base_api(base, args.seed)
    api.set_seed(args.seed)
    patient = load_patient(base, args.patient)
    specs = [resolve_model_spec(base, args.patient, layer, args.seed) for layer in layers]
    validate_checkpoint_splits(api, patient, specs)
    if len({item.max_words_length for item in specs}) != 1:
        raise RuntimeError("Layer classifiers use different max_words_length values")

    preflight(api, patient, specs, args.splits)
    if args.preflight:
        print("[preflight] OK; source project was read only")
        return 0

    bundles = [load_bundle(api, patient, spec, args.allow_nonzero_lead) for spec in specs]
    train_idx, val_idx, test_idx = api.make_split(
        len(patient["files_list"]), patient["test_start_file_classification_index"]
    )
    split_indices = {"val": val_idx, "test": test_idx}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    is_full_run = not args.debug and args.limit_seconds <= 0
    profiles_by_split: Dict[str, List[FileProfile]] = {}
    timeline_paths: List[str] = []
    for split in args.splits:
        indices = list(split_indices[split])
        if args.debug:
            indices = indices[:1]
        profiles: List[FileProfile] = []
        print(f"[{split}] continuous files={indices}")
        for file_index in indices:
            profile = process_file(api, patient, bundles, file_index, args)
            profiles.append(profile)
            if not args.no_save_timelines:
                path = save_timeline(
                    api, split, profile, args.seed, layers, timestamp, is_full_run
                )
                timeline_paths.append(str(path))
                print(f"  [saved] {path}")
        profiles_by_split[split] = profiles

    ensemble_name = "L3+L4+L5" if layers == [3, 4, 5] else "ensemble"
    systems = [f"L{layer}" for layer in layers] + [ensemble_name]
    results_by_split: Dict[str, Dict[str, dict]] = {}
    private_results: Dict[str, Dict[str, dict]] = {}
    for split, profiles in profiles_by_split.items():
        private_results[split] = {}
        results_by_split[split] = {}
        for system in systems:
            null_seed = (
                args.seed * 1_000_003
                + (0 if split == "val" else 50_021)
                + sum((index + 1) * ord(char) for index, char in enumerate(system))
            )
            result = event_pr_curve(
                api, profiles, system, args.threshold_points,
                null_permutations=args.null_permutations,
                null_seed=null_seed,
            )
            private_results[split][system] = result
            results_by_split[split][system] = strip_private(result)
            best = result["best_f1_posthoc"]
            print(
                f"[{split} {system}] best-posthoc F1={best['f1']:.3f} "
                f"P={best['precision']:.3f} R={best['recall']:.3f} theta={best['threshold']:.3f}"
            )

    val_threshold_test_points: Dict[str, dict] = {}
    if "val" in private_results and "test" in profiles_by_split:
        for system in systems:
            threshold = private_results["val"][system]["best_f1_posthoc"]["threshold"]
            candidates = private_results["test"][system]["_candidates"]
            point = score_at_threshold(candidates, profiles_by_split["test"], threshold)
            val_threshold_test_points[system] = point
            results_by_split["test"][system]["operating_point_from_val"] = point
            print(
                f"[test {system} @ val theta={threshold:.3f}] "
                f"F1={point['f1']:.3f} P={point['precision']:.3f} R={point['recall']:.3f} "
                f"FP/min={point['false_events_per_min']:.2f}"
            )

    plot_path = plot_pr(
        api,
        results_by_split.get("test", {}),
        val_threshold_test_points,
        args.patient,
        args.seed,
        layers,
        timestamp,
        args.smooth_ms,
        args.smoothing,
        is_full_run,
    )
    output_dir = OUTPUT_ROOT / "results"
    if not is_full_run:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = "_".join(str(item) for item in layers)
    output_path = output_dir / f"async_{args.patient}_seed{args.seed}_L{layer_tag}_{timestamp}.json"
    result = {
        "kind": "offline_asynchronous_word_replay",
        "article": {
            "citation": "Petrosyan et al., Journal of Neural Engineering 19 (2022) 066016",
            "doi": ARTICLE_DOI,
            "implemented_sections": ["3.3.2", "3.5", "4.5"],
        },
        "created": timestamp,
        "run": {
            "is_full_run": is_full_run,
            "debug_first_file_only": bool(args.debug),
            "limit_seconds_per_file": float(args.limit_seconds),
            "requested_splits": list(args.splits),
        },
        "base_project_read_only": str(base),
        "patient": args.patient,
        "seed": args.seed,
        "layers": layers,
        "protocol": {
            "regression_lag_backward_samples": int(bundles[0].regression.LAG_BACKWARD),
            "regression_lag_forward_samples": int(bundles[0].regression.LAG_FORWARD),
            "ecog_lead_ms": float(getattr(bundles[0].regression, "ECOG_LEAD_MS", 0.0)),
            "word_window_hidden_frames": specs[0].max_words_length,
            "hidden_stride": int(api.HIDDEN_STRIDE),
            "async_step_frames": args.step_frames,
            "smooth_ms": args.smooth_ms,
            "smoothing": args.smoothing,
            "threshold_points": args.threshold_points,
            "null_permutations": args.null_permutations,
            "match_pre_ms": args.match_pre_ms,
            "match_post_ms": args.match_post_ms,
            "word_boundaries_used_for_inference": False,
            "word_boundaries_used_for_scoring_only": True,
            "silence_emits_an_event": False,
            "duplicate_correct_peaks_count_as_fp": True,
        },
        "model_pairs": [
            {
                "layer": spec.layer,
                "model": spec.model_name,
                "date": spec.date,
                "seed": spec.seed,
                "max_words_length": spec.max_words_length,
                "regression_checkpoint": str(spec.regression_path),
                "classifier_checkpoint": str(spec.classifier_path),
                "classification_result": str(spec.result_path),
                "synchronous_single_layer_accuracy_from_json": spec.synchronous_accuracy,
            }
            for spec in specs
        ],
        "synchronous_fixed_ensemble_reference": synchronous_reference(
            base, args.patient, args.seed, layers
        ),
        "splits": {
            split: {
                "files": [
                    {
                        "index": profile.file_index,
                        "filename": profile.filename,
                        "duration_s": profile.duration_s,
                        "diagnostics": profile.diagnostics,
                    }
                    for profile in profiles_by_split[split]
                ],
                "systems": results_by_split[split],
            }
            for split in profiles_by_split
        },
        "timeline_files": timeline_paths,
        "plot": str(plot_path) if plot_path else None,
        "caveats": [
            "Offline asynchronous: inference receives no word boundaries, but base preprocessing uses zero-phase filters.",
            "ECoG and hidden features are standardized using whole-file statistics, so this is not strict online causality.",
            (
                f"Centered {args.smooth_ms:g} ms smoothing uses about "
                f"{args.smooth_ms / 2:g} ms of future probability samples; use causal "
                "smoothing for an online-like secondary analysis."
                if args.smoothing == "centered"
                else "Causal trailing smoothing uses no future probability samples."
            ),
            "The word head was trained synchronously on aligned/padded word segments; sliding replay is a distribution shift.",
            "Fixed phrase order can leak previous-word and phrase-position information into a 1.5 s effective context.",
            "Test PR and best-test-F1 are descriptive; the operating point selected on validation is the primary fixed threshold result.",
            "The article and public GitHub repository do not define their chance-curve implementation; this run uses the explicit label-null described in chance_label_null.",
        ],
    }
    output_path.write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[saved] {output_path}")
    if plot_path:
        print(f"[saved] {plot_path}")
    print("[done] source project was imported/read only; all new artifacts are in the async folder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
