"""Pre-specified L3+L4+L5 probability ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


FIXED_LAYERS = (3, 4, 5)


@dataclass(frozen=True)
class LayerProbabilities:
    sample_ids: tuple[str, ...]
    probabilities: np.ndarray

    @classmethod
    def create(
        cls, sample_ids: Sequence[str], probabilities: np.ndarray
    ) -> "LayerProbabilities":
        ids = tuple(str(item) for item in sample_ids)
        values = np.asarray(probabilities, dtype=np.float64)
        if len(ids) != len(set(ids)):
            raise ValueError("sample IDs must be unique")
        if values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] < 2:
            raise ValueError("probabilities must have shape (sample_ids, at least 2 classes)")
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError("every probability row must sum to one")
        frozen = values.astype(np.float32)
        frozen.setflags(write=False)
        return cls(sample_ids=ids, probabilities=frozen)


@dataclass(frozen=True)
class EnsembleProbabilities:
    sample_ids: tuple[str, ...]
    probabilities: np.ndarray
    layers: tuple[int, int, int] = FIXED_LAYERS
    rule: str = "arithmetic_mean_of_probabilities"


def fixed_l345_probability_ensemble(
    layer_outputs: Mapping[int, LayerProbabilities],
) -> EnsembleProbabilities:
    layers = tuple(sorted(int(layer) for layer in layer_outputs))
    if layers != FIXED_LAYERS:
        raise ValueError(f"ensemble is pre-specified as {FIXED_LAYERS}; received {layers}")
    reference = layer_outputs[3]
    arrays = []
    for layer in FIXED_LAYERS:
        current = layer_outputs[layer]
        if current.sample_ids != reference.sample_ids:
            raise ValueError(
                f"L{layer} sample IDs/order differ from L3; refusing a misaligned ensemble"
            )
        if current.probabilities.shape != reference.probabilities.shape:
            raise ValueError("layer probability matrices have different shapes")
        arrays.append(np.asarray(current.probabilities, dtype=np.float64))
    averaged = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
    averaged.setflags(write=False)
    return EnsembleProbabilities(reference.sample_ids, averaged)
