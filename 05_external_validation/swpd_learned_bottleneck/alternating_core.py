"""Deterministic constrained alternating decoder/projector optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from core import component_metrics, fit_linear


@dataclass(frozen=True)
class AlternatingConfig:
    dimension: int = 50
    maximum_iterations: int = 25
    patience: int = 5
    seed: int = 42

    def validate(self) -> None:
        if self.dimension <= 0 or self.maximum_iterations < 0 or self.patience <= 0:
            raise ValueError("Invalid alternating configuration")


def polar_orthonormal(cross_covariance: np.ndarray) -> np.ndarray:
    """Return the closest column-orthonormal matrix to a D x K matrix."""

    values = np.asarray(cross_covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < values.shape[1]:
        raise ValueError("Polar update requires a tall two-dimensional matrix")
    u, _, vt = np.linalg.svd(values, full_matrices=False)
    result = u @ vt
    error = np.linalg.norm(result.T @ result - np.eye(result.shape[1]), ord="fro")
    if error > 1e-8:
        raise RuntimeError("Alternating projector orthogonality failed")
    return result


def fit_alternating(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    train_mel: np.ndarray,
    validation_mel: np.ndarray,
    config: AlternatingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Alternate exact linear decoding and an orthogonal Procrustes target update.

    Iteration zero is the train-only PCA initialization. A fixed projector first
    defines target scores and a fixed decoder is then fitted by OLS. With the
    decoder fixed, the target projector is updated by the polar factor of the
    train target/prediction cross-covariance. Selection uses validation MEL80
    correlation only; test data are not accepted by this function.
    """

    config.validate()
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    vx = np.asarray(validation_x, dtype=np.float64)
    vy = np.asarray(validation_y, dtype=np.float64)
    mel = np.asarray(train_mel, dtype=np.float64)
    vmel = np.asarray(validation_mel, dtype=np.float64)
    if x.shape[0] != y.shape[0] or vx.shape[0] != vy.shape[0]:
        raise ValueError("Neural and target rows must align")
    pca = PCA(
        n_components=config.dimension,
        whiten=False,
        svd_solver="full",
        random_state=config.seed,
    ).fit(y)
    projection = np.asarray(pca.components_.T, dtype=np.float64)
    best_score = -float("inf")
    best_iteration = -1
    best: dict[str, np.ndarray] | None = None
    history: list[dict[str, Any]] = []
    wait = 0
    for iteration in range(config.maximum_iterations + 1):
        train_scores = y @ projection
        validation_scores = vy @ projection
        decoder = fit_linear(x, train_scores)
        train_prediction = decoder.predict(x)
        validation_prediction = decoder.predict(vx)
        mel_probe = fit_linear(train_scores, mel)
        validation_mel_prediction = mel_probe.predict(validation_prediction)
        validation_metric = component_metrics(vmel, validation_mel_prediction)
        score = float(validation_metric["fisher_z_component_correlation"])
        improved = score > best_score + 1e-7
        if improved:
            best_score = score
            best_iteration = iteration
            wait = 0
            best = {
                "projection": projection.copy(),
                "decoder_coef": np.asarray(decoder.coef_, dtype=np.float64),
                "decoder_intercept": np.asarray(decoder.intercept_, dtype=np.float64),
                "mel_probe_coef": np.asarray(mel_probe.coef_, dtype=np.float64),
                "mel_probe_intercept": np.asarray(mel_probe.intercept_, dtype=np.float64),
            }
        else:
            wait += 1
        history.append(
            {
                "iteration": iteration,
                "validation_mel80_fisher_r": score,
                "validation_score50_fisher_r": component_metrics(
                    validation_scores, validation_prediction
                )["fisher_z_component_correlation"],
                "projector_change_frobenius": None,
                "best": improved,
            }
        )
        if iteration == config.maximum_iterations or wait >= config.patience:
            break
        updated = polar_orthonormal(y.T @ train_prediction)
        history[-1]["projector_change_frobenius"] = float(
            np.linalg.norm(updated - projection, ord="fro")
        )
        projection = updated
    if best is None:
        raise RuntimeError("Alternating optimization produced no valid model")
    return best, {
        "best_iteration": best_iteration,
        "best_validation_mel80_fisher_r": best_score,
        "iterations_completed": history[-1]["iteration"],
        "early_stopped": history[-1]["iteration"] < config.maximum_iterations,
        "history": history,
    }


def predict_linear(values: np.ndarray, coefficient: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) @ coefficient.T + intercept
