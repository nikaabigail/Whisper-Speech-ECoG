"""Pure numerical core for exact contextual L4 alternating regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from swpd_protocol_bridge.bridge_core import component_metrics


@dataclass(frozen=True)
class AffineMap:
    """Portable multi-output affine map using sklearn's coefficient layout."""

    coef: np.ndarray
    intercept: np.ndarray

    def __post_init__(self) -> None:
        coef = np.asarray(self.coef, dtype=np.float64)
        intercept = np.asarray(self.intercept, dtype=np.float64)
        if coef.ndim != 2 or intercept.shape != (coef.shape[0],):
            raise ValueError("invalid affine-map geometry")
        if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
            raise ValueError("affine map contains non-finite values")

    def predict(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.coef.shape[1]:
            raise ValueError("affine-map input geometry changed")
        return data @ self.coef.T + self.intercept


def fit_affine(inputs: np.ndarray, targets: np.ndarray) -> AffineMap:
    model = LinearRegression(n_jobs=1).fit(
        np.asarray(inputs, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )
    return AffineMap(np.asarray(model.coef_), np.atleast_1d(model.intercept_))


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("invalid standardizer geometry")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("invalid standardizer values")

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        fitted = StandardScaler().fit(np.asarray(values, dtype=np.float64))
        return cls(np.asarray(fitted.mean_), np.asarray(fitted.scale_))

    def transform(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(self.mean):
            raise ValueError("standardizer input geometry changed")
        return (data - self.mean) / self.scale


@dataclass(frozen=True)
class PCATransform:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    whiten: bool

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        components = np.asarray(self.components, dtype=np.float64)
        variance = np.asarray(self.explained_variance, dtype=np.float64)
        if components.ndim != 2 or mean.shape != (components.shape[1],):
            raise ValueError("invalid PCA geometry")
        if variance.shape != (components.shape[0],) or np.any(variance <= 0):
            raise ValueError("invalid PCA variance")
        if any(not np.isfinite(item).all() for item in (mean, components, variance)):
            raise ValueError("PCA contains non-finite values")

    @classmethod
    def fit(cls, values: np.ndarray, components: int, *, whiten: bool) -> "PCATransform":
        model = PCA(n_components=int(components), whiten=whiten, svd_solver="full")
        model.fit(np.asarray(values, dtype=np.float64))
        return cls(
            np.asarray(model.mean_),
            np.asarray(model.components_),
            np.asarray(model.explained_variance_),
            bool(whiten),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.components.shape[1]:
            raise ValueError("PCA input geometry changed")
        result = (data - self.mean) @ self.components.T
        if self.whiten:
            result = result / np.sqrt(self.explained_variance)
        return result

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.components.shape[0]:
            raise ValueError("PCA score geometry changed")
        if self.whiten:
            values = values * np.sqrt(self.explained_variance)
        return values @ self.components + self.mean


@dataclass(frozen=True)
class TargetSearchSpace:
    scaler: Standardizer
    pca: PCATransform
    output_dim: int

    @classmethod
    def fit(
        cls,
        raw_train: np.ndarray,
        *,
        search_dim: int = 128,
        output_dim: int = 50,
        eigenvalue_ratio_floor: float = 1e-8,
    ) -> "TargetSearchSpace":
        if not output_dim <= search_dim <= raw_train.shape[1]:
            raise ValueError("target search dimensions are invalid")
        scaler = Standardizer.fit(raw_train)
        pca = PCATransform.fit(scaler.transform(raw_train), search_dim, whiten=True)
        ratio = float(pca.explained_variance[-1] / pca.explained_variance[0])
        if ratio < eigenvalue_ratio_floor:
            raise RuntimeError(
                f"target whitening is ill-conditioned: eigenvalue ratio {ratio:.3e}"
            )
        return cls(scaler, pca, int(output_dim))

    @property
    def search_dim(self) -> int:
        return int(self.pca.components.shape[0])

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(raw))

    def initial_projector(self) -> np.ndarray:
        projector = np.zeros((self.output_dim, self.search_dim), dtype=np.float64)
        projector[:, : self.output_dim] = np.eye(self.output_dim)
        return projector

    def scores(self, raw: np.ndarray, projector: np.ndarray) -> np.ndarray:
        return project_scores(self.transform(raw), projector)

    def reconstruct_standardized(self, scores: np.ndarray, projector: np.ndarray) -> np.ndarray:
        q = validate_projector(projector, self.output_dim, self.search_dim)
        search_scores = np.asarray(scores, dtype=np.float64) @ q
        return self.pca.inverse_transform(search_scores)


def validate_projector(projector: np.ndarray, output_dim: int, search_dim: int) -> np.ndarray:
    q = np.asarray(projector, dtype=np.float64)
    if q.shape != (output_dim, search_dim) or not np.isfinite(q).all():
        raise ValueError("projector geometry or values are invalid")
    error = float(np.linalg.norm(q @ q.T - np.eye(output_dim), ord="fro"))
    if error > 1e-8:
        raise ValueError(f"projector is not row-orthonormal: {error:.3e}")
    return q


def project_scores(whitened_targets: np.ndarray, projector: np.ndarray) -> np.ndarray:
    values = np.asarray(whitened_targets, dtype=np.float64)
    q = validate_projector(projector, projector.shape[0], values.shape[1])
    return values @ q.T


def mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("MSE arrays are not aligned")
    return float(np.mean(np.square(left - right)))


def covariance_error(whitened_train: np.ndarray) -> float:
    values = np.asarray(whitened_train, dtype=np.float64)
    covariance = np.cov(values, rowvar=False, ddof=1)
    return float(np.linalg.norm(covariance - np.eye(values.shape[1]), ord="fro"))


def exact_projector_update(
    whitened_train: np.ndarray,
    fixed_prediction: np.ndarray,
    previous_projector: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the rectangular Procrustes step in an isotropic target space."""

    targets = np.asarray(whitened_train, dtype=np.float64)
    prediction = np.asarray(fixed_prediction, dtype=np.float64)
    previous = validate_projector(
        previous_projector, prediction.shape[1], targets.shape[1]
    )
    if prediction.shape[0] != targets.shape[0]:
        raise ValueError("projector update rows are not aligned")
    whitening_error = covariance_error(targets)
    if whitening_error > 1e-6 * np.sqrt(targets.shape[1]):
        raise RuntimeError(f"target search space is not white: {whitening_error:.3e}")
    cross = targets.T @ prediction
    left, singular, right_t = np.linalg.svd(cross, full_matrices=False)
    if singular[-1] <= np.finfo(np.float64).eps * max(1.0, singular[0]) * max(cross.shape):
        raise RuntimeError("projector cross-covariance is rank deficient")
    updated = validate_projector((left @ right_t).T, prediction.shape[1], targets.shape[1])
    old_loss = mse(project_scores(targets, previous), prediction)
    new_loss = mse(project_scores(targets, updated), prediction)
    allowed = tolerance * max(1.0, abs(old_loss))
    if new_loss > old_loss + allowed:
        raise RuntimeError(
            f"exact projector step increased MSE: {old_loss:.12g} -> {new_loss:.12g}"
        )
    receipt = {
        "old_mse": old_loss,
        "new_mse": new_loss,
        "delta_mse": new_loss - old_loss,
        "whitening_fro_error": whitening_error,
        "projector_orthogonality_fro_error": float(
            np.linalg.norm(updated @ updated.T - np.eye(updated.shape[0]), ord="fro")
        ),
        "cross_singular_min": float(singular[-1]),
        "cross_singular_max": float(singular[0]),
    }
    return updated, receipt


def common_mel_metrics(
    probe: AffineMap,
    predicted_scores: np.ndarray,
    truth_mel_z: np.ndarray,
) -> dict[str, Any]:
    return component_metrics(
        np.asarray(truth_mel_z, dtype=np.float64),
        probe.predict(np.asarray(predicted_scores, dtype=np.float64)),
    )


def projected_variance_receipt(scores: np.ndarray) -> dict[str, float]:
    variance = np.var(np.asarray(scores, dtype=np.float64), axis=0, ddof=1)
    return {
        "minimum": float(variance.min()),
        "maximum": float(variance.max()),
        "mean": float(variance.mean()),
    }
