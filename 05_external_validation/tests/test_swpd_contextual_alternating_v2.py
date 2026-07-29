from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from swpd_contextual_alternating_v2.core import (  # noqa: E402
    TargetSearchSpace,
    exact_projector_update,
    mse,
    project_scores,
)


def _random_projector(rng: np.random.Generator, output: int, search: int) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(search, output)))
    return basis.T


def test_cycle_zero_is_exact_whitened_pca50_analogue() -> None:
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(500, 18)) @ np.diag(np.geomspace(0.05, 9.0, 18))
    space = TargetSearchSpace.fit(raw, search_dim=10, output_dim=4)
    actual = space.scores(raw, space.initial_projector())
    expected = PCA(n_components=4, whiten=True, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(raw)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_exact_update_cannot_increase_objective_on_anisotropic_raw_data() -> None:
    rng = np.random.default_rng(44)
    mixing = rng.normal(size=(24, 24))
    raw = rng.normal(size=(700, 24)) @ np.diag(np.geomspace(1e-3, 20.0, 24)) @ mixing
    space = TargetSearchSpace.fit(raw, search_dim=12, output_dim=5)
    whitened = space.transform(raw)
    previous = _random_projector(rng, 5, 12)
    prediction = rng.normal(size=(len(raw), 5))
    updated, receipt = exact_projector_update(whitened, prediction, previous)
    assert receipt["new_mse"] <= receipt["old_mse"] + 1e-12
    assert mse(project_scores(whitened, updated), prediction) <= mse(
        project_scores(whitened, previous), prediction
    ) + 1e-12
    np.testing.assert_allclose(updated @ updated.T, np.eye(5), atol=1e-10)
    np.testing.assert_allclose(
        np.cov(project_scores(whitened, updated), rowvar=False),
        np.eye(5),
        atol=1e-10,
    )


def test_stationary_projector_is_preserved_up_to_numerical_precision() -> None:
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(600, 16))
    space = TargetSearchSpace.fit(raw, search_dim=10, output_dim=4)
    whitened = space.transform(raw)
    previous = _random_projector(rng, 4, 10)
    prediction = project_scores(whitened, previous)
    updated, receipt = exact_projector_update(whitened, prediction, previous)
    np.testing.assert_allclose(updated, previous, rtol=1e-10, atol=1e-10)
    assert receipt["new_mse"] < 1e-20


def test_reconstruction_has_original_standardized_geometry() -> None:
    rng = np.random.default_rng(99)
    raw = rng.normal(size=(300, 14))
    space = TargetSearchSpace.fit(raw, search_dim=9, output_dim=3)
    projector = space.initial_projector()
    scores = space.scores(raw, projector)
    reconstructed = space.reconstruct_standardized(scores, projector)
    assert reconstructed.shape == raw.shape
    assert np.isfinite(reconstructed).all()
