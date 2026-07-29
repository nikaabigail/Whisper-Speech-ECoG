from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "swpd_learned_bottleneck"))

from core import (  # noqa: E402
    CachedBlock,
    fit_pca_bottleneck,
    fit_supervised_rrr_bottleneck,
    fold_indexes,
    select,
)
from clip_core import (  # noqa: E402
    ClipConfig,
    LinearClip,
    clip_loss,
    temporally_separated_indices,
    train_clip,
)
from alternating_core import AlternatingConfig, fit_alternating, polar_orthonormal  # noqa: E402


def test_srrr_projector_is_orthonormal_and_nonzero() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 8))
    y = x @ rng.normal(size=(8, 20)) + rng.normal(scale=0.1, size=(300, 20))
    model = fit_supervised_rrr_bottleneck(x, y, dimension=6)
    np.testing.assert_allclose(model.projection.T @ model.projection, np.eye(6), atol=1e-10)
    assert np.linalg.norm(model.projection) > 0
    transformed = model.transform(y)
    assert transformed.shape == (300, 6)
    assert np.isfinite(transformed).all()


def test_pca_inverse_is_a_projection_in_raw_space() -> None:
    rng = np.random.default_rng(8)
    y = rng.normal(size=(200, 12))
    model = fit_pca_bottleneck(y, dimension=5, seed=42)
    reconstructed = model.inverse_transform(model.transform(y))
    assert reconstructed.shape == y.shape
    assert np.isfinite(reconstructed).all()
    assert model.orthogonality_error() < 1e-10


def test_fold_split_is_disjoint_and_cyclic() -> None:
    blocks = []
    for index in range(5):
        ids = np.asarray([f"block-{index}-frame-{frame}" for frame in range(4)])
        blocks.append(
            CachedBlock(
                index,
                ids,
                np.arange(4) + index * 10,
                np.ones((4, 3)),
                {name: np.ones((4, dim)) for name, dim in {
                    "mel80": 80, "L3": 512, "L4": 512, "L5": 512, "L345": 1536
                }.items()},
            )
        )
    train, validation, test = fold_indexes(4)
    assert validation == 0
    # fold 4 wraps validation to block 0, leaving blocks 1,2,3 for training
    assert train == (1, 2, 3)
    assert test == 4
    train_ids, _, _ = select(blocks, train)
    val_ids, _, _ = select(blocks, (validation,))
    test_ids, _, _ = select(blocks, (test,))
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))


def test_invalid_fold_is_rejected() -> None:
    with pytest.raises(ValueError):
        fold_indexes(5)


def test_clip_negative_sampler_separates_same_block_frames() -> None:
    ids = []
    times = []
    for block in range(3):
        for frame in range(200):
            ids.append(f"sub-01:block-{block:02d}:frame-{frame:05d}")
            times.append(block * 60 + frame * 0.02)
    indexes = temporally_separated_indices(
        ids,
        np.asarray(times),
        minimum_separation_seconds=0.5,
        seed=4,
        epoch=1,
    )
    selected_ids = np.asarray(ids)[indexes]
    selected_times = np.asarray(times)[indexes]
    for block in range(3):
        mask = np.char.find(selected_ids.astype(str), f"block-{block:02d}") >= 0
        assert np.min(np.diff(np.sort(selected_times[mask]))) >= 0.5 - 1e-9


def test_clip_loss_is_finite_and_target_retraction_is_orthonormal() -> None:
    import torch

    rng = np.random.default_rng(9)
    target_projection = np.linalg.qr(rng.normal(size=(12, 5)))[0]
    model = LinearClip(
        5,
        12,
        5,
        target_initial_projection=target_projection,
        neural_initial_weight=np.eye(5),
        neural_initial_bias=np.zeros(5),
    )
    with torch.no_grad():
        model.target_projection.add_(0.05)
    model.retract_target_projection()
    weight = model.target_projection.detach().numpy()
    np.testing.assert_allclose(weight.T @ weight, np.eye(5), atol=1e-5)
    neural = torch.as_tensor(rng.normal(size=(32, 5)), dtype=torch.float32)
    target = torch.as_tensor(rng.normal(size=(32, 12)), dtype=torch.float32)
    loss, details = clip_loss(
        model.neural_embedding(neural),
        model.target_embedding(target),
        ClipConfig(dimension=5),
    )
    assert torch.isfinite(loss)
    assert np.isfinite(list(details.values())).all()


def test_clip_checkpoint_resumes_from_cpu_payload(tmp_path: Path) -> None:
    import torch

    rng = np.random.default_rng(10)
    neural = rng.normal(size=(200, 5)).astype(np.float32)
    target = np.concatenate(
        [neural, neural[:, :3] + rng.normal(scale=0.1, size=(200, 3))], axis=1
    ).astype(np.float32)
    ids = [f"sub-01:block-{i // 100:02d}:frame-{i % 100:05d}" for i in range(200)]
    times = np.asarray([(i // 100) * 10 + (i % 100) * 0.02 for i in range(200)])
    projection = np.linalg.qr(rng.normal(size=(8, 5)))[0]

    def make_model() -> LinearClip:
        return LinearClip(
            5,
            8,
            5,
            target_initial_projection=projection,
            neural_initial_weight=np.eye(5),
            neural_initial_bias=np.zeros(5),
        )

    config = ClipConfig(
        dimension=5,
        batch_size=8,
        maximum_epochs=2,
        patience=2,
        minimum_negative_separation_seconds=0.1,
        seed=4,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    first, first_info = train_clip(
        make_model(), neural, target, ids, times, neural, target, ids, times,
        config, torch.device("cpu"), checkpoint, "fixture"
    )
    resumed, resumed_info = train_clip(
        make_model(), neural, target, ids, times, neural, target, ids, times,
        config, torch.device("cpu"), checkpoint, "fixture"
    )
    assert first_info == resumed_info
    for left, right in zip(first.parameters(), resumed.parameters()):
        np.testing.assert_allclose(left.detach().numpy(), right.detach().numpy())


def test_alternating_projector_remains_orthonormal_and_selects_validation() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, 8))
    y = x @ rng.normal(size=(8, 20)) + rng.normal(scale=0.2, size=(300, 20))
    mel = y[:, :6] + rng.normal(scale=0.1, size=(300, 6))
    model, training = fit_alternating(
        x[:220], y[:220], x[220:], y[220:], mel[:220], mel[220:],
        AlternatingConfig(dimension=6, maximum_iterations=4, patience=2),
    )
    np.testing.assert_allclose(model["projection"].T @ model["projection"], np.eye(6), atol=1e-8)
    assert 0 <= training["best_iteration"] <= 4
    assert len(training["history"]) >= 1
    assert np.isfinite(training["best_validation_mel80_fisher_r"])


def test_polar_update_rejects_wide_matrix() -> None:
    with pytest.raises(ValueError):
        polar_orthonormal(np.ones((3, 5)))
