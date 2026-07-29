from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from swpd_contextual_alternating_v2.core import (  # noqa: E402
    PCATransform,
    TargetSearchSpace,
    exact_projector_update,
    fit_affine,
    mse,
    project_scores,
)
from swpd_contextual_neural_e2e.core import (  # noqa: E402
    ContextualResidualDecoder,
    clone_state_dict_cpu,
    fold_legacy_pipeline,
    projector_receipt,
    state_dict_sha256,
)


def test_frozen_evaluator_enables_deterministic_cuda_runtime() -> None:
    source = (
        ROOT / "swpd_contextual_neural_e2e" / "evaluate_frozen_sub01.py"
    ).read_text(encoding="utf-8")
    assert "torch.backends.cudnn.deterministic = True" in source
    assert "torch.backends.cudnn.benchmark = False" in source
    assert "torch.use_deterministic_algorithms(True)" in source


def test_frozen_evaluator_keeps_cache_and_output_manifest_paths_distinct() -> None:
    source = (
        ROOT / "swpd_contextual_neural_e2e" / "evaluate_frozen_sub01.py"
    ).read_text(encoding="utf-8")
    assert 'artifact_manifest_path = run_dir / "artifact_manifest.json"' in source
    assert 'cache_manifest_path = cache / f"block_{fold:02d}.json"' in source
    assert not any(
        line.strip().startswith("manifest_path = cache /")
        for line in source.splitlines()
    )
    assert "MANIFEST_PATH_SHADOW_HOTFIX_KIND" in source


def test_folded_legacy_skip_exactly_matches_pca_then_ols() -> None:
    rng = np.random.default_rng(10)
    standardized = rng.normal(size=(400, 27))
    pca = PCATransform.fit(standardized, 8, whiten=False)
    scores = pca.transform(standardized)
    decoder = fit_affine(scores, rng.normal(size=(len(scores), 5)))
    folded = fold_legacy_pipeline(pca, decoder)
    np.testing.assert_allclose(
        folded.predict(standardized),
        decoder.predict(scores),
        rtol=1e-11,
        atol=1e-11,
    )


def test_contextual_decoder_starts_at_exact_legacy_affine_for_both_geometries() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(23, 5, 3)).astype(np.float32)
    weight = rng.normal(size=(4, 15)).astype(np.float32)
    bias = rng.normal(size=4).astype(np.float32)
    torch.manual_seed(7)
    model = ContextualResidualDecoder(
        context_steps=5,
        channels=3,
        output_dim=4,
        spatial_dim=6,
        recurrent_dim=5,
    )
    model.initialize_legacy_skip(weight, bias)
    expected = values.reshape(len(values), -1) @ weight.T + bias
    with torch.inference_mode():
        sequence_output = model(torch.from_numpy(values)).numpy()
        flat_output = model(torch.from_numpy(values.reshape(len(values), -1))).numpy()
    np.testing.assert_allclose(sequence_output, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(flat_output, expected, rtol=2e-6, atol=2e-6)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.architecture_receipt()["dropout"] == 0.0


def test_paired_initial_state_is_identical_and_seed_controls_residual_weights() -> None:
    weight = np.zeros((4, 15), dtype=np.float32)
    bias = np.arange(4, dtype=np.float32)
    states = []
    for _ in range(2):
        torch.manual_seed(42)
        model = ContextualResidualDecoder(5, 3, 4, 6, 5)
        model.initialize_legacy_skip(weight, bias)
        states.append(clone_state_dict_cpu(model))
    assert state_dict_sha256(states[0]) == state_dict_sha256(states[1])
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


def test_whitened_projector_update_is_noncollapsing_and_nonincreasing() -> None:
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(600, 18)) @ np.diag(np.geomspace(0.02, 8.0, 18))
    space = TargetSearchSpace.fit(raw, search_dim=12, output_dim=5)
    whitened = space.transform(raw)
    q0 = space.initial_projector()
    prediction = rng.normal(size=(len(raw), 5))
    updated, receipt = exact_projector_update(whitened, prediction, q0)
    assert receipt["new_mse"] <= receipt["old_mse"] + 1e-12
    assert mse(project_scores(whitened, updated), prediction) <= mse(
        project_scores(whitened, q0), prediction
    ) + 1e-12
    audit = projector_receipt(whitened, updated)
    assert audit["rank"] == 5
    assert audit["orthogonality_fro_error"] < 1e-10
    assert audit["zero_variance_collapse"] is False
    np.testing.assert_allclose(
        np.var(project_scores(whitened, updated), axis=0, ddof=1),
        np.ones(5),
        atol=1e-10,
    )
