"""External validation package for the Ossadtchi Whisper-ECoG project."""

from .classifier import HiddenSequenceClassifier
from .ensemble import LayerProbabilities, fixed_l345_probability_ensemble
from .evaluation import (
    EvaluationResult,
    evaluate_hidden_classifier,
    evaluate_regression,
)
from .model import OneSecondEcogEncoder
from .neural_data import (
    ChannelStandardizerArtifact,
    FrameWindowDataset,
    fit_train_only_channel_standardizer,
)
from .protocol import (
    FivePairAssignment,
    SplitManifest,
    TestGate,
    TestGateAuthorization,
    TestGateClosed,
    make_swpd_fixed_neural_split,
    make_swpd_rotating_linear_splits,
    swpd_neural_pair_assignment,
)
from .reducer import ReducerArtifact, fit_train_only_reducer
from .reproducibility import set_deterministic_seed
from .targets import MelTargetExtractor, WhisperLayerTargetExtractor
from .training import (
    TrainingConfig,
    TrainingResult,
    model_state_fingerprint,
    train_hidden_classifier,
    train_regression,
)

__version__ = "0.1.0"

__all__ = [
    "ChannelStandardizerArtifact",
    "EvaluationResult",
    "FivePairAssignment",
    "FrameWindowDataset",
    "HiddenSequenceClassifier",
    "LayerProbabilities",
    "MelTargetExtractor",
    "OneSecondEcogEncoder",
    "ReducerArtifact",
    "SplitManifest",
    "TestGate",
    "TestGateAuthorization",
    "TestGateClosed",
    "TrainingConfig",
    "TrainingResult",
    "WhisperLayerTargetExtractor",
    "evaluate_hidden_classifier",
    "evaluate_regression",
    "fit_train_only_channel_standardizer",
    "fit_train_only_reducer",
    "fixed_l345_probability_ensemble",
    "make_swpd_fixed_neural_split",
    "make_swpd_rotating_linear_splits",
    "model_state_fingerprint",
    "set_deterministic_seed",
    "swpd_neural_pair_assignment",
    "train_hidden_classifier",
    "train_regression",
]
