"""Regression configurations used by the MEL/Whisper L3-L5 release paths.

The working project contains many historical LPC, MFCC, SSL, lead-shift and
channel-scan variants.  This release module deliberately keeps only the two
channel layouts used by the current MEL baseline and Whisper-base L3/L4/L5
experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import common_preprocessing, whisper_target
from .loggers import LearningLogStorer
from .models_regression import SimpleNet
from .runtime import DEVICE


class BenchModelRegressionBase:
    BATCH_SIZE = 100
    DOWNSAMPLED_APPROX_SAMPLING_RATE = 1000
    LEARNING_RATE = 0.0003
    HIGH_PASS_HZ = 10
    LOW_PASS_HZ = 200
    SELECTED_CHANNELS = None

    def __init__(self, patient):
        self.patient = patient
        self.downsampling_coef = round(
            self.patient["sampling_rate"] / self.DOWNSAMPLED_APPROX_SAMPLING_RATE
        )
        self.TEST_START_FILE_INDEX = self.patient["test_start_file_regression_index"]
        self.selected_channels = (
            self.SELECTED_CHANNELS
            if self.SELECTED_CHANNELS is not None
            else self.patient["ecog_channels"]
        )
        self.input_size = len(self.selected_channels)
        self.init()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.logger = LearningLogStorer(
            SummaryWriter(
                comment=(
                    f"___regression___{self.patient['name']}___"
                    f"{self.__class__.__name__}"
                )
            )
        )

    def init(self):
        raise NotImplementedError

    def preprocess_ecog(self, ecog, sampling_rate):
        raise NotImplementedError

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        raise NotImplementedError

    def detect_voice(self, y_batch):
        raise NotImplementedError


class SimpleNetBase(BenchModelRegressionBase):
    def init(self):
        self.model = SimpleNet(
            self.input_size,
            self.OUTPUT_SIZE,
            self.LAG_BACKWARD,
            self.LAG_FORWARD,
        ).to(DEVICE)

    def preprocess_ecog(self, ecog, sampling_rate):
        assert self.patient["sampling_rate"] == sampling_rate
        selected = (
            self.SELECTED_CHANNELS
            if self.SELECTED_CHANNELS is not None
            else self.patient["ecog_channels"]
        )
        processed = common_preprocessing.classic_ecog_pipeline(
            ecog,
            self.patient["sampling_rate"],
            self.downsampling_coef,
            self.LOW_PASS_HZ,
            self.HIGH_PASS_HZ,
        )
        return processed[:, selected]


class SimpleNetMels40Base(SimpleNetBase):
    N_MELS = 40
    F_MAX = 2000
    OUTPUT_SIZE = N_MELS
    LAG_BACKWARD = 1000
    LAG_FORWARD = 0

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        assert self.patient["sampling_rate"] == sampling_rate
        return common_preprocessing.classic_melspectrogram_pipeline(
            sound,
            self.patient["sampling_rate"],
            self.downsampling_coef,
            ecog_size,
            self.N_MELS,
            self.F_MAX,
        )

    def detect_voice(self, y_batch):
        return np.sum(y_batch > 1, axis=1) > int(self.N_MELS * 0.25)


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__40MELS(
    SimpleNetMels40Base
):
    SELECTED_CHANNELS = list(range(8, 16))


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__40MELS(
    SimpleNetMels40Base
):
    SELECTED_CHANNELS = list(range(6, 12))


class SimpleNetWhisperBase(SimpleNetBase):
    WHISPER_MODEL = "openai/whisper-base"
    WHISPER_LAYER = 4
    WHISPER_D = 512
    USE_PCA = True
    PCA_COMPONENTS = 50
    OUTPUT_SIZE = PCA_COMPONENTS
    IS_WHISPER_TARGET = True
    LAG_BACKWARD = 1000
    LAG_FORWARD = 0

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        assert self.patient["sampling_rate"] == sampling_rate
        return whisper_target.classic_whisper_pipeline(
            sound,
            self.patient["sampling_rate"],
            self.downsampling_coef,
            ecog_size,
            self.WHISPER_LAYER,
            self.WHISPER_MODEL,
        )

    def detect_voice(self, y_batch):
        return np.ones(len(y_batch), dtype=bool)


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L3(
    SimpleNetWhisperBase
):
    WHISPER_LAYER = 3
    SELECTED_CHANNELS = list(range(8, 16))


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L4(
    SimpleNetWhisperBase
):
    WHISPER_LAYER = 4
    SELECTED_CHANNELS = list(range(8, 16))


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L5(
    SimpleNetWhisperBase
):
    WHISPER_LAYER = 5
    SELECTED_CHANNELS = list(range(8, 16))


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L3(
    SimpleNetWhisperBase
):
    WHISPER_LAYER = 3
    SELECTED_CHANNELS = list(range(6, 12))


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L4(
    SimpleNetWhisperBase
):
    WHISPER_LAYER = 4
    SELECTED_CHANNELS = list(range(6, 12))


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L5(
    SimpleNetWhisperBase
):
    WHISPER_LAYER = 5
    SELECTED_CHANNELS = list(range(6, 12))
