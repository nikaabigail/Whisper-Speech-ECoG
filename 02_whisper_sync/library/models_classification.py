import torch
import torch.nn as nn
import torch.nn.functional as F

class Mel2WordSimple(nn.Module):
    MEL_PRE_DONSAMPLING = 10
    MELS_CONV_SIZE = 200
    HIDDEN_CHANNELS = 100
    MEL_POST_DONSAMPLING = 10

    def __init__(self, in_channels, out_channels):
        super().__init__()  # было super(self.__class__, self) — ломало наследование

        self.mels2features = nn.Sequential(
            nn.Conv2d(1, self.HIDDEN_CHANNELS, kernel_size=(in_channels, 10)),
        )
        self.max_pool = torch.nn.MaxPool1d(self.MEL_POST_DONSAMPLING)

        # PyTorch applies recurrent dropout only between stacked LSTM layers.
        # With num_layers=1 a non-zero value is ignored and only emits a warning.
        self.lstm = nn.LSTM(
            self.HIDDEN_CHANNELS,
            self.HIDDEN_CHANNELS,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(200, out_channels),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x[:, :, :, ::self.MEL_PRE_DONSAMPLING]
        mels_features = self.mels2features(x)
        mels_features = mels_features.squeeze(2)
        mels_features = self.max_pool(mels_features)
        mels_features = mels_features.transpose(1, 2)
        lstm_out, (lstm_hidden_state_h, lstm_hidden_state_c) = self.lstm(mels_features)
        lstm_hidden_state_h = lstm_hidden_state_h.transpose(0, 1)
        features = lstm_hidden_state_h.reshape((lstm_hidden_state_h.size(0), -1)) # flatten
        output = self.fc_layer(features)
        return output


class Mel2WordHidden(Mel2WordSimple):
    """
    Вариант классификатора для hidden-state входа (Фаза 2.1, вариант A).

    На вход подаётся внутреннее представление энкодера регрессии (features_scaled),
    уже прорежённое во времени в MEL_PRE_DONSAMPLING раз ещё на этапе извлечения.
    Поэтому здесь внутреннее прореживание отключено (MEL_PRE_DONSAMPLING = 1),
    чтобы не прорежать дважды. in_channels тут большой (напр. 3030) — свёртка
    Conv2d с ядром (in_channels, 10) сворачивает всю размерность признаков.
    Всё остальное (maxpool, BiLSTM, fc) наследуется без изменений.
    """
    MEL_PRE_DONSAMPLING = 1
