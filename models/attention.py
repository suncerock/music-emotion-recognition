# Now only simple regression
# just used to test the dataloader and the metrics
import time

import torch
import torch.nn as nn


class SingleLayerLSTM(nn.Module):
    def __init__(self, input_size=260, hidden_size=1024):
        super(SingleLayerLSTM, self).__init__()
        self.rnn_a = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.rnn_v = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear_a = nn.Linear(hidden_size, 1)
        self.linear_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_a, _ = self.rnn_a(x)
        x_v, _ = self.rnn_v(x)
        arousal_pred = self.linear_a(x_a[:, -1]).squeeze(dim=-1)
        valence_pred = self.linear_v(x_v[:, -1]).squeeze(dim=-1)
        return arousal_pred, valence_pred


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size=260, hidden_size=(700, 128)):
        super(MultiLayerLSTM, self).__init__()
        self.rnn_a1 = nn.LSTM(input_size, hidden_size[0], batch_first=True)
        self.rnn_a2 = nn.LSTM(input_size, hidden_size[1], batch_first=True)
        self.rnn_v1 = nn.LSTM(input_size, hidden_size[0], batch_first=True)
        self.rnn_v2 = nn.LSTM(input_size, hidden_size[1], batch_first=True)
        self.linear_a = nn.Linear(hidden_size, 1)
        self.linear_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_a, _ = self.rnn_a1(x)
        x_a, _ = self.rnn_a2(x_a)
        x_v, _ = self.rnn_v1(x)
        x_v, _ = self.rnn_v2(x_v)
        arousal_pred = self.linear_a(x_a[:, -1]).squeeze(dim=-1)
        valence_pred = self.linear_v(x_v[:, -1]).squeeze(dim=-1)
        return arousal_pred, valence_pred


class SingleLayerAttn(nn.Module):
    pass


class MultiLayerAttn(nn.Module):
    pass