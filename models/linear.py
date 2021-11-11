# Now only simple regression
# just used to test the dataloader and the metrics
import time

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_dim=260):
        super(Linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.linear_arousal = nn.Linear(128, 1)
        self.linear_valence = nn.Linear(128, 1)

    def forward(self, x):
        feature = self.linear(x)
        arousal_pred = self.linear_arousal(feature).squeeze(dim=-1)
        valence_pred = self.linear_valence(feature).squeeze(dim=-1)
        return arousal_pred, valence_pred