import time

import torch
import torch.nn as nn
   

class SingleLayerLSTM(nn.Module):
    def __init__(self, input_size=260, hidden_size=512,
                 frame=False, pooling='mean'):
        super(SingleLayerLSTM, self).__init__()
        self.rnn_a = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.rnn_v = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear_a = nn.Linear(hidden_size, 1)
        self.linear_v = nn.Linear(hidden_size, 1)
        
        self.frame = frame
        self.pooling = pooling

    def forward(self, x):
        x = x.transpose(-1, -2)
        x_a, _ = self.rnn_a(x)
        x_v, _ = self.rnn_v(x)
        if self.frame:
            arousal_pred = self.linear_a(x_a).squeeze(dim=-1)
            valence_pred = self.linear_v(x_v).squeeze(dim=-1)
        else:
            if self.pooling == 'mean':
                arousal_pred = self.linear_a(x_a.mean(dim=1)).squeeze(dim=-1)
                valence_pred = self.linear_v(x_v.mean(dim=1)).squeeze(dim=-1)
            elif self.pooling == 'last':
                arousal_pred = self.linear_a(x_a[:, -1]).squeeze(dim=-1)
                valence_pred = self.linear_v(x_v[:, -1]).squeeze(dim=-1)
            else:
                raise Exception("Unknown pooling mode {}!".format(pooling))
        return torch.tanh(arousal_pred), torch.tanh(valence_pred)


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size=260, hidden_size=(512, 256),
                 frame=False, pooling='mean'):
        super(MultiLayerLSTM, self).__init__()
        self.rnn_a1 = nn.LSTM(input_size, hidden_size[0], batch_first=True)
        self.rnn_a2 = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)
        self.rnn_v1 = nn.LSTM(input_size, hidden_size[0], batch_first=True)
        self.rnn_v2 = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)
        self.linear_a = nn.Linear(hidden_size[1], 1)
        self.linear_v = nn.Linear(hidden_size[1], 1)
        
        self.frame = frame
        self.pooling = pooling


    def forward(self, x):
        x = x.transpose(-1, -2)
        x_a, _ = self.rnn_a1(x)
        x_a, _ = self.rnn_a2(x_a)
        x_v, _ = self.rnn_v1(x)
        x_v, _ = self.rnn_v2(x_v)
        if self.frame:
            arousal_pred = self.linear_a(x_a).squeeze(dim=-1)
            valence_pred = self.linear_v(x_v).squeeze(dim=-1)
        else:
            if self.pooling == 'mean':
                arousal_pred = self.linear_a(x_a.mean(dim=1)).squeeze(dim=-1)
                valence_pred = self.linear_v(x_v.mean(dim=1)).squeeze(dim=-1)
            elif self.pooling == 'last':
                arousal_pred = self.linear_a(x_a[:, -1]).squeeze(dim=-1)
                valence_pred = self.linear_v(x_v[:, -1]).squeeze(dim=-1)
            else:
                raise Exception("Unknown pooling mode {}!".format(pooling))
        return torch.tanh(arousal_pred), torch.tanh(valence_pred)


class SingleLayerAttn(nn.Module):
    def __init__(self, input_size=260, hidden_size=700):
        super(SingleLayerAttn, self).__init__()
        self.rnn_a = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.rnn_v = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.query_a = nn.Linear(hidden_size, 1)
        self.query_v = nn.Linear(hidden_size, 1)

        self.linear_a = nn.Linear(hidden_size, 1)
        self.linear_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_a, _ = self.rnn_a(x)
        x_v, _ = self.rnn_v(x)

        x_a = (torch.softmax(self.query_a(x_a), dim=1) * x_a).sum(axis=1)
        x_v = (torch.softmax(self.query_v(x_v), dim=1) * x_v).sum(axis=1)

        arousal_pred = self.linear_a(x_a).squeeze(dim=-1)
        valence_pred = self.linear_v(x_v).squeeze(dim=-1)
        return torch.tanh(arousal_pred), torch.tanh(valence_pred)

        
class MultiLayerAttn(nn.Module):
    def __init__(self, input_size=260, hidden_size=(1024, 256)):
        super(MultiLayerAttn, self).__init__()
        self.rnn_a1 = nn.LSTM(input_size, hidden_size[0], batch_first=True)
        self.rnn_a2 = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)
        self.rnn_v1 = nn.LSTM(input_size, hidden_size[0], batch_first=True)
        self.rnn_v2 = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)

        self.query_a = nn.Linear(hidden_size[1], 1)
        self.query_v = nn.Linear(hidden_size[1], 1)

        self.linear_a = nn.Linear(hidden_size[1], 1)
        self.linear_v = nn.Linear(hidden_size[1], 1)

    def forward(self, x):
        x_a, _ = self.rnn_a1(x)
        x_a, _ = self.rnn_a2(x_a)
        x_v, _ = self.rnn_v1(x)
        x_v, _ = self.rnn_v2(x_v)

        x_a = (torch.softmax(self.query_a(x_a), dim=1) * x_a).sum(axis=1)
        x_v = (torch.softmax(self.query_v(x_v), dim=1) * x_v).sum(axis=1)

        arousal_pred = self.linear_a(x_a).squeeze(dim=-1)
        valence_pred = self.linear_v(x_v).squeeze(dim=-1)
        return torch.sigmoid(arousal_pred), torch.sigmoid(valence_pred)

if __name__ == '__main__':
    X = torch.randn(4, 40, 260)
    net = MultiLayerLSTM(frame=True, pooling='last')
    arousal_pred, valence_pred = net(X)
    print(arousal_pred.shape)