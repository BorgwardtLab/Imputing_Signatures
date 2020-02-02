import torch.nn as nn


class LSTM(nn.Module):
    """As the usual nn.LSTM module.
        - Must specify the hidden_size parameter.
        - The batch_first parameter is fixed to False.
        - The bidirectional parameter is fixed to False (we want to be causal).
        - Must also specify an out_channels parameter. A linear map will be added from the hidden state to the output.
    """

    def __init__(self, out_channels, **kwargs):
        hidden_size = kwargs['hidden_size']
        kwargs['batch_first'] = True
        kwargs['bidirectional'] = False

        self.lstm = nn.LSTM(**kwargs)
        self.linear = nn.Linear(2 * hidden_size, out_channels)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        h = h[-1]
        c = c[-1]
        x = torch.cat([h, c], dim=1)
        x = self.linear(x)
        return x


class GRU(nn.Module):
    """As the usual nn.GRU module.
        - Must specify the hidden_size parameter.
        - The batch_first parameter is fixed to False.
        - The bidirectional parameter is fixed to False (we want to be causal).
        - Must also specify an out_channels parameter. A linear map will be added from the hidden state to the output.
    """

    def __init__(self, out_channels, **kwargs):
        kwargs['batch_first'] = True
        kwargs['bidirectional'] = False
        self.hidden_size = kwargs['hidden_size']
        self.num_layers = kwargs.get('num_layers', 1)

        self.gru = nn.GRU(**kwargs)
        self.linear = nn.Linear(self.hidden_size, out_channels)

    def forward(self, x):
        _, x = self.gru(x)
        x = x[-1]  # take the last GRU layer
        x = self.linear(x)
        return x
