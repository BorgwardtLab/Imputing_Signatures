import torch
import torch.nn as nn
import signatory

class SimpleDeepModel(nn.Module):
    def __init__(self, n_input_dims):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input_dims, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.model(x)


class DeepSignatureModel(nn.Module):
    def __init__(self, n_input_dims):
        super().__init__()
        #self.augment1 = signatory.Augment(in_channels=in_channels,
        #                                  layer_sizes=(8, 8, 4),
        #                                  kernel_size=4,
        #                                  include_original=True,
        #                                  include_time=True)
        self.model = nn.Sequential(
            nn.Linear(n_input_dims, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.model(x)


