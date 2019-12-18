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
    def __init__(self, in_channels, out_dimension=2, sig_depth=3):
        super().__init__()
        self.augment1 = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(8, 8, 4),
                                          kernel_size=4,
                                          include_original=True,
                                          include_time=True)
        self.signature1 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        # +5 because self.augment1 is used to add time, and 4 other
        # channels, as well
        sig_channels1 = signatory.signature_channels(channels=in_channels + 5,
                                                     depth=sig_depth)
        self.augment2 = signatory.Augment(in_channels=sig_channels1,
                                          layer_sizes=(8, 8, 4),
                                          kernel_size=4,
                                          include_original=False,
                                          include_time=False)
        self.signature2 = signatory.Signature(depth=sig_depth,
                                              stream=False)

        # 4 because that's the final layer size in self.augment2
        sig_channels2 = signatory.signature_channels(channels=4,
                                                     depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels2, out_dimension)
    
    def forward(self, x):
        # in docu: input is a three dimensional tensor of shape (batch, stream, in_channels)
        a = self.augment1(x)
        if a.size(-2) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # a is a three dimensional tensor of shape (batch, stream, in_channels + 5)
        b = self.signature1(a, basepoint=True)
        # b is a three dimensional tensor of shape (batch, stream, sig_channels1)
        c = self.augment2(b)
        if c.size(-2) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # c is a three dimensional tensor of shape (batch, stream, 4)
        d = self.signature2(c, basepoint=True)
        # d is a two dimensional tensor of shape (batch, sig_channels2)
        e = self.linear(d)
        # e is a two dimensional tensor of shape (batch, out_dimension)
        return e
