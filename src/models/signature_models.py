import torch
import torch.nn as nn
import signatory


def stack(tensors, dim):
    # Small efficiency boost in the common case that there's only one tensor to stack
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim)


class SignatureModel(nn.Module):
    """This is a reasonably simple, reasonably flexible signature model.
    It can be used as either as a shallow signature model, or as a simple deep signature model.
    It:
    (a) Sweeps a linear augmentation over the data
    (b) Takes a signature of the whole stream
    (c) Takes a linear function of the signature
    """

    def __init__(self, in_channels, extra_channels, channel_groups, include_original, include_time, sig_depth,
                 out_channels):
        """
        Inputs:
            in_channels: An integer specifying the number of input channels in the data.
            extra_channels: How many channels to learn before the signature. Can be set to 0 to not learn any extra
                channels.
            channel_groups: We compute the signature on channel_groups many paths. Note that this is only worth setting
                to something other than 1 if extra_channels is nonzero.
            include_original: Whether to pass the original data to the signature. If True then the original data and the
                extra learnt channels will be concatenated. (So you probably want True if the number of channels in the
                original data is small). If False then the original data will not be included (so you probably want
                False if the number of channels in the original data is large - then you get just the learnt channels.)
            include_time: Whether to include a time parameter in the augmentation. You probably want to set this to
                True.
            sig_depth: What depth of signature to calculate. Careful - you'll get exponentially many more parameters as
                this number is increased. Reducing the value of extra_channels or toggling off include_original will
                also help reduce the number of parameters.
            out_channels: How many channels to learn a linear map to, from the signature.

        Examples:
            extra_channels=0, include_original=True, channel_groups=1:
                This corresponds to shallow signature models, without any learnt transformation before the signature.

            extra_channels=10, incldue_original=False:
                This corresponds to the simplest possible deep signature model, just learning a simple transformation
                before the signature.
        """

        super().__init__()

        self.sig_depth = sig_depth

        layer_sizes = () if extra_channels == 0 else (extra_channels,)
        self.augments = nn.ModuleList(signatory.Augment(in_channels=in_channels,
                                                        layer_sizes=layer_sizes,
                                                        kernel_size=1,
                                                        include_original=include_original,
                                                        include_time=include_time) for _ in range(channel_groups))

        in_sig_channels = extra_channels
        if include_original:
            in_sig_channels += in_channels
        if include_time:
            in_sig_channels += 1
        sig_channels = signatory.signature_channels(in_sig_channels, sig_depth)
        sig_channels *= channel_groups

        self.linear = nn.Linear(sig_channels, out_channels)

    def forward(self, x):
        # x should be a three dimensional tensor (batch, stream, channel)

        x = stack([augment(x) for augment in self.augments], dim=1)
        # channel_group is essentially an extra batch dimension, but unfortunately signatory.signature doesn't support
        # multiple batch dimensions. So the trick is just to combine all the batch dimensions and then peel them apart
        # again afterwards.
        batch, channel_group, stream, channels = x.shape
        x = x.view(batch * channel_group, stream, channels)
        x = signatory.signature(x, self.sig_depth)
        x = x.view(batch, -1)
        x = self.linear(x)
        return torch.sigmoid(x)


class RNNSignatureModel(nn.Module):
    """This is a reasonably simple implementation of a model that:
    (a) Sweeps a linear augmentation over the data
    (b) Takes the signature (of the augmented data) over a series of sliding windows
    (c) Runs a GRU over the sequence of signatures of each window.
    """

    def __init__(self, in_channels, extra_channels, channel_groups, include_original, include_time, sig_depth, step,
                 length, rnn_channels, out_channels):
        """
        Inputs:
            in_channels: As SignatureModel.
            extra_channels: As SignatureModel.
            channel_groups: As SignatureModel.
            include_original: As SignatureModel.
            include_time: As SignatureModel.
            sig_depth: As SignatureModel.
            step: The number of indices to move the sliding window forward by.
            length: The length of the sliding window that a signature is taken over.
            rnn_channels: The size of the hidden state of the GRU.
            out_channels: As SignatureModel.

        Note:
            Unless step, length, and the length of the input stream all suitably line up, then the final pieces of data
            in the input stream may not end up being used, because the next sliding window would go 'off the end' of the
            data.
            This can be avoided by setting the parameters appropriately, e.g. by taking step=1.
        """

        super().__init__()

        self.sig_depth = sig_depth
        self.step = step
        self.length = length
        self.rnn_channels = rnn_channels
        self.out_channels = out_channels

        layer_sizes = () if extra_channels == 0 else (extra_channels,)
        self.augments = nn.ModuleList(signatory.Augment(in_channels=in_channels,
                                                        layer_sizes=layer_sizes,
                                                        kernel_size=1,
                                                        include_original=include_original,
                                                        include_time=include_time) for _ in range(channel_groups))

        in_sig_channels = extra_channels
        if include_original:
            in_sig_channels += in_channels
        if include_time:
            in_sig_channels += 1
        sig_channels = signatory.signature_channels(in_sig_channels, sig_depth)
        sig_channels *= channel_groups
        self.rnn_cell = nn.GRUCell(sig_channels, rnn_channels)

        self.linear = nn.Linear(rnn_channels, out_channels)

    def forward(self, x):
        # x should be a three dimensional tensor (batch, stream, channel)

        x = stack([augment(x) for augment in self.augments], dim=1)
        batch, channel_group, stream, channels = x.shape
        x = x.view(batch * channel_group, stream, channels)
        path = signatory.Path(x, self.sig_depth)

        # x now represents the hidden state of the GRU
        x = torch.zeros(batch, self.rnn_channels)
        for index in range(0, path.size(1) - self.length + 1, self.step):
            # Compute the signature over a sliding window
            signature_of_window = path.signature(index, index + self.length)
            signature_of_window = signature_of_window.view(batch, -1)
            x = self.rnn_cell(signature_of_window, x)

        x = self.linear(x)
        return torch.sigmoid(x)
