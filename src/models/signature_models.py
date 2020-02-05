import torch
import torch.nn as nn
import signatory


def stack(tensors, dim):
    # Small efficiency boost in the common case that there's only one tensor to stack
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim)


def become_constant_trick(x, lengths):
    """
    Input:
        x: a tensor of shape (batch, stream, channel)
        lengths: a tensor of shape (batch,)

    Returns:
        A tensor y of shape (batch, stream, channel), such that
        y[i, j, k] = x[i, j, k] if j < lengths[i] else x[i, lengths[i], k]

    This is useful when taking the signature afterwards, as it makes the signature not notice any changes made after
    length.
    """

    batch, stream, channel = x.shape
    if (stream < lengths).any():
        raise ValueError("x's stream dimension is of length {} but one of the lengths is longer than this. lengths={}"
                         "".format(stream, lengths))
    lengths = lengths - 1  # go from length-of-sequence to index-of-final-element-in-sequence
    expanded_lengths = lengths.unsqueeze(1).unsqueeze(2).expand(batch, 1, channel)
    final_value = x.gather(dim=1, index=expanded_lengths)
    final_value.expand(batch, stream, channel)
    mask = torch.arange(0, stream, device=x.device).unsqueeze(0) > lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).expand(batch, stream, channel)
    return x.masked_scatter(mask, final_value.masked_select(mask))


class SignatureModel(nn.Module):
    """This is a reasonably simple, reasonably flexible signature model.
    It can be used as either as a shallow signature model, or as a simple deep signature model.
    It:
    (a) Sweeps a linear augmentation over the data
    (b) Takes a signature of the whole stream
    (c) Takes a linear function of the signature
    """

    def __init__(self, in_channels, extra_channels, channel_groups, include_original, include_time, sig_depth,
                 out_channels, final_network):
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
            out_channels: How many channels to output.
            final_network: What kind of network to use on the result of the signature. Should be a tuple or scalars,
                representing the sizes of hidden layers in a small feedforward neural network. ReLU nonlinearities will
                be placed in between. For example, an empty tuple represents no hidden layers; i.e. just a linear map.

        Examples:
            extra_channels=0, include_original=True, channel_groups=1:
                This corresponds to shallow signature models, without any learnt transformation before the signature.

            extra_channels=10, incldue_original=False:
                This corresponds to the simplest possible deep signature model, just learning a simple transformation
                before the signature.
        """

        super().__init__()

        self.channel_groups = channel_groups
        self.sig_depth = sig_depth

        layer_sizes = () if extra_channels == 0 else (extra_channels,)
        self.augments = nn.ModuleList(signatory.Augment(in_channels=in_channels,
                                                        layer_sizes=layer_sizes,
                                                        # IMPORTANT. We rely on kernel_size=1 to make trick for handling
                                                        # variable-length inputs work.
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

        layers = []
        prev_size = sig_channels
        for size in final_network:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, out_channels))
        self.neural = nn.Sequential(*layers)

    def forward(self, x, lengths):
        # `x` should be a three dimensional tensor (batch, stream, channel)
        # `lengths` should be a one dimensional tensor (batch,) giving the true length of each batch element along the
        # stream dimension

        batch, stream, channel = x.shape

        x = become_constant_trick(x, lengths)

        x = stack([augment(x) for augment in self.augments], dim=1)
        # channel_group is essentially an extra batch dimension, but unfortunately signatory.signature doesn't support
        # multiple batch dimensions. So the trick is just to combine all the batch dimensions and then peel them apart
        # again afterwards.
        x = x.view(batch * self.channel_groups, stream, x.size(-1))
        x = signatory.signature(x, self.sig_depth)
        x = x.view(batch, -1)
        x = self.neural(x)
        return x


class RNNSignatureModel(nn.Module):
    """This is a reasonably simple implementation of a model that:
    (a) Sweeps a linear augmentation over the data
    (b) Takes the signature (of the augmented data) over a series of sliding windows
    (c) Runs a GRU or LSTM over the sequence of signatures of each window.
    """

    def __init__(self, in_channels, extra_channels, channel_groups, include_original, include_time, sig_depth, step,
                 length, rnn_channels, out_channels, rnn_type):
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
            rnn_type: Either 'gru' or 'lstm'.

        Note:
            Unless step, length, and the length of the input stream all suitably line up, then the final pieces of data
            in the input stream may not end up being used, because the next sliding window would go 'off the end' of the
            data.
            This can be avoided by setting the parameters appropriately, e.g. by taking step=1.
        """

        super().__init__()

        self.channel_groups = channel_groups
        self.sig_depth = sig_depth
        self.step = step
        self.length = length
        self.rnn_channels = rnn_channels
        self.out_channels = out_channels
        self.rnn_type = rnn_type

        layer_sizes = () if extra_channels == 0 else (extra_channels,)
        self.augments = nn.ModuleList(signatory.Augment(in_channels=in_channels,
                                                        layer_sizes=layer_sizes,
                                                        # IMPORTANT. We rely on kernel_size=1 to make trick for handling
                                                        # variable-length inputs work.
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
        if rnn_type == 'gru':
            self.rnn_cell = nn.GRUCell(sig_channels, rnn_channels)
        elif rnn_type == 'lstm':
            self.rnn_cell = nn.LSTMCell(sig_channels, rnn_channels)
        else:
            raise ValueError('rnn_type of value "{}" not understood'.format(rnn_type))

        self.linear = nn.Linear(rnn_channels, out_channels)

    def forward(self, x, lengths):
        # `x` should be a three dimensional tensor (batch, stream, channel)
        # `lengths` should be a one dimensional tensor (batch,) giving the true length of each batch element along the
        # stream dimension

        batch, stream, channel = x.shape

        x = become_constant_trick(x, lengths)

        x = stack([augment(x) for augment in self.augments], dim=1)
        x = x.view(batch * self.channel_groups, stream, x.size(-1))
        path = signatory.Path(x, self.sig_depth)

        # x now represents the hidden state of the GRU
        x = torch.zeros(batch, self.rnn_channels, device=x.device, dtype=x.dtype)
        for index in range(0, path.size(1) - self.length + 1, self.step):
            # Compute the signature over a sliding window
            signature_of_window = path.signature(index, index + self.length)
            signature_of_window = signature_of_window.view(batch, -1)
            # mask to only use the update when we're not exceeding the maximum length of this sequence
            mask = (index + self.length) > lengths
            mask = mask.unsqueeze(1).expand(batch, self.rnn_channels)
            y = self.rnn_cell(signature_of_window, x)
            if self.rnn_type == 'lstm':
                y = torch.cat(y, dim=1)
            x = torch.where(mask, x, y)

        x = self.linear(x)
        return x


class DeepSignatureModel(nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, kernel_size, include_original, include_time,
                 sig_depth, out_channels):
        """
        Inputs:
            in_channels: As SignatureModel.
            hidden_channels1: How large to make certain hidden channels within the model.
            hidden_channels2: How large to make certain hidden channels within the model.
            kernel_size: How far to look back in time.
            include_original: As SignatureModel.
            include_time: As SignatureModel.
            sig_depth: As SignatureModel.
            out_channels: As SignatureModel.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.augment1 = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(hidden_channels1, hidden_channels1, hidden_channels2),
                                          kernel_size=kernel_size,
                                          include_original=include_original,
                                          include_time=include_time)
        self.signature1 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        sig_hidden_channels = hidden_channels2
        if include_original:
            sig_hidden_channels += in_channels
        if include_time:
            sig_hidden_channels += 1
        sig_channels1 = signatory.signature_channels(channels=sig_hidden_channels,
                                                     depth=sig_depth)
        self.augment2 = signatory.Augment(in_channels=sig_channels1,
                                          layer_sizes=(hidden_channels1, hidden_channels1, hidden_channels2),
                                          kernel_size=kernel_size,
                                          include_original=False,
                                          include_time=False)
        self.signature2 = signatory.Signature(depth=sig_depth,
                                              stream=False)

        sig_channels2 = signatory.signature_channels(channels=hidden_channels2,
                                                     depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels2, out_channels)

    def forward(self, x, lengths):
        # `x` should be a three dimensional tensor (batch, stream, channel)
        # `lengths` should be a one dimensional tensor (batch,) giving the true length of each batch element along the
        # stream dimension
        
        adjusted_lengths = lengths - 2 * self.kernel_size + 2
        if (adjusted_lengths < 0).any():
            raise ValueError('The kernel size is too large top operate this model on a stream this short.')

        x = self.augment1(x)
        x = self.signature1(x, basepoint=True)
        x = self.augment2(x)
        x = become_constant_trick(x, adjusted_lengths)
        x = self.signature2(x, basepoint=True)
        x = self.linear(x)
        return x
