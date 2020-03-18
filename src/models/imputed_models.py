import torch.nn as nn

from src.models.signature_models import SignatureModel, RNNSignatureModel, DeepSignatureModel
from src.models.non_signature_models import GRU


class ImputedSignatureModel(nn.Module):
    """
    SignatureModel using preprocessed imputations
    """

    def __init__(self, n_input_dims, out_dimension, sig_depth=2, extra_channels=10, channel_groups=2, include_original=False, final_network=(30,30)):
        super(ImputedSignatureModel, self).__init__()
        
        self.model = SignatureModel(in_channels=n_input_dims+1, #as we feed time also
                             extra_channels=extra_channels,
                             channel_groups=channel_groups,
                             include_original=include_original,
                             include_time=False, #as we feed time
                             sig_depth=sig_depth,
                             out_channels=out_dimension,
                             final_network=final_network
        )

    def forward(self, *data):
        return self.model(*data)

class ImputedRNNSignatureModel(nn.Module):
    """
    SignatureModel using preprocessed imputations
    """

    def __init__(self, n_input_dims, out_dimension, sig_depth=2, extra_channels=10, channel_groups=2, include_original=False, step=1, length=6, 
                 rnn_channels=32, rnn_type='gru'):
        super(ImputedRNNSignatureModel, self).__init__()
        
        self.model = RNNSignatureModel(in_channels=n_input_dims+1, #as we feed time also
                             extra_channels=extra_channels,
                             channel_groups=channel_groups,
                             include_original=include_original,
                             include_time=False, #as we feed time
                             sig_depth=sig_depth,
                             step=step,
                             length=length,
                             rnn_channels=rnn_channels,
                             out_channels=out_dimension,
                             rnn_type=rnn_type 
        )

    def forward(self, *data):
        return self.model(*data)


class ImputedDeepSignatureModel(nn.Module):
    """
    SignatureModel using preprocessed imputations
    """

    def __init__(self, n_input_dims, out_dimension, sig_depth=2, hidden_channels1=8, hidden_channels2=4, kernel_size=4,
                 include_original=True):
        super(ImputedDeepSignatureModel, self).__init__()

        self.model = DeepSignatureModel(in_channels=n_input_dims + 1,  # as we feed time also
                                        hidden_channels1=hidden_channels1,
                                        hidden_channels2=hidden_channels2,
                                        kernel_size=kernel_size,
                                        include_original=include_original,
                                        include_time=False,
                                        sig_depth=sig_depth,
                                        out_channels=out_dimension
                                       )

    def forward(self, *data):
        return self.model(*data)

class ImputedRNNModel(nn.Module):
    """
    RNN Model using preprocessed imputations
    """

    def __init__(self, n_input_dims, out_dimension, hidden_size=32, rnn_type='gru'):
        super(ImputedRNNModel, self).__init__()
        
        if rnn_type == 'gru':
            clf_class = GRU
        elif rnn_type == 'lstm':
            clf_class = LSTM
        else:
            raise ValueError('No valid RNN type provided [gru, lstm]')
 
        self.model = clf_class(out_channels=out_dimension,
                         input_size=n_input_dims+1, #as we feed time
                         hidden_size=hidden_size,
        )

    def forward(self, *data):
        return self.model(*data)


