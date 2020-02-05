import torch.nn as nn
import gpytorch

from src.models.mgp import GPAdapter
from src.models.signature_models import SignatureModel, RNNSignatureModel, DeepSignatureModel
from src.models.non_signature_models import GRU, LSTM

class GPSignatureModel(nn.Module):
    """
    GP Adapter combined with the SignatureModel
    """

    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, n_devices, output_device, sig_depth=2,
                 kernel='rbf', mode='normal', extra_channels=10, channel_groups=2, include_original=False, final_network=(30,30)):
        super(GPSignatureModel, self).__init__()
        
        #safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1 
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2*n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device, non_blocking=True)

        clf = SignatureModel(in_channels=clf_input_dims,
                             extra_channels=extra_channels,
                             channel_groups=channel_groups,
                             include_original=include_original,
                             include_time=True,
                             sig_depth=sig_depth,
                             out_channels=out_dimension,
                             final_network=final_network
                             )

        self.model = GPAdapter(clf,
                               None,
                               n_mc_smps,
                               sampling_type,
                               likelihood,
                               n_input_dims + 1,
                               n_devices,
                               output_device,
                               kernel,
                               mode
                               )

    def forward(self, *data):
        return self.model(*data)


class GPRNNSignatureModel(nn.Module):
    """
    GP Adapter combined with a RNNSignatureModel
    """

    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, n_devices, output_device, sig_depth=2,
                 kernel='rbf', mode='normal', extra_channels=10, channel_groups=2, include_original=False, step=1,
                 length=6, rnn_channels=32, rnn_type='gru'):
        super(GPRNNSignatureModel, self).__init__()
        
        #safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1 
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2*n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device, non_blocking=True)

        clf = RNNSignatureModel(in_channels=clf_input_dims,
                                extra_channels=extra_channels,
                                channel_groups=channel_groups,
                                include_original=include_original,
                                include_time=True,
                                sig_depth=sig_depth,
                                step=step,
                                length=length,
                                rnn_channels=rnn_channels,
                                out_channels=out_dimension,
                                rnn_type=rnn_type
                                )
        
        self.model = GPAdapter(clf,
                               None,
                               n_mc_smps,
                               sampling_type,
                               likelihood,
                               n_input_dims + 1,
                               n_devices,
                               output_device,
                               kernel,
                               mode
                               )

    def forward(self, *data):
        return self.model(*data)


class GPDeepSignatureModel(nn.Module):
    """
    GP Adapter combined with a DeepSignatureModel
    """

    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, n_devices, output_device, sig_depth=2,
                 kernel='rbf', mode='normal', hidden_channels1=8, hidden_channels2=4, kernel_size=4,
                 include_original=True):
        super(GPDeepSignatureModel, self).__init__()

        # safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2 * n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device, non_blocking=True)

        clf = DeepSignatureModel(in_channels=clf_input_dims,
                                        hidden_channels1=hidden_channels1,
                                        hidden_channels2=hidden_channels2,
                                        kernel_size=kernel_size,
                                        include_original=include_original,
                                        include_time=False,
                                        sig_depth=sig_depth,
                                        out_channels=out_dimension
                                        )

        self.model = GPAdapter(clf,
                               None,
                               n_mc_smps,
                               sampling_type,
                               likelihood,
                               n_input_dims + 1,
                               n_devices,
                               output_device,
                               kernel,
                               mode
                               )

    def forward(self, *data):
        return self.model(*data)



class GPRNNModel(nn.Module):
    """
    GP Adapter combined with a RNN
    """

    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, n_devices, output_device, sig_depth=2,
                 kernel='rbf', mode='normal', hidden_size=32, rnn_type='gru'):
        super(GPRNNModel, self).__init__()
        
        #safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1 
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2*n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device, non_blocking=True)
        if rnn_type == 'gru':
            clf_class = GRU
        elif rnn_type == 'lstm':
            clf_class = LSTM
        else:
            raise ValueError('No valid RNN type provided [gru, lstm]')
        
        clf = clf_class(out_channels=out_dimension,
                        input_size=clf_input_dims,
                        hidden_size=hidden_size
        ) 
        
        self.model = GPAdapter(clf,
                               None,
                               n_mc_smps,
                               sampling_type,
                               likelihood,
                               n_input_dims + 1,
                               n_devices,
                               output_device,
                               kernel,
                               mode
        )

    def forward(self, *data):
        return self.model(*data)

