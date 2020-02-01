import torch.nn as nn
import gpytorch

from src.models.mgp import GPAdapter
from src.models.deep_models import DeepSignatureModel
from src.models.signature_models import SignatureModel, RNNSignatureModel

class GP_Sig(nn.Module):
    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, 
            n_devices, output_device, sig_depth=2, kernel='rbf', mode='normal'):
        super(GP_Sig, self).__init__()
        
        #safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1 
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2*n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                        output_device, non_blocking=True)
        clf = DeepSignatureModel(   in_channels=clf_input_dims, 
                                    out_dimension=out_dimension, 
                                    sig_depth=sig_depth
                                )
        
        self.model = GPAdapter( clf, 
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


class GPSignatureModel(nn.Module):
    """
    GP Adapter combined with the SignatureModel
    """

    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, 
                n_devices, output_device, sig_depth=2, kernel='rbf', mode='normal', extra_channels=10, 
                channel_groups=2, include_original=False ):
        super(GPSignatureModel, self).__init__()
        
        #safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1 
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2*n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                        output_device, non_blocking=True)

        clf = SignatureModel(   in_channels=clf_input_dims,
                                extra_channels=extra_channels,
                                channel_groups=channel_groups, 
                                include_original=include_original,
                                include_time=True,
                                sig_depth=sig_depth,
                                out_channels=out_dimension, 
        )
        
        self.model = GPAdapter( clf, 
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


class GPGRUSignatureModel(nn.Module):
    """
    GP Adapter combined with a GRU-SignatureModel
    """

    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, 
                n_devices, output_device, sig_depth=2, kernel='rbf', mode='normal', extra_channels=10, 
                channel_groups=2, include_original=False, step=1, length=6, 
                rnn_channels=32):
        super(GPGRUSignatureModel, self).__init__()
        
        #safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1 
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2*n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                        output_device, non_blocking=True)

        clf = RNNSignatureModel(    in_channels=clf_input_dims,
                                    extra_channels=extra_channels,
                                    channel_groups=channel_groups, 
                                    include_original=include_original,
                                    include_time=True,
                                    sig_depth=sig_depth,
                                    step=step,
                                    length=length,
                                    rnn_channels=rnn_channels,
                                    out_channels=out_dimension, 
        )
        
        self.model = GPAdapter( clf, 
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

