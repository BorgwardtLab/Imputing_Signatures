import torch.nn as nn
import gpytorch

from src.models.mgp import GPAdapter
from src.models.deep_models import DeepSignatureModel, DeeperSignatureModel


class GP_Sig(nn.Module):
    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, 
            n_devices, output_device, sig_depth=2, kernel='rbf'):
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
                                kernel 
        )

    def forward(self, *data):
        return self.model(*data)


class GP_DeepSig(nn.Module):
    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, 
            n_devices, output_device, sig_depth=2, kernel='rbf'):
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
        clf = DeeperSignatureModel(   in_channels=clf_input_dims, 
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
                                kernel 
        )

    def forward(self, *data):
        return self.model(*data)
