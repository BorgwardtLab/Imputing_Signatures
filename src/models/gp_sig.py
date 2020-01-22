import torch.nn as nn
import gpytorch

from src.models.mgp import GPAdapter
from src.models.deep_models import DeepSignatureModel


class GP_Sig(nn.Module):
    def __init__(self, n_input_dims, n_mc_smps, n_devices, output_device, n_classes=2, sig_depth=2):
        super(GP_Sig, self).__init__()
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                        output_device, non_blocking=True)
        clf = DeepSignatureModel(in_channels=n_input_dims, out_dimension=n_classes, sig_depth=sig_depth)
        
        self.model = GPAdapter( clf, 
                                None, 
                                n_mc_smps, 
                                likelihood, 
                                n_input_dims + 1, 
                                n_devices, 
                                output_device)
    def forward(self, *data):
        return self.model(*data)
