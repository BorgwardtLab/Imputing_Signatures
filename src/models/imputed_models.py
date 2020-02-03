import torch.nn as nn

from src.models.signature_models import SignatureModel, RNNSignatureModel



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
                             include_time=False,
                             sig_depth=sig_depth,
                             out_channels=out_dimension,
                             final_network=final_network
                             )

    def forward(self, *data):
        return self.model(*data)



