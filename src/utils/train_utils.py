import torch

def augment_labels(labels, n_samples):
    """
    Function to expand labels for multiple MC samples in the GP Adapter 
        - takes tensor of size [n]
        - returns expanded tensor of size [n_mc_samples, n]
    """
    return labels.unsqueeze(-1).expand(labels.shape[0],n_samples).transpose(1,0)


