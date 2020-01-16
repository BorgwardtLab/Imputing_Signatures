import gpytorch
import numpy as np
from sklearn.metrics import roc_auc_score as auc
import torch
import torch.nn as nn
from functools import partial

from torch.utils.data import DataLoader

import sys,os
sys.path.append(os.getcwd())

from src.datasets import Physionet2012Dataset
from src.datasets import dict_collate_fn, to_gpytorch_format
from src.models.mgp import GPAdapter
from src.models.deep_models import DeepSignatureModel
# from src.utils.train_utils import augment_labels

# ----------------------------------------------
# Training a GP-Sig adapter on physionet 2012
# ----------------------------------------------


def augment_labels(labels, n_samples):
    """Expand labels for multiple MC samples in the GP Adapter.

    Args:
         Takes tensor of size [n]

    Returns:
        expanded tensor of size [n_mc_samples, n]

    """
    return labels.expand(labels.shape[0], n_samples).transpose(1, 0)


input_transform = partial(to_gpytorch_format, grid_spacing=1.)

# Setup training data
d = Physionet2012Dataset(split='training', transform=input_transform)
n_tasks = d.measurement_dims
# Should use the index of the auxillary task for padding
collate_fn = partial(
    dict_collate_fn,
    padding_values={
        'indices': n_tasks,
        'test_indices': n_tasks
    }
)
batch_size = 25
data_loader = DataLoader(d, batch_size, collate_fn=collate_fn)


# Setup validation data
d = Physionet2012Dataset(split='validation', transform=input_transform)
data_loader = DataLoader(d, batch_size, collate_fn=collate_fn)

# Training Parameters:
n_epochs = 50
device = 'cuda'

# Setting up parameters of GP:
n_mc_smps = 5
# augment tasks with dummy task for imputed 0s for tensor format
num_tasks = n_tasks + 1
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Initializing GP adapter model (for now assuming imputing to equal length time
# series)
clf = DeepSignatureModel(in_channels=n_tasks, sig_depth=2)
model = GPAdapter(clf, None, n_mc_smps, likelihood, num_tasks)
model.to(device)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.001)

# Loss function:
loss_fn = nn.CrossEntropyLoss(reduction='mean')

for epoch in range(n_epochs):
    for d in data_loader:  # for test trial, overfit same batch of samples
        y_true = augment_labels(d['label'], n_mc_smps)

        # activate training mode of deep model:
        model.train()

        # forward pass:
        # with gpytorch.settings.fast_pred_samples():
        
        inputs = d['inputs']
        indices = d['indices'] 
        values = d['values']
        test_inputs = d['test_inputs']
        test_indices = d['test_indices'] 
        
        if device == 'cuda':
            inputs  = inputs.cuda(non_blocking = True)
            indices = indices.cuda(non_blocking = True)
            values  = values.cuda(non_blocking = True)
            test_inputs = test_inputs.cuda(non_blocking = True)
            test_indices = test_indices.cuda(non_blocking = True)
        
        with gpytorch.settings.fast_pred_samples():
        #with gpytorch.settings.fast_pred_var():
            logits = model( inputs, 
                            indices, 
                            values, 
                            test_inputs, 
                            test_indices )

        #evaluate loss
        if device == 'cuda':
            y_true = y_true.long().flatten().cuda()
        else: 
            y_true = y_true.long().flatten()
        loss = loss_fn(logits, y_true)

        # Optimize:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            AUROC = auc(y_true.detach().cpu().numpy(),logits[:,1].flatten().detach().cpu().numpy()) #logits[:,:,1]
            print(f'Epoch {epoch}, Train Loss: {loss.item():03f}  Train AUC: {AUROC:03f}')

        torch.cuda.empty_cache()
