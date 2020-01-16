import gc
import gpytorch
import numpy as np
from sklearn.metrics import roc_auc_score as auc
import time
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

#Hyperparameters:
batch_size = 32
lr = 0.0005 #0.001

training_loader = DataLoader(d, batch_size, collate_fn=collate_fn)

# Setup validation data
d = Physionet2012Dataset(split='validation', transform=input_transform)
validation_loader = DataLoader(d, batch_size, collate_fn=collate_fn)

# Training Parameters:
n_epochs = 50
device = 'cuda'
output_device = torch.device('cuda:0')

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
], lr=lr)

# Loss function:
loss_fn = nn.CrossEntropyLoss(reduction='mean')

n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))

for epoch in range(n_epochs):

    y_true_total = []
    y_score_total = []
    loss_total = []
    start = time.time()

    for iteration, d in enumerate(training_loader):  # for test trial, overfit same batch of samples

        iter_start = time.time() 
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
        
        #with gpytorch.settings.fast_pred_samples():
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(40):
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
            y_true = y_true.detach().cpu().numpy()
            y_score = logits[:,1].flatten().detach().cpu().numpy()  
            #AUROC = auc(y_true, y_score) #logits[:,:,1]
            #print(f'Epoch {epoch}, (MB) Train Loss: {loss.item():03f}, (MB) Train AUC: {AUROC:03f}, Iteration of batch-size {batch_size} took: {time.time() - iter_start:02f} seconds')
            print(f'Epoch {epoch}, (MB) Train Loss: {loss.item():03f}, Iteration of batch-size {batch_size} took: {time.time() - iter_start:02f} seconds')

        y_true_total.append(y_true)
        y_score_total.append(y_score)
        loss_total.append(loss.item())
        torch.cuda.empty_cache()
        gc.collect()
    
    y_true_total = np.concatenate(y_true_total)
    y_score_total = np.concatenate(y_score_total)
    mean_loss = np.mean(loss_total)
    AUROC = auc(y_true_total, y_score_total) #logits[:,:,1]
    print(f'>>> Epoch {epoch}, Mean Train Loss: {mean_loss:03f}, Overall Train AUROC: {AUROC:03f}. Epoch took {time.time() - start } seconds.')
 
