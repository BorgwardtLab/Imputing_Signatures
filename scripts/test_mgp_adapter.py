import gpytorch
import numpy as np
from sklearn.metrics import roc_auc_score as auc
import torch
import torch.nn as nn

from src.dataset.synthetic_dataset import create_synthetic_dataset
from src.models.mgp import GPAdapter
from src.models.deep_models import DeepSignatureModel
from src.utils.train_utils import augment_labels
 
# ----------------------------------------------
# Training a simple MGP adapter (synthetic data)
# ----------------------------------------------

# Generate Data:
n_samples=20
n_tasks=3
n_query=51
noise=3
labels, inputs, values, indices, test_inputs, test_indices = create_synthetic_dataset(n_samples, n_tasks, n_query, noise) 

# Training Parameters:
n_epochs = 50

# Setting up parameters of GP:
n_mc_smps = 10
n_input_dims = test_inputs.shape[1]
num_tasks=n_tasks+1 #augment tasks with dummy task for imputed 0s for tensor format
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Initializing GP adapter model (for now assuming imputing to equal length time series)
clf = DeepSignatureModel(in_channels=n_tasks, sig_depth=3)
model = GPAdapter(  clf, 
                    n_input_dims, 
                    n_mc_smps, 
                    likelihood, 
                    num_tasks)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()}, 
], lr=0.01)

# Loss function:
loss_fn = nn.CrossEntropyLoss(reduction='mean')

for i in np.arange(n_epochs): #for test trial, overfit same batch of samples
    #augment labels:
    y_true = augment_labels(labels, n_mc_smps)
    
    #activate training mode of deep model:
    model.train()
    
    #forward pass:
    #with gpytorch.settings.fast_pred_samples(): 
    with gpytorch.settings.fast_pred_var():
        logits = model(inputs, indices, values, test_inputs, test_indices) 
    
    #evaluate loss
    loss = loss_fn(logits, y_true.long().flatten())
    
    #Optimize:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        AUROC = auc(y_true.long().flatten().detach().numpy(),logits[:,1].flatten().detach().numpy()) #logits[:,:,1]
        print(f'Epoch {i}, Train Loss: {loss.item():03f}  Train AUC: {AUROC:03f}')


