import gpytorch
import numpy as np
from sklearn.metrics import roc_auc_score as auc
import torch
import torch.nn as nn

import sys,os
sys.path.append(os.getcwd())

from src.datasets.synthetic_dataset import create_synthetic_dataset
from src.models.gp_sig import GP_Sig
#from src.models.mgp import GPAdapter
#from src.models.deep_models import DeepSignatureModel
#from src.utils.train_utils import augment_labels
 
# ----------------------------------------------
# Training a simple MGP adapter (synthetic data)
# ----------------------------------------------

def augment_labels(labels, n_samples):
    """Expand labels for multiple MC samples in the GP Adapter.

    Args:
         Takes tensor of size [n]

    Returns:
        expanded tensor of size [n_mc_samples, n]

    """
    return labels.unsqueeze(-1).expand(labels.shape[0], n_samples).transpose(1, 0)


# Setup Parameters:
device = 'cuda'

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

output_device = torch.device('cuda:0') #in case we want to use multiple GPUs
n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))

model = GP_Sig(n_tasks, n_mc_smps, n_devices, output_device)
model.cuda()

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
   
    #input data to device if cuda:
    if device == 'cuda':
        inputs = inputs.cuda(non_blocking = True)
        indices = indices.cuda(non_blocking = True)
        values = values.cuda(non_blocking = True)
        test_inputs = test_inputs.cuda(non_blocking = True)
        test_indices = test_indices.cuda(non_blocking = True)

    #forward pass:
    #with gpytorch.settings.fast_pred_samples(): 
    with gpytorch.settings.fast_pred_var():
        logits = model(inputs, indices, values, test_inputs, test_indices) 
    
    #evaluate loss
    if device == 'cuda':
        y_true = y_true.long().flatten().cuda()
    else: 
        y_true = y_true.long().flatten()
    loss = loss_fn(logits, y_true)
    
    #Optimize:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        AUROC = auc(y_true.detach().cpu().numpy(),logits[:,1].flatten().detach().cpu().numpy()) #logits[:,:,1]
        print(f'Epoch {i}, Train Loss: {loss.item():03f}  Train AUC: {AUROC:03f}')


