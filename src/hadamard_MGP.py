
# coding: utf-8

# # Testing Hadamard MGP with GPytorch

# In[332]:


import numpy as np
import math
import torch
import torch.nn as nn
import gpytorch
from matplotlib import pyplot as plt
import signatory

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from collections import defaultdict
nested_dict = lambda: defaultdict(nested_dict)

def generate_dataset(n_samples=5, n_tasks=3, noise=1):
    """
    Generating Variable Length, Irregularly observed Time Series (with sine and cosine)
    Returns:
    -data dict:
        'x':
            'task_0': list of tensors
            'task_1': ..
             .. (n_tasks-1)
        'y': 
            'task_0': list of tensors
            'task_1': ..
             .. (n_tasks-1)
    list elements represent samples / instances
    """
    #determine labels (balanced)
    n_cases = int(np.floor(n_samples/2))
    n_controls = int(n_samples - n_cases)
    labels = torch.cat([torch.ones(n_cases), torch.zeros(n_controls)])
    #add signals correlated with label:
    signal = 5 #affects the frequency of the sine/cosines
    
    #first draw number of observations per sample (uniform between 20 and 50)
    lengths = torch.randint(50, 150, (n_tasks,n_samples))
    data = nested_dict()
    for i, length in enumerate(lengths): #loop over tasks
        x_list = []
        y_list = []
        for j, l in enumerate(length): #loop over samples
            x = torch.rand(l)
            if i % 2 == 0: #even tasks with sin, uneven with cos
                y = torch.sin(torch.randn(1) + x * ((5 + (signal*labels[j])*torch.randn(1))* math.pi)) + noise*torch.randn(x.size()) * 0.1
            else:
                y = torch.cos(torch.randn(1) + x * ((5+ (signal*labels[j])*torch.randn(1))* math.pi)) + noise*torch.randn(x.size()) * 0.1
            x_list.append(x)
            y_list.append(y)
        data['x']['task_' + str(i)] = x_list
        data['y']['task_' + str(i)] = y_list
    return data, labels    


# ## Dataset TODO:
# generate synthetic data in following format:
# list of per-instance tuples: [ (data, label), .. ()]
# data = (times (n_timepoints) , values (n_timepoints, n_dims))
# --> Unobserved points are NaNs
# Write collate_fn which takes this format and compiles the triple: inputs, indices, values (per batch)

# In[322]:


from torch.utils.data import Dataset


# In[323]:


class SyntheticData(Dataset):
    def __init__(self, n_samples=500, n_tasks=3):
        """
        Generate synthetic data when initializing
        """
        super().__init__()
        self.data, self.labels = generate_dataset(n_samples, n_tasks)
        
        #prepare data for hadamard GP setup (decompose into inputs (times), task indices and observed values)
        self.inputs, self.values, self.indices = prepare_train_data(self.data)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        return self.values[idx]
    

syn_dat = SyntheticData()


from collections import defaultdict
nested_dict = lambda: defaultdict(nested_dict)

def generate_test_points(n_samples=5, n_tasks=3, n_query=51):
    inputs = torch.linspace(0, 1, n_query)
    
    def get_indices(data,fill=0):
        return torch.full_like(data, dtype=torch.long, fill_value=fill)
    indices = torch.stack(
                   [ torch.cat(
                        [get_indices(inputs, i) for i in np.arange(n_tasks)]
                    ) for j in np.arange(n_samples)
                   ]
    ) 
    
    inputs = inputs.repeat(n_samples,n_tasks)

    #add extra dimension for GPytorch:
    inputs = inputs.unsqueeze(-1)
    indices = indices.unsqueeze(-1)
    return inputs, indices


# In[5]:


n_samples=10
n_tasks=3
data,labels = generate_dataset(n_samples=n_samples, n_tasks=n_tasks, noise=3)


# In[6]:


n_query=51
test_inputs, test_indices = generate_test_points(n_samples=n_samples, n_tasks=n_tasks, n_query=n_query)


# In[7]:


def plot_data(data):
    n_samples = len(data['x']['task_0'])
    n_tasks = len(list(data['x'].keys()))
    fig, ax = plt.subplots(n_samples, figsize=(10,10))
    for i in np.arange(n_samples):
        for j in np.arange(n_tasks):  
            x = data['x']['task_'+str(j)][i]
            y = data['y']['task_'+str(j)][i]
            ax[i].plot(x,y, 'o')


# In[8]:


plot_data(data) 


# In[9]:


def pad_data(data_list, fill=0):
    """
    Pad tensors with fill-value such that list of variable shaped tensors can be stacked
    """
    n_samples = len(data_list)
    #first determine longest time series:
    lengths = [x.shape[0] for x in data_list]
    max_len = max(lengths)
    
    output = fill*torch.ones((n_samples, max_len))
    for i in np.arange(n_samples):
        output[i,:lengths[i]] = data_list[i]
    return output
        


# In[10]:


def merge_tasks(data):
    """
    For GPyTorch MGP, the tasks need to be merged into one tensor, and a index tensor identifies the task at hand
    Inputs:
    - data: nested dictionary, from generate_dataset()
    Returns:
    - x: [n_samples, len_all_tasks]
    - y: [n_samples, len_all_tasks] 
    - task_indices: [n_samples, len_all_tasks] 
    
    """
    n_samples = len(data['x']['task_0'])
    n_tasks = len(list(data['x'].keys()))
    
    #first determine maximal length of all tasks
    inputs_samples = [] # as final output size is not known here, we append all data to list..
    values_samples = []
    indices_samples = [] 
   
    for i in np.arange(n_samples):
        indices_list = []
        inputs_list = []
        values_list = []
        for j in np.arange(n_tasks):
            x = data['x']['task_' + str(j)][i]
            y = data['y']['task_'+str(j)][i]
            task_index = torch.full_like(x, dtype=torch.long, fill_value=j)
            #Append data per task into one list, and then concatenate it
            inputs_list.append(x)
            values_list.append(y)
            indices_list.append(task_index)
        #Get tensor of current sample
        inputs = torch.cat(inputs_list)
        values = torch.cat(values_list)
        indices = torch.cat(indices_list)

        #Append all tensors to list over samples
        inputs_samples.append(inputs)
        values_samples.append(values)
        indices_samples.append(indices)
    
    return  pad_data(inputs_samples), pad_data(values_samples), pad_data(indices_samples,fill=n_tasks).to(dtype=torch.long)
    #we pad the task indices with 99 as an exotic number, to not mix it up with real tasks
    


# In[11]:


inputs, values, indices = merge_tasks(data)


# In[12]:


# GPyTorch seems to require additional dim for inputs
inputs = inputs.unsqueeze(-1)
indices = indices.unsqueeze(-1)


# In[13]:


def prepare_train_data(data):
    inputs, values, indices = merge_tasks(data)
    inputs = inputs.unsqueeze(-1)
    indices = indices.unsqueeze(-1)
    return inputs, values, indices


# In[14]:


inputs, values, indices = prepare_train_data(data)


# In[272]:





# # MGP Model

# In[304]:


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=3)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

likelihood = gpytorch.likelihoods.GaussianLikelihood()


# ### try out set_train_data() with synthetic data to ensure that new batch inputs + values are used for posterior distribution!

# In[305]:


#start MGP layer for MGP-Sig adapter:
class MGP_Layer(MultitaskGPModel):
    def __init__(self,likelihood, num_tasks=2):
        super().__init__(None, None, likelihood, num_tasks) 
        #we don't intialize with train data for more flexibility
        likelihood.train()
        
    def forward(self, inputs, indices):
        return super(MGP_Layer, self).forward(inputs, indices)
    
    def condition_on_train_data(self, inputs, indices, values):
        self.set_train_data(inputs=(inputs, indices), targets=values, strict=False)


# In[17]:


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)


# In[314]:


class SimpleDeepModel(nn.Module):
    def __init__(self, n_input_dims):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input_dims, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.model(x)
        


# In[ ]:


class DeepSignatureModel(nn.Module):
    def __init__(self, n_input_dims):
        super().__init__()
        self.augment1 = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(8, 8, 4),
                                          kernel_size=4,
                                          include_original=True,
                                          include_time=True)
        self.model = nn.Sequential(
            nn.Linear(n_input_dims, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.model(x)


# In[336]:


class GPAdapter(nn.Module):
    def __init__(self, n_input_dims, n_mc_smps, *gp_params):
        super(GPAdapter, self).__init__()
        self.n_mc_smps = n_mc_smps
        self.mgp = MGP_Layer(*gp_params)
        self.clf = SimpleDeepModel(n_input_dims)
        #self.clf = nn.Sequential(
        #    nn.Linear(n_input_dims, 100),
        #    nn.ReLU(True),
        #    nn.Linear(100, 2),
        #    nn.ReLU(True)
        #)
        
    def forward(self, *data):
        """
        The GP Adapter takes input data as a list of 5 torch tensors (3 for train points, 2 for prediction points)
            - inputs: input points of time grid (batch, timesteps, 1)
            - indices: indices of task or channels (batch, timesteps, 1)
            - values: values (or targets) of actual observations (batch, timesteps)
            - test_inputs: query points in time (batch, timesteps, 1)
            - test_indices: query tasks for given point in time (batch, timesteps, 1)
        """
        
        self.posterior = self.gp_forward(*data)
        Z = self.draw_samples(self.posterior, self.n_mc_smps)
        return self.clf(Z)
                
    def gp_forward(self, *data):
        #Unpack data:
        inputs, indices, values, test_inputs, test_indices = [*data]
        
        #Condition MGP on training data:
        self.mgp.condition_on_train_data(inputs, indices, values)

        #Return posterior distribution:
        return self.mgp(test_inputs, test_indices)

    def draw_samples(self, posterior, n_mc_smps):
        #Draw monte carlo samples (with gradient) from posterior:
        return posterior.rsample(torch.Size([n_mc_smps])) #mc_samples form a new (outermost) dimension
        
    def parameters(self):
        return list(self.mgp.parameters()) + list(self.clf.parameters())
        
    def train(self, mode=True):
        """
        only set classifier to train mode, MGP always in eval mode for posterior draws
        """
        if mode:
            super().train()
            self.mgp.eval()
        else: 
            super().train(False)
            
    def eval(self):
        """
        eval simply calls eval of super class (which in turn activates train with False)
        """
        super().eval()


# In[311]:


def augment_labels(labels, n_samples):
    """
    Function to expand labels for multiple MC samples in the GP Adapter 
        - takes tensor of size [n]
        - returns expanded tensor of size [n_mc_samples, n]
    """
    return labels.unsqueeze(-1).expand(labels.shape[0],n_samples).transpose(1,0)


# ## Training a simple MGP Adapter

# In[334]:


# Training Parameters:
n_epochs = 50

# Setting up parameters of GP:
n_mc_smps = 10
n_input_dims = test_inputs.shape[1]
num_tasks=n_tasks+1 #augment tasks with dummy task for imputed 0s for tensor format
likelihood = gpytorch.likelihoods.GaussianLikelihood()
#likelihood.eval()

# Initializing GP adapter model (for now assuming imputing to equal length time series)
model = GPAdapter(n_input_dims, n_mc_smps, likelihood, num_tasks)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()}, 
], lr=0.1)

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
    loss = loss_fn(logits.flatten(0,1), y_true.long().flatten())
    
    #Optimize:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        print(f'Epoch {i}, Train Loss: ', loss.item())


# In[333]:





# In[103]:


# Here we have two iterms that we're passing in as train_inputs
#model = MultitaskGPModel((inputs, indices), values, likelihood, num_tasks=n_tasks+1)

#model = MultitaskGPModel(None, None, likelihood, num_tasks=n_tasks+1)
model = MGP_Layer(likelihood, num_tasks=n_tasks+1) 


# In[104]:


# Find optimal model hyperparameters
model.train() 
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    #model.update_data(inputs=(inputs, indices), targets=values)
    model.set_train_data(inputs=(inputs, indices), targets=values, strict=False)
    optimizer.zero_grad()
    output = model(inputs, indices)
    loss = -mll(output, values).mean()
    loss.backward()
    print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()


# In[16]:


# Train MGP Layer:
model.train() 
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    #model.update_data(inputs=(inputs, indices), targets=values)
    model.condition_on_train_data(inputs, indices, values)
    optimizer.zero_grad()
    output = model(inputs, indices)
    loss = -mll(output, values).mean()
    loss.backward()
    print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()


# In[132]:


output.rsample(torch.Size([3])).shape


# In[183]:


obj = likelihood(model(inputs,indices))


# In[125]:


# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
#with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #observed_pred_y1 = likelihood(model(test_x, test_i_task1))
    #dummy = torch.sum(likelihood(model(test_inputs, test_indices)).mean)
    #dummy.backward() 
    observed_pred_y = likelihood(model(test_inputs, test_indices))


# In[172]:


# model.zero_grad()
model.covar_module.raw_lengthscale.grad


# In[201]:





# In[126]:


def extract_train_data(data, indices, sample, task):
    """
    Utiliy function to extract data (values or inputs) of tensor for a single sample and task
    """
    return data.squeeze()[sample,:][indices.squeeze()[sample,:] == task]


def extract_test_data(data, n_query, sample, task):
    return data[sample,n_query*(task):n_query*(task+1)]


# In[131]:


from matplotlib.pyplot import cm
colors=cm.rainbow(np.linspace(1,0,n_tasks+2))

# Initialize plots
f, ax = plt.subplots(n_samples, figsize=(10, 18))

# Unpack predictions:
mean = observed_pred_y.mean #try out random samples with: mean = observed_pred_y.sample(torch.Size([3]))[2]
lower, upper = observed_pred_y.confidence_region()

for i in np.arange(n_samples):
    for j in np.arange(n_tasks):
        # Plot training data as black stars
        #get train input data of current sample and task:
        train_x = extract_train_data(inputs, indices, i, j)
        train_y = extract_train_data(values, indices, i, j)
        #Plot train data (irregularly observed)
        ax[i].plot(train_x.detach().numpy(), train_y.detach().numpy(), '*', c=colors[j])
        
        test_x = extract_test_data(test_inputs.squeeze(), n_query, i, j)
        test_y_mean = extract_test_data(mean, n_query, i, j)
        
        ax[i].plot(test_x.detach().numpy(), test_y_mean.detach().numpy(), 'b', c=colors[j]) #, 'b'
        
        test_y_lower = extract_test_data(lower, n_query, i, j)
        test_y_upper = extract_test_data(upper, n_query, i, j)
        
        ax[i].fill_between(test_x.detach().numpy(), test_y_lower.detach().numpy(), test_y_upper.detach().numpy(), alpha=0.2, color=colors[j])
        
        ax[i].set_xlabel(f'Sample {i}')
ax[0].set_title('GPyTorch Hadamard MGP with global parameters (shared over samples)')


# In[204]:


f.savefig('plots/hadamard_MGP.pdf')


# In[242]:


for param_name, param in model.named_parameters():
    print(f'Name: {param_name} Value = {param.data}')


# In[87]:





# In[29]:


test_x = torch.linspace(0, 1, 51).repeat(4,1).unsqueeze(-1)


# In[30]:


test_x.shape


# In[ ]:


# Initialize plots
f, ax = plt.subplots(n_samples, figsize=(20, 15))

n_tasks = train_y.shape[-1]
for i in np.arange(n_samples):
    for j in np.arange(n_tasks):
        # Plot training data as black stars
        ax[i,j].plot(train_x[i].detach().numpy(), train_y[i,:,j].detach().numpy(), 'k*')
        
        # Predictive mean as blue line
        ax[i,j].plot(test_x[i].numpy(), mean[i,:,j].numpy(), 'b')
        # Shade in confidence
        ax[i,j].fill_between(test_x[i].numpy().squeeze(), lower[i,:,j].numpy(), upper[i,:,j].numpy(), alpha=0.5)
        #ax[i,j].set_ylim([-3, 3])
        #ax[i,j].legend(['Observed Data', 'Mean', 'Confidence'])
        #ax[i,j].set_title('Observed Values (Likelihood)')


# In[147]:


Train_x = train_x.unsqueeze(0).repeat(4,1)


# In[212]:


shifts = torch.tensor([1,-0.5,0,0.7]).unsqueeze(-1)
frequencies = torch.tensor([2,0.5,3,1]).unsqueeze(-1)


# In[231]:


Train_y = torch.stack([
    torch.sin(Train_x * (2*frequencies* math.pi) + shifts ) + torch.randn(Train_x.size()) * 0.2,
    torch.cos(Train_x * (3*frequencies* math.pi) + shifts ) + torch.randn(Train_x.size()) * 0.2,
], -1)


# In[244]:


a = torch.tensor(float('nan')) 


# In[249]:


Train_y[0,0,0] = float('nan')


# In[252]:


def plot_data(Train_x, Train_y):
    n_samples = Train_x.shape[0]
    fig, ax = plt.subplots(n_samples, figsize=(10,10))
    for i, (x,y) in enumerate(zip(Train_x, Train_y)):
        ax[i].plot(x,y)
        


# In[253]:


plot_data(Train_x, Train_y)


# In[142]:


plt.plot(train_x, train_y)


# # Kronecker MGP here

# In[92]:


#MGP Model Class
class MultitaskGPModel(gpytorch.models.ExactGP ):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# In[254]:


train_x = Train_x
train_y = Train_y


# In[255]:


train_x = train_x.unsqueeze(-1)


# In[256]:


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y, likelihood)


# In[257]:


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 50
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y).mean() #.mean()
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()


# In[241]:


# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51).unsqueeze(0).repeat(4,1).unsqueeze(-1)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task


# In[242]:


# Initialize plots
f, ax = plt.subplots(4, 2, figsize=(20, 15))

n_samples = test_x.shape[0]
n_tasks = train_y.shape[-1]
for i in np.arange(n_samples):
    for j in np.arange(n_tasks):
        # Plot training data as black stars
        ax[i,j].plot(train_x[i].detach().numpy(), train_y[i,:,j].detach().numpy(), 'k*')
        
        # Predictive mean as blue line
        ax[i,j].plot(test_x[i].numpy(), mean[i,:,j].numpy(), 'b')
        # Shade in confidence
        ax[i,j].fill_between(test_x[i].numpy().squeeze(), lower[i,:,j].numpy(), upper[i,:,j].numpy(), alpha=0.5)
        #ax[i,j].set_ylim([-3, 3])
        #ax[i,j].legend(['Observed Data', 'Mean', 'Confidence'])
        #ax[i,j].set_title('Observed Values (Likelihood)')


# In[177]:





# In[ ]:


# Predictive mean as blue line
y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
# Shade in confidence
y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
y1_ax.set_ylim([-3, 3])
y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
y2_ax.set_ylim([-3, 3])
y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y2_ax.set_title('Observed Values (Likelihood)')

