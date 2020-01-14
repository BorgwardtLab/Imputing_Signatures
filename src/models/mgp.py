import gpytorch
import torch
import torch.nn as nn

# Exact Hadamard Multi-task Gaussian Process Model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
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


# MGP Layer for Neural Network using MultitaskGPModel
class MGP_Layer(MultitaskGPModel):
    def __init__(self,likelihood, num_tasks=2):
        super().__init__(None, None, likelihood, num_tasks) 
        #we don't intialize with train data for more flexibility
        likelihood.train()
        
    def forward(self, inputs, indices):
        return super(MGP_Layer, self).forward(inputs, indices)
    
    def condition_on_train_data(self, inputs, indices, values):
        self.set_train_data(inputs=(inputs, indices), targets=values, strict=False)

# MGP Adapter
class GPAdapter(nn.Module):
    def __init__(self, clf, n_input_dims, n_mc_smps, *gp_params):
        super(GPAdapter, self).__init__()
        self.n_mc_smps = n_mc_smps
        # num_tasks includes dummy task for padedd zeros
        self.n_tasks = gp_params[-1] - 1
        self.mgp = MGP_Layer(*gp_params)
        self.clf = clf #(self.n_tasks)
        #more generic would be something like: self.clf = clf(n_input_dims) #e.g. SimpleDeepModel(n_input_dims)

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

        #draw sample in MGP format (all tasks in same dimension)
        Z = self.draw_samples(self.posterior, self.n_mc_smps)

        #reshape such that tensor has timesteps and tasks/channels in independent dimensions for Signature network:
        Z = self.channel_reshape(Z)

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

    def channel_reshape(self, X):
        """
        reshaping function required as hadamard MGP's output format is not directly compatible with subsequent network
        """
        X_reshaped = X.view(X.shape[:-1] 
                            + torch.Size([self.n_tasks]) 
                            + torch.Size([int(X.shape[-1] / self.n_tasks)]) 
                           )
        # finally, swap last two dims: timestep and channel dim for Signature Augmentations
        X_reshaped = X_reshaped.transpose(-2,-1)
        X_reshaped = X_reshaped.flatten(0,1) #SigNet requires 3 dim setup, so we flatten out the mc dimension with batch
        return X_reshaped
