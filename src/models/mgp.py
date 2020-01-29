import gpytorch
import torch
import torch.nn as nn


# Exact Hadamard Multi-task Gaussian Process Model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, output_device, num_tasks=2, n_devices=1, kernel='rbf'):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.output_device = output_device
        self.mean_module = gpytorch.means.ConstantMean()
        valid_kernels = ['rbf', 'ou']
        if kernel not in valid_kernels:
            raise ValueError(f'parsed kernel: {kernel} not among implemented kernels: {valid_kernels}')
        elif kernel == 'rbf':
            base_covar_module = gpytorch.kernels.RBFKernel()
        elif kernel == 'ou':
            base_covar_module = gpytorch.kernels.MaternKernel(nu=0.5)

        if n_devices > 1: #in multi-gpu setting
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_covar_module, device_ids=range(n_devices),
                output_device=self.output_device)
            #self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=3)
            base_task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=3)
            self.task_covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_task_covar_module, device_ids=range(n_devices),
                output_device=self.output_device)
        else:
            self.covar_module = base_covar_module #gpytorch.kernels.RBFKernel()
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
    def __init__(self,likelihood, num_tasks, n_devices, output_device, kernel):
        super().__init__(None, None, likelihood, output_device, num_tasks, n_devices, kernel)
        #we don't intialize with train data for more flexibility
        likelihood.train()

    def forward(self, inputs, indices):
        return super(MGP_Layer, self).forward(inputs, indices)

    def condition_on_train_data(self, inputs, indices, values):
        self.set_train_data(inputs=(inputs, indices), targets=values, strict=False)

# Custom GP Adapter:
# MGP Adapter
class GPAdapter(nn.Module):
    def __init__(self, clf, n_input_dims, n_mc_smps, sampling_type, *gp_params):
        super(GPAdapter, self).__init__()
        self.n_mc_smps = n_mc_smps
        # num_tasks includes dummy task for padedd zeros
        self.n_tasks = gp_params[1] - 1
        self.mgp = MGP_Layer(*gp_params)
        self.clf = clf #(self.n_tasks)
        #more generic would be something like: self.clf = clf(n_input_dims) #e.g. SimpleDeepModel(n_input_dims)
        self.sampling_type = sampling_type # 'monte_carlo', 'moments'
        self.return_gp = False

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

        inputs, indices, values, test_inputs, test_indices = data

        #Get regularly-spaced "latent" timee series Z:
        if self.sampling_type == 'monte_carlo':
            #draw sample in MGP format (all tasks in same dimension)
            Z = self.draw_samples(self.posterior, self.n_mc_smps)
        elif self.sampling_type == 'moments':
            #feed moments of GP posterior to classifier (mean, variance)
            Z = self.feed_moments(self.posterior)

        #reshape such that tensor has timesteps and tasks/channels in independent dimensions for Signature network:
        if self.return_gp: #useful for visualizations 
            Z, Z_raw = self._channel_reshape(Z, test_indices, return_gp=True)
        else:
            Z = self._channel_reshape(Z, test_indices)

        Z = self._collapse_tensor(Z)

        if self.return_gp:
            return self.clf(Z), Z, Z_raw
        else:
            return self.clf(Z)

    def gp_forward(self, *data):
        #Unpack data:
        inputs, indices, values, test_inputs, test_indices = data

        #Condition MGP on training data:
        self.mgp.condition_on_train_data(inputs, indices, values)

        #Return posterior distribution:
        return self.mgp(test_inputs, test_indices)

    def draw_samples(self, posterior, n_mc_smps):
        #Draw monte carlo samples (with gradient) from posterior:
        return posterior.rsample(torch.Size([n_mc_smps])) #mc_samples form a new (outermost) dimension

    def feed_moments(self, posterior):
        """
        Get mean and variance of posterior and concatenate them along the channel dimension for feeding them to feed the clf
        """
        mean = posterior.mean
        var = posterior.variance
        #return torch.cat([ mean, var ], axis=-1) # concat in channel dim
        return torch.stack([mean, var], axis=0) # stacked in a innermost dim to replace mc_smp dim

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
    
    def _channel_reshape(self, Z_raw, test_indices, return_gp=False):
        """
        reshaping function required as hadamard MGP's output format is not directly compatible with subsequent network
        batch-wise reshaping is not super easy due to padding, for now hacky & slow approach with loops. TODO: Fix this
        """
        channel_dim = self.n_tasks
        stats_dim  = Z_raw.shape[0] #either mc samples or [mean/var] dimension depending on sampling type
        batch_size = Z_raw.shape[1]
        max_len = int(Z_raw.shape[-1] / channel_dim) # max_len of time series in the batch

        #initialise resulting tensor:
        Z = torch.zeros([stats_dim, batch_size, max_len, channel_dim])
        for sample in torch.arange(stats_dim):
            for i in torch.arange(batch_size):
                Z_raw_i = Z_raw[sample, i] # GP draw array of instance i, pooling all values in single vector
                test_indices_i = test_indices[i].flatten() # array pointing to index of each value of raw gp output
                # Create output channels:
                for j in torch.arange(channel_dim):
                    z_j = Z_raw_i[test_indices_i == j] #select all (time-ordered) values that belong to task/channel j
                    curr_len = z_j.shape[0]
                    Z[sample, i, :curr_len, j] = z_j
        if return_gp:
            return Z, Z_raw
        else:
            return Z

    def _collapse_tensor(self, X):
        """
        Util function to collapse tensor depending on sampling_type
            - mc samples are joined in batch dimension
            - moments are stacked in channel dimension
        In dimensions:
            [stats_dim, batch, length, channel]
        Out dimension:
            [batch, length, channel]
        """
        if self.sampling_type == 'monte_carlo':
            return X.flatten(0,1) # merge mc_smp and batch dim
        elif self.sampling_type == 'moments':
            #X = X.transpose(0,2).tranpose(0,1)
            X = X.permute(1,2,0,3)
            return X.flatten(-2,-1) #merge channel and moments dim
