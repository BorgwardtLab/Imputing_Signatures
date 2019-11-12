import numpy as np
import os
from IPython import embed
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from utils import get_ucr_dataset, subsample
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--path', 
                    default='../data/UCRArchive_2018',
                    help='Path to data directory')
parser.add_argument('--dataset', 
                    default=33,
                    help='Dataset index pointing to one of 128 UCR datasets')
parser.add_argument('--thres', 
                    default=0.2,
                    help='Threshold for subsampling. Probability of dropping an observation')
parser.add_argument('--interpol', 
                    default='GP',
                    help='Interpolation Scheme after subsampling: [GP, linear] ')


# Parse Arguments and unpack the args:
args = parser.parse_args()
path = args.path # path to UCR datasets 
dataset_index = args.dataset # index, to currently used UCR dataset
thres = args.thres # threshold for subsampling
interpol = args.interpol # interpolation scheme


datasets = os.listdir(path) #list of all available UCR datasets

#Load one UCR dataset
X_train, y_train, X_test, y_test = get_ucr_dataset(path, datasets[dataset_index]) #'ItalyPowerDemand')

np.random.seed(42)

#select sample
i = 4
prob = 0.8
X_max = len(X_train[i,:])
T_grid = np.arange(X_max)
T = np.atleast_2d(T_grid).T
X = X_train[i,:]
T_old, X_old = T, X
T,X = subsample(T,X,thres)

embed()

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel,
                              n_restarts_optimizer=10)
# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(T, X)

# Make the prediction on the meshed x-axis (ask for MSE as well)

T_pred = np.atleast_2d( 
            np.linspace(0,X_max,50) 
         ).T
X_pred, sigma = gp.predict(T_pred, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE

plt.figure()
plt.plot(T,X, 'o', label='Subsampled and observed')
plt.plot(T_old,X_old, '.', label='Original training points')
plt.plot(T_pred, X_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([T_pred, T_pred[::-1]]),
         np.concatenate([X_pred - 1.9600 * sigma,
                        (X_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
#plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()




