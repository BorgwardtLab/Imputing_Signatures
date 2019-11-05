import numpy as np

from utils import get_ucr_dataset
from IPython import embed
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def subsample(X,y,prob=0.5):
    """
    Function to subsample time series (times X and values y) according to sampling probability prob
    """
    mask_ind = np.random.choice(2,len(X), p=[prob, 1-prob])
    mask = mask_ind.astype(bool)
    return X[mask], y[mask]

X_train, y_train, X_test, y_test = get_ucr_dataset('data/UCRArchive_2018', 'ItalyPowerDemand') #'ItalyPowerDemand')

np.random.seed(42)

#select sample
i = 4
prob = 0.8
X_max = len(X_train[i,:])
X_grid = np.arange(X_max)
X = np.atleast_2d(X_grid).T
y = X_train[i,:]
X_old, y_old = X, y
X,y = subsample(X,y,prob)

embed()

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel,
                              n_restarts_optimizer=15)
# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)

X_pred = np.atleast_2d( 
            np.linspace(0,X_max,50) 
         ).T
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE

plt.figure()
plt.plot(X,y, 'o', label='Subsampled and observed')
plt.plot(X_old,y_old, '.', label='Original training points')
plt.plot(X_pred, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([X_pred, X_pred[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()




