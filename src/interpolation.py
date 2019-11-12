import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def interpolate_dataset(X, interpolation_fn):
    """
    Apply instance-wise interpolation on entire dataset.
    
    Inputs:
        - X: Time series values (n x t array, n instances, t time steps (non-consecutive)) 
        - interpolation_fn: function to interpolate time series instance (e.g. gp_interpolation, linear_interpolation)
    Outputs:
        - X_i : interpolated Time series data set (n x t') with t' > t 
    """
    n_instances = X.shape[0]
    instances = []
    for i in np.arange(n_instances): 
        X_instance = X[i,None,:] #slice array, s.t. instance is still 2d (for easier concatenation at the end)
        

def gp_interpolation(X,T,T_max): 
    """
    Computes Gaussian Process Interpolation on single instances (no shared 
    parameters with simple sklearn implementation available). This function
    assumes, that the input time series is irregularly sampled (subsampled), 
    s.t. the time steps are not always consecutive, e.g. T = 
    np.array([0,2,3,5,6,11])
    
    Inputs:
        - X: Time series values (1D array)
        - T: Time step indicator (1D array) 
        - T_max: Maximal Time step of original time series
    
    Outputs: 
        - X_pred: Time series values (1D array) of predictions / interpolations
        - T_pred: Time step indicator (1D array) of predictions / interpolations
        - sigma: standard deviation of predicted time series values (1D array)
    """

    #first convert time steps to sklearn GP format:
    #T = np.atleast_2d(T).T
    
    #setup kernel (hard-coded for now)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10, normalize_y=True)
    
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(T, X)

    # Make the prediction on the meshed x-axis (ask for MSE as well)

    T_pred =  np.linspace(0,T_max,50).reshape(-1,1)

    X_pred, sigma = gp.predict(T_pred, return_std=True)
    
    return X_pred, T_pred, sigma
 
