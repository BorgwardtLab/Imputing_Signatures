import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from preprocessing import subsample

class Interpolation:
    """
    Class for interpolation object of an instance, gathering various processing stages of the input data
    """
    def __init__(self, X, thres, interpolation_fn):
        self.X = X
        self.T_max = len(self.X) #determine length of time series
        self.T_grid = np.arange(self.T_max) #create array of input time steps
        self.T = self.T_grid.reshape(-1,1) #reshape it to shape (t,1) for sklearn gp compatibility 
        self.T_sub, self.X_sub = subsample(self.T,self.X, thres) #subsample time series for irregular sampling
        ## Apply interpolation to current sample:
        self.X_pred, self.T_pred, self.sigma = interpolation_fn(self.X_sub, self.T_sub, self.T_max)



def interpolate_dataset(X, thres, interpolation_type, plot_sample=4):
    """
    Apply instance-wise interpolation on entire dataset.
    
    Inputs:
        - X: Time series values (n x t array, n instances, t time steps (non-consecutive)) 
        - interpolation_type: (e.g. GP)
    Outputs:
        - X_int : interpolated Time series data set (n x t') with t' > t 
    """
    if interpolation_type == "GP":
        interpolation_fn = gp_interpolation
    n_instances = X.shape[0]
    instances = []
    for i in np.arange(n_instances): 
        X_instance = X[i,:] #slice array, s.t. instance is still 2d (for easier concatenation at the end)
        ip = Interpolation(X_instance, thres, interpolation_fn)
        #T_max = len(X_instance) #determine length of time series
        #T_grid = np.arange(T_max) #create array of input time steps
        #T = T_grid.reshape(-1,1) #reshape it to shape (t,1) for sklearn gp compatibility 
        #T_old, X_old = T, X_instance #keep original series (before subsampling) for later visualization
        #T,X = subsample(T,X,thres) #subsample time series for irregular sampling

        ## Apply interpolation to current sample:
        #X_pred, T_pred, sigma = interpolation_fn(ip.X_sub, ip.T_sub, ip.T_max)

        if i == plot_sample:
            # plot the interpolation of sample f{plot_sample}
            plot_interpolation(ip)
    
        instances.append(ip.X_pred.reshape(1,-1)) #gather time series as row vecs (for easier concatenation)
    X_int =  np.concatenate(instances, axis=0)    

    return X_int
         

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

    T_pred =  np.linspace(0,T_max,T_max).reshape(-1,1) #(0,T_max,50) for testing

    X_pred, sigma = gp.predict(T_pred, return_std=True)
    
    return X_pred, T_pred, sigma


def plot_interpolation(ip):
    """
    Plot the function, the prediction and the 95% confidence interval based on
    the MSE
    Inputs:
        - Interpolation object 
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(ip.T_sub,ip.X_sub, 'o', label='Subsampled and observed')
    plt.plot(ip.T,ip.X, '.', label='Original training points')
    plt.plot(ip.T_pred, ip.X_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([ip.T_pred, ip.T_pred[::-1]]),
             np.concatenate([ip.X_pred - 1.9600 * ip.sigma,
                            (ip.X_pred + 1.9600 * ip.sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    #plt.ylim(-10, 20)
    plt.legend(loc='upper left')

    #plt.show()
    plt.savefig('interpolation_sample.pdf')

 
