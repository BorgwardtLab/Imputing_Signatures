import iisignature
import numpy as np

def ts2path(X):
    """ Convert single time series to path (by adding time axis))
        X: np array (n time steps, d dimensions)
    """
    if len(X.shape) == 1:
        X = X.reshape(-1,1)
    if X.shape[1] > 1:
        raise ValueError("Multivariate Time series not implemented yet!")
    n = X.shape[0]
    steps = np.arange(n).reshape(-1,1)
    path = np.concatenate((steps, X), axis=1)
    return path

def compute_signatures(data, trunc=6):
    """ Compute signatures of dataset of time series 
        Input:
        -data (n samples, d time steps)
        Output:
        -signatures (n_samples, n_signature_components)
    """
    n = data.shape[0]
   
    signatures = []
    for sample in data:
        #convert sample time series to path
        path = ts2path(sample)
        #compute signature of path of ts 
        sig = iisignature.sig(path, trunc)
        #convert to 2d array for easy concatenation
        
        signatures.append(sig.reshape(-1,1))
        #append to results
    return np.concatenate(signatures, axis=1).T

def to_signatures(X_train, X_test, trunc=6):
    S_train = compute_signatures(X_train, trunc=trunc)
    S_test = compute_signatures(X_test, trunc=trunc)
    return S_train, S_test

