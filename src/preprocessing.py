import numpy as np

"""
Preprocessing Utilities

"""
def subsample(X,y,prob=0.5):
    """
    Function to subsample time series (times X and values y) according to sampling probability prob
    """
    mask_ind = np.random.choice(2,len(X), p=[prob, 1-prob])
    mask = mask_ind.astype(bool)
    return X[mask], y[mask]


