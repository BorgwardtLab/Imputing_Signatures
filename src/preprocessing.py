import numpy as np
import pandas as pd
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


def standardize(X_train, X_test): 
    """
    Standardize both train and test data with train stats (can handle NaNs with pandas)
    Inputs: Np arrays,
    Outputs: pd DataFrames!
    """ 
    #Convert to stacked df, for easier NaN handling. stacked for overall stats
    X_tr = pd.DataFrame(X_train)
    X_te = pd.DataFrame(X_test)
    #Compute stats:
    mean = X_tr.stack().mean()
    std = X_tr.stack().std()
    #Apply standardization:
    Z_tr = (X_tr - mean) / std
    Z_te = (X_te - mean) / std
    return Z_tr, Z_te 

def impute(X):
    """
    Takes pd Dataframe (already standardized), and imputes using forward filling and then 0-imputation (of remaining NaNs)
    """ 
    X = X.ffill()
    X = X.fillna(0)
    return X


