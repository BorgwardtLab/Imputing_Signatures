'''
Author: Christian  Bock
(modified by Michael Moor)
Dataset utilities for UCR Archive 2018 (tsv format)
'''


import numpy as np
import os
from IPython import embed
import pickle 

import uea_ucr_datasets

def equal_length_datasets(min_length=100):
    datasets = pickle.load(open('data/equal_lengths.pkl','rb')) 
    used_datasets = [i for i,j in datasets.items() if j >= min_length]
    return used_datasets    
def check_equal_length(name):
    datasets = equal_length_datasets() 
    return True if name in datasets else False    

def gather_iterator_in_arrays(iterator):
    """ takes iterator from uea_ucr package returning tuples (x_i,y_i) 
        and converts to array (sklearn format): 
            X (n_samples, n_timesteps), 
            y (n_samples,) 
        --> only works for 1D time series of EQUAL length!
    """
    # unpack iterator:
    time_series = []
    labels = []
    time_series_length = iterator[0][0].shape[0]
    for x,y in iterator:
        if x.shape[0] != time_series_length:
            raise Exception('Current Data set of variable length, currently not implemented..')
        time_series.append(x)
        labels.append(y)
    X = np.concatenate(time_series, axis=1).T    
    y = np.array(labels)
    return X,y 
 
def strip_suffix(s, suffix):
    '''
    Removes a suffix from a string if the string contains it. Else, the
    string will not be modified and no error will be raised.
    '''

    if not s.endswith(suffix):
        return s
    return s[:len(s)-len(suffix)]

def get_ucr_dataset(data_dir: str, dataset_name: str, used_format='tsv'):
    '''
    Loads train and test data from a folder in which
    the UCR data sets are stored.
    '''
    if used_format == 'tsv':
        X_train, y_train, _ = read_ucr_data(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN.tsv'))
        X_test, y_test, _ = read_ucr_data(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST.tsv'))
    elif used_format == 'ts':
        train_iterator = uea_ucr_datasets.Dataset(dataset_name, train=True)
        test_iterator = uea_ucr_datasets.Dataset(dataset_name, train=False)
        X_train, y_train = gather_iterator_in_arrays(train_iterator)  
        X_test, y_test = gather_iterator_in_arrays(test_iterator)  
    else: 
        raise ValueError('Provided format is not among the implemented ones: [ts, tsv]')
    return X_train, y_train, X_test, y_test

def read_ucr_data(filename):
    '''
    Loads an UCR data set from a file, returning the samples and the
    respective labels. Also extracts the data set name such that one
    may easily display results.
    '''

    data = np.loadtxt(filename, delimiter='\t') #','
    Y = data[:, 0]
    X = data[:, 1:]

    # Remove all potential suffixes to obtain the data set name. This is
    # somewhat inefficient, but we only have to do it once.
    name = os.path.basename(filename)
    name = strip_suffix(name, '_TRAIN')
    name = strip_suffix(name, '_TEST')

    return X, Y, name


