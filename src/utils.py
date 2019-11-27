import os
import numpy as np

def load_data(dataset_dir, train=True):
    """ Function to load train or test data (from preprocessed npz format)
    """
    filename = 'X_train.npz' if train else 'X_test.npz'
    data_path = os.path.join(dataset_dir, filename)
    data = np.load(data_path)
    return data['X']

def load_train_and_test(dataset_dir):
    X_train = load_data(dataset_dir, train=True)
    X_test = load_data(dataset_dir, train=False)
    return X_train, X_test


