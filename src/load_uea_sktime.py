from IPython import embed
import os
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load


# Load Sktime format:

def get_uea_dataset(data_dir: str, dataset_name: str):
    '''
    Loads train and test data from a folder in which
    the UEA data sets are stored. (following sktime format)
    '''

    X_train, y_train, = load(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN.ts'))
    X_test, y_test,   = load(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST.ts'))

    return X_train, y_train, X_test, y_test


input_path = 'data/Multivariate_ts'

index = 0
datasets = os.listdir(input_path)
dataset = datasets[index]
#dataset_path = os.path.join(input_path, dataset)

X_train, y_train, X_test, y_test = get_uea_dataset(input_path, dataset) 


