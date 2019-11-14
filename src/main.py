import argparse 
import numpy as np
import os
from IPython import embed
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tempfile import TemporaryFile

from dataset import get_ucr_dataset
from preprocessing import subsample
from interpolation import gp_interpolation, interpolate_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', 
                    default='../data/UCRArchive_2018',
                    help='Path to input data directory')
parser.add_argument('--output_path', 
                    default='../data/processed',
                    help='Path to processed data directory')
parser.add_argument('--dataset', 
                    default=10,
                    help='Dataset index pointing to one of 128 UCR datasets')
parser.add_argument('--thres', 
                    default=0.5, type=float, 
                    help='Threshold for subsampling. Probability of dropping an observation')
parser.add_argument('--interpol', 
                    default='GP',
                    help='Interpolation Scheme after subsampling: [GP, linear] ')

def main():
    # Parse Arguments and unpack the args:
    args = parser.parse_args()
    input_path = args.input_path # path to UCR datasets 
    output_path = args.output_path # path to processed dataset 
    dataset_index = args.dataset # index, to currently used UCR dataset
    thres = args.thres # threshold for subsampling
    interpol = args.interpol # interpolation scheme

    datasets = os.listdir(input_path) #list of all available UCR datasets
    dataset = datasets[dataset_index] 
    #Load current UCR dataset
    X_train, y_train, X_test, y_test = get_ucr_dataset(input_path, dataset) #'ItalyPowerDemand')

    np.random.seed(42)
    
    X_int = interpolate_dataset(X_train, thres, interpolation_type=interpol, plot_sample=4)

    
    file_path = os.path.join(output_path, dataset, interpol, 'dropped_'+str(thres) ) 
    os.makedirs(file_path)
    file_name = os.path.join(file_path, 'X_interpolated.npy') 
    with open(file_name, 'wb') as f:
        np.save(f, X_int)

if __name__ in "__main__":
    main()

