import argparse 
import numpy as np
import os
from IPython import embed
from tempfile import TemporaryFile
from time import time

from dataset import get_ucr_dataset
from preprocessing import standardize, impute
from interpolation import gp_interpolation, interpolate_dataset

def main(parser):
    np.random.seed(42)

    # Parse Arguments and unpack the args:
    args = parser.parse_args()
    input_path = args.input_path # path to UCR datasets 
    output_path = args.output_path # path to processed dataset 
    dataset_index = args.dataset # index, to currently used UCR dataset
    used_format = args.used_format
    thres = args.thres # threshold for subsampling
    interpol = args.interpol # interpolation scheme

    datasets = os.listdir(input_path) #list of all available UCR datasets
    dataset = datasets[dataset_index] 
    #try variable length dataset with setting: dataset = 'PLAID'
    #try dataset with Nans:
    dataset = 'DodgerLoopDay' 
    
    #Load current UCR dataset
    X_train, y_train, X_test, y_test = get_ucr_dataset(input_path, dataset, used_format) #'ItalyPowerDemand')
    
    #First preprocessing steps:
    #-------------------------
    #1. Apply standardization here (still with NaNs!), CAVE: conversion to pd DataFrame!
    X_train, X_test = standardize(X_train, X_test) #now we have dfs
 
    for data, name in zip([X_train, X_test], ['X_train', 'X_test']):
        #2. Impute the few Nans (carry forward, then 0):
        data = impute(data)
        
        #3. Subsample and interpolate again in np array format
        start = time() # take time as this step takes the longest.. 
        X_int = interpolate_dataset(data.values, thres, interpolation_type=interpol, plot_sample=None)
        print(f'Interpolating the {dataset} {name} dataset took {time() - start} seconds.')
        file_path = os.path.join(output_path, dataset, interpol, 'dropped_'+str(thres) ) 
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, name + '.npz') 
        np.savez_compressed(file_name, X=X_int)
        
    #loading:
    #loaded = np.load(file_name)
    #X_int = loaded['X']

if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', 
                        default='data/Univariate_ts', #'data/UCRArchive_2018'
                        help='Path to input data directory')
    parser.add_argument('--output_path', 
                        default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--dataset', 
                        default=10,
                        help='Dataset index pointing to one of 128 UCR datasets')
    parser.add_argument('--used_format', 
                        default='ts',
                        help='Used Data Format [tsv, ts]')
    parser.add_argument('--thres', 
                        default=0.5, type=float, 
                        help='Threshold for subsampling. Probability of dropping an observation')
    parser.add_argument('--interpol', 
                        default='GP',
                        help='Interpolation Scheme after subsampling: [GP, linear] ')

    main(parser)

