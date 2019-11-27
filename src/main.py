import argparse 
import numpy as np
import os
from IPython import embed
import sys
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import get_ucr_dataset, check_equal_length, equal_length_datasets
from interpolation import gp_interpolation, interpolate_dataset
from preprocessing import main as main_preprocessing

def main(parser):
    np.random.seed(42)

    # Parse Arguments and unpack the args:
    args = parser.parse_args()
    input_path = args.input_path # path to UCR datasets 
    output_path = args.output_path # path to processed dataset 
    dataset_index = int(args.dataset) # index, to currently used UCR dataset
    dataset_name = args.dataset_name
    used_format = args.used_format
    thres = float(args.thres/100) # threshold for subsampling
    min_length = int(args.min_length) # minimal length of time series for dataset to use
    interpol = args.interpol # interpolation scheme
    overwrite = args.overwrite
    show_sample = args.show_sample
    if show_sample is not None:
        show_sample = int(show_sample)

    datasets = equal_length_datasets(min_length)
    print(f'Number of datasets fulfilling min_length of {min_length}: {len(datasets)}')

    if dataset_name:
        print('Using dataset as provided by string argument')
        dataset = dataset_name
    else:
        print('Using dataset as provided by index argument')
        dataset = datasets[dataset_index] 
    print(f'SETUP: Dataset = {dataset}, threshold = {thres}, Interpolation = {interpol}, format = {used_format}')

    #skip variable length datasets:
    if not check_equal_length(dataset):
        print('dataset not contained in list of equal length time series datasets, skipping for now..')
        sys.exit()
 

    #Determine if output already exists, if yes skip..
    file_path = os.path.join(output_path, dataset, interpol, 'dropped_'+str(thres) ) 
    output_file = os.path.join(file_path, 'X_train.npz') 
    output_file2 = os.path.join(file_path, 'X_test.npz') 
    only_test = False #bool whether only test split has to processed (e.g. large dataset which did not fit into job time)
    if os.path.exists(output_file) and os.path.exists(output_file2):
        pass
    else:
        #Run preprocessing script and feed it the current parser
        main_preprocessing(parser)
    ### CONTINUE HERE
    
    #Load current UCR dataset
    X_train, y_train, X_test, y_test = get_ucr_dataset(input_path, dataset, used_format) #'ItalyPowerDemand')
    
    #First preprocessing steps:
    #-------------------------
    #1. Apply standardization here (still with NaNs!), CAVE: conversion to pd DataFrame!
    X_train, X_test = standardize(X_train, X_test) #now we have dfs
 
    for data, name in zip([X_train, X_test], ['X_train', 'X_test']):
        if (only_test == True and name == 'X_train'):
            print('Skipping train split, only test split needs to be processed..')
            continue
        #2. Impute the few Nans (carry forward, then 0):
        data = impute(data)
        
        #3. Subsample and interpolate again in np array format
        start = time() # take time as this step takes the longest.. 
        X_int, plot_indicator = interpolate_dataset(data.values, thres, interpolation_type=interpol, plot_sample=show_sample)
        print(f'Interpolating the {dataset} {name} dataset took {time() - start} seconds.')
        if plot_indicator is not None:
            plt.savefig(f'plots/{interpol}_interpolation_{dataset}_sample_{show_sample}.pdf') 
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, name + '.npz') 
        np.savez_compressed(file_name, X=X_int)

        
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
    parser.add_argument('--dataset_name', 
                        default=None,
                        help='Dataset Name (instead of going via index of list of datasets)')
    parser.add_argument('--used_format', 
                        default='ts',
                        help='Used Format: [ts] currently only ts imlemented!')
    parser.add_argument('--thres', 
                        default=50, type=float, 
                        help='Threshold for subsampling. Percentage observations to KEEP')
    parser.add_argument('--min_length',
                        default=100, type=int, 
                        help='minimal length of time series for using the dataset (as subsampling very short time series is meaningless')
    parser.add_argument('--interpol', 
                        default='GP',
                        help='Interpolation Scheme after subsampling: [GP, linear] ')
    parser.add_argument('--overwrite', 
                        action='store_true', default=False,
                        help='To overwrite existing npz output files')
    parser.add_argument('--show_sample', 
                        default=None, 
                        help='If specificed, display the interpolation of this sample (by index) [e.g. 3] ')

    main(parser)

