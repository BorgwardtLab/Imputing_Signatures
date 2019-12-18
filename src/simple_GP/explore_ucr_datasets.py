import argparse 
import numpy as np
import os
import pandas as pd
from dataset import get_ucr_dataset
import csv 
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', 
                    default='data/Univariate_ts', #'data/UCRArchive_2018'
                    help='Path to input data directory')
parser.add_argument('--used_format', 
                        default='ts',
                        help='Used Data Format [tsv, ts]')
args = parser.parse_args()
input_path = args.input_path # path to UCR datasets 
used_format = args.used_format

datasets = os.listdir(input_path)

# Loop over all datasets:

equal_lengths = defaultdict()

for dataset in datasets:
    #Load current UCR dataset
    try:
        X_train, y_train, X_test, y_test = get_ucr_dataset(input_path, dataset, used_format) #'ItalyPowerDemand')
        #X_train = pd.DataFrame(X_train).stack() #just for checking whether it handles NaNs.. 
        print(f'Dataset {dataset}, Train/Test: Mean = {X_train.mean()}/{X_test.mean()}, Std = {X_train.std()} / {X_test.std()} ')
        equal_lengths[dataset] = X_train.shape[1] #length of time series
    except:
        print('Dataset of variable length!')
     
with open('data/equal_lengths.pkl', 'wb') as f:
    pickle.dump(equal_lengths,f)




    
