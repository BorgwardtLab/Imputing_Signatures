import argparse 
from collections import defaultdict
import numpy as np
import os
import pandas as pd
from IPython import embed
import sys
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from dataset import get_ucr_dataset, check_equal_length, equal_length_datasets
from interpolation import gp_interpolation, interpolate_dataset
from preprocessing import main as main_preprocessing
from preprocessing import prepro_without_subsampling
from utils import load_train_and_test
from transformers import SignatureTransform

def main(parser):

    # Parse Arguments and unpack the args:
    args = parser.parse_args()
    input_path = args.input_path # path to UCR datasets 
    output_path = args.output_path # path to processed dataset 
    dataset_index = int(args.dataset) # index, to currently used UCR dataset
    dataset_name = args.dataset_name
    used_format = args.used_format
    use_subsampling = args.use_subsampling
    thres = float(args.thres/100) if use_subsampling else None # threshold for subsampling
    min_length = int(args.min_length) # minimal length of time series for dataset to use
    interpol = args.interpol if use_subsampling else None #interpolation scheme
    overwrite = args.overwrite
    n_iter_search = args.n_iter_search
    method = args.method
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

    #Check if the classification ran before succesfully (result file exists)
    result_file = os.path.join(args.result_path, method + '_' + dataset +'_'+ 'subsampling' 
        + str(use_subsampling) +  '_' + str(interpol) +'_interpolation_' + 'dropped_'+str(thres) 
        + '_n_iters_' + str(n_iter_search) + '.csv') 
    
    if os.path.exists(result_file):
        print('Classification Result already exists. Skip this job..')
        sys.exit()
    
    #Load raw data (with labels) 
    X_train_raw, y_train, X_test_raw, y_test = get_ucr_dataset(input_path, dataset, used_format)
    if thres is not None:
        print(f'Working with subsampled data of threshold {thres}')
        #If we run on subsampled data, check if it was preprocessed before, otherwise do it now.
        #Determine if output already exists 
        file_path = os.path.join(output_path, dataset, interpol, 'dropped_'+str(thres) ) 
        output_file = os.path.join(file_path, 'X_train.npz') 
        output_file2 = os.path.join(file_path, 'X_test.npz') 
        only_test = False #bool whether only test split has to processed (e.g. large dataset which did not fit into job time)
        if os.path.exists(output_file) and os.path.exists(output_file2):
            pass
        else:
            #Run preprocessing script and feed it the current parser
            main_preprocessing(parser)
        #Determine path to preprocessed data:
        dataset_dir = os.path.join(output_path, dataset, interpol, f'dropped_{thres}')
        #Load preprocessed (subsampled + imputed) data: 
        X_train, X_test = load_train_and_test(dataset_dir)
    else:
        print(f'Working with the original data (no subsampling)') 
        #Preprocess raw data as we do not work with the subsampled data:
        X_train, X_test = prepro_without_subsampling(X_train_raw, X_test_raw) 

    
    if method == 'Sig_LGBM': 
        import lightgbm as lgb

        est = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1,
                              n_estimators=100, n_jobs=4)
        pipe = Pipeline(steps=[('sig', SignatureTransform() ), ('est', est)])

        param_dist = {
        'sig__truncation': sp_randint(2, 10),
        'est__boosting_type': ['gbdt', 'dart'],
        'est__learning_rate':[0.001, 0.1, 0.05, 0.01],
        'est__reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
        'est__reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
        }
    elif method == 'Sig_kNN':
        from sklearn.neighbors import KNeighborsClassifier
        est = KNeighborsClassifier(n_jobs=4)
        pipe = Pipeline(steps=[('sig', SignatureTransform() ), ('est', est)])

        param_dist = {
        'sig__truncation': sp_randint(2, 10),
        'est__n_neighbors': sp_randint(1,10),
        'est__p': sp_randint(1,4),
        }
    elif method == 'Sig_LR':
        from sklearn.linear_model import LogisticRegression as LR
        #est = LR(solver='saga')
        est = LR()
        pipe = Pipeline(steps=[('sig', SignatureTransform() ), ('est', est)])

        param_dist = {
        'sig__truncation': sp_randint(2, 10),
        'est__penalty':['l1', 'l2'],
        'est__C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
        }
    elif method == 'DTW_kNN':
        from transformers import DTW_KNN  
        pipe = Pipeline(steps=[ ('est', DTW_KNN())])
        param_dist = {
        'est__n_neighbors': sp_randint(1, 10) }
    else:
        raise ValueError(f'Provided Method {method} not implemented! Choose from [Sig_kNN, Sig_LR, Sig_LGBM, DTW_kNN]')
    #Use defined pipeline and param_dist for randomized search:
    
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, scoring='accuracy',
                                   n_iter=n_iter_search, cv=3, iid=False)
    start = time()
    rs.fit(X_train, y_train)
    elapsed = time() - start 
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((elapsed), n_iter_search))

    y_pred = rs.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {acc}')
    #Write results to csv:
    results = defaultdict() 
    results['test_accuracy'] = [acc]
    results['thres'] = [thres]
    results['interpol'] = [interpol]
    results['use_subsampling'] = [use_subsampling]
    results['dataset'] = [dataset] 
    results['method'] = [method]
    results['n_iter_search'] = [n_iter_search]
    results['runtime'] = [elapsed]
 
    df = pd.DataFrame.from_dict(results, orient='columns')
    print(df)
    result_file = os.path.join(args.result_path, method + '_' + dataset +'_'+ 'subsampling' 
        + str(use_subsampling) +  '_' + str(interpol) +'_interpolation_' + 'dropped_'+str(thres) 
        + '_n_iters_' + str(n_iter_search) + '.csv') 
    df.to_csv(result_file,index=False)    
     
if __name__ in "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', 
                        default='data/Univariate_ts', #'data/UCRArchive_2018'
                        help='Path to input data directory')
    parser.add_argument('--output_path', 
                        default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--result_path', 
                        default='results/',
                        help='Path to experimental results')
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
                        default=50, type=int, 
                        help='Threshold for subsampling. Percentage observations to KEEP')
    parser.add_argument("--use_subsampling", action='store_true') 
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
    parser.add_argument('--method', 
                        default='Sig_kNN', type=str, 
                        help='Method to use for classification [Sig_kNN, Sig_LR, Sig_LGBM, DTW_kNN]')
    parser.add_argument('--n_iter_search', 
                        type=int, 
                        default=20,
                        help='Number of iterations in randomized search for hyperparameter optimization')
    main(parser)

