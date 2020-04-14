import csv
import json
import os
from IPython import embed
import sys
import argparse
import pandas as pd
import numpy as np
from generate_hypersearch_commands import get_count_to_submit

#def get_best_configs(data):
#    configs = {'GP': [], 'imputed': []}
#    for dataset in data.keys():
#        for model in data[dataset].keys():
#            path = data[dataset][model]['best']['path']
#            if 'GP' in path:
#                configs['GP'].append(path)
#            else:
#                configs['imputed'].append(path)
#    return configs

def continue_loop(count):
    if count == 0:
        print('job already completed, skipping..')
        return True
    elif count < 5:
        print(f'Warning: {count} repetitions missing!') 
        return False
    else:
        return False

def grep_config(data, dataset, model, subsampler=None):
    """
    find best config matching the remaining args (dataset, model, etc)
    """
    if subsampler is not None:
        dataset = os.path.join(dataset, subsampler)
    try: 
        path = data[dataset][model]['best']['path']
        config = os.path.join('experiments', 'hyperparameter_search', path, 'config.json')
    except:
        config = 'NaN'
    return config 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resubmit', action='store_true', default=True, 
        help='whether completed runs should be checked and NOT resubmitted, but only remaining jobs')
    args = parser.parse_args()
    # number of repetitions for each method:
    n_repetitions = 5
    
    #Set up paths:
    exp_dir = 'exp'
    fit_module = 'train_model'
    fit_module_path = os.path.join(exp_dir, fit_module + '.py')

    datasets = ['PenDigits', 'LSST', 'CharacterTrajectories', 'Physionet2012']
    gp_models = ['GP_mc_SignatureModel', 'GP_mom_SignatureModel', 'GP_mc_GRUSignatureModel', 'GP_mom_GRUSignatureModel', 'GP_mom_GRUModel', 'GP_mc_GRUModel', 'GP_mom_DeepSignatureModel', 'GP_mc_DeepSignatureModel' ]
    imputed_models = ['ImputedSignatureModel', 'ImputedRNNSignatureModel', 'ImputedRNNModel', 'ImputedDeepSignatureModel']
     
    data_formats = ['zero', 'linear', 'forwardfill', 'causal', 'indicator' ] #only for imputed models
    subsamplers = ['LabelBasedSubsampler', 'MissingAtRandomSubsampler']
    model_types = ['GP', 'imputed'] #we distinguish between those two types of models
    resubmit_failed_jobs = args.resubmit #True

    #read best configs file:
    with open('results/best_runs.json', 'r') as f:
        best_runs = json.load(f) #dict that contains {dataset: { model: count, .. }, .. } 
    
    # In case we resubmit failed jobs, read dictionary listing the counts of completed jobs:
    if resubmit_failed_jobs:
        with open('scripts/completed_repetitions_counts.json', 'r') as f:
            counts = json.load(f) #dict that contains {dataset: { model: count, .. }, .. } 
    else:
        count = n_repetitions
 
    # Create command files for both model types seperately, GP and imputed models
    commands = []
    for model_type in model_types:
        for dataset in datasets:
            outfile = os.path.join('scripts', 'commands', f'command_{dataset}_{model_type}_repetitions.csv')
        
            #Remove existing outfile to not append commands there..
            if os.path.exists(outfile):
                os.remove(outfile)
                print('Out file already exists, removing it to create a new one..')

            #########################################################
            #First: Loop over datasets, models, and subsampling type:
            #########################################################

            # to this end, determine which models to use:
            if model_type == 'GP':
                models = gp_models
            else:
                models=imputed_models
            
            print(f'Looping over {model_type} models: {models}')
            if dataset == 'Physionet2012':
                #process Physionet without subsampling (already irregular sampled)
                #also drop original channels (due to high dimensionality!)
                for model in models: 
                    #actual runs here
                    if model_type == 'imputed':
                        #for imputed models, we need additional loop over imputation strategies
                        for data_format in data_formats:
                            imputed_model = data_format + model
                            #define output directory of current hypersearch experiment
                            outdir = os.path.join('experiments', fit_module, dataset, imputed_model)
                            config = grep_config(best_runs, dataset, imputed_model)
                            if resubmit_failed_jobs:
                                count = get_count_to_submit(counts, n_repetitions, dataset, model, data_format=data_format)
                                if continue_loop(count):
                                    continue  
                            for r in np.arange(n_repetitions):
                                command = f'python {fit_module_path} -F {outdir} with {config} rep{r+1}' 
                                commands.append(command)
                    else:
                        #GP models
                        outdir = os.path.join('experiments', fit_module, dataset, model)
                        config = grep_config(best_runs, dataset, model)
                        if resubmit_failed_jobs:
                            count = get_count_to_submit(counts, n_repetitions, dataset, model)
                            if continue_loop(count):
                                continue  
                        for r in np.arange(n_repetitions):
                            command = f'python {fit_module_path} -F {outdir} with {config} rep{r+1}' 
                            commands.append(command)
                        
            else: #UEA datasets here ..         
                for subsampler in subsamplers:
                    for model in models: 
                        if model_type == 'imputed':
                            #for imputed models, we need additional loop over imputation strategies
                            for data_format in data_formats:
                                imputed_model = data_format + model
                                #define output directory of current hypersearch experiment
                                outdir = os.path.join('experiments', fit_module, dataset, subsampler, imputed_model)
                                config = grep_config(best_runs, dataset, imputed_model, subsampler) #here we additionally feed the subsampler!
                                if resubmit_failed_jobs:
                                    count = get_count_to_submit(counts, n_repetitions, dataset, model, subsampler, data_format)
                                    if continue_loop(count):
                                        continue  
                                #write python command 
                                for r in np.arange(n_repetitions):
                                    command = f'python {fit_module_path} -F {outdir} with {config} rep{r+1}' 
                                    commands.append(command)
                        else:
                            #define output directory of current GP hypersearch experiments
                            outdir = os.path.join('experiments', fit_module, dataset, subsampler, model)
                            config = grep_config(best_runs, dataset, model, subsampler) #here we additionally feed the subsampler!
                            if resubmit_failed_jobs:
                                count = get_count_to_submit(counts, n_repetitions, dataset, model, subsampler)
                                if continue_loop(count):
                                    continue  
                            for r in np.arange(n_repetitions):
                                command = f'python {fit_module_path} -F {outdir} with {config} rep{r+1}' 
                                commands.append(command)
                
            #Write commands to outfile:
            commands_out = pd.Series(commands)
            #If preprocessing only write imputed ones:
            commands_out.to_csv(outfile, index=False, header=False) 
            commands = [] #reset commands 
       
