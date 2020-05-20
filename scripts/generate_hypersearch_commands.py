import csv
import json
import os
import sys
import argparse
import pandas as pd

def get_count_to_submit(counts, n_total, dataset, method, subsampling=None, data_format=None):
    """
    Function that takes info about job and checks in 
    counts dictionary how many runs are still missing
    (= less than n_total)
    """
    count = None #initialization 

    if subsampling is not None:
        # append subsampling type to dataset. chose this format to prevent varying
        # levels of hierarchies
        dataset = dataset + '/' + subsampling
    if data_format is not None:
        # same for imputation scheme here
        method = data_format + method
    if dataset in counts.keys():
        if method in counts[dataset].keys():
            count = counts[dataset][method]
    if count is None: 
        count = 0 # if we don't find any count, we assume there is no completed job 
    
    #return, how many counts are missing until n_total
    return max(n_total - count, 0)    

def format_counts(count):
    return f' n_calls={count} n_random_starts={count}'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resubmit', action='store_true', default=True, 
        help='whether completed runs should be checked and NOT resubmitted, but only remaining jobs')
    args = parser.parse_args()
    # number of runs for each method:
    n_total = 20
    eval_str = 'evaluation_metric=balanced_accuracy' #string to set eval metric (necessary for multiclass datasets)
    
    #Set up paths:
    exp_dir = 'exp'
    fit_module = 'hyperparameter_search'
    fit_module_path = os.path.join(exp_dir, fit_module + '.py')

    datasets = ['PenDigits', 'LSST', 'CharacterTrajectories', 'Physionet2012']
    gp_models = ['GP_mc_SignatureModel', 'GP_mom_SignatureModel', 'GP_mc_GRUSignatureModel', 'GP_mom_GRUSignatureModel', 'GP_mom_GRUModel', 'GP_mc_GRUModel', 'GP_mom_DeepSignatureModel', 'GP_mc_DeepSignatureModel' ]
    imputed_models = ['ImputedSignatureModel', 'ImputedRNNSignatureModel', 'ImputedRNNModel', 'ImputedDeepSignatureModel']
     
    data_formats = ['zero', 'linear', 'forwardfill', 'causal', 'indicator' ] #only for imputed models
    subsamplers = ['LabelBasedSubsampler', 'MissingAtRandomSubsampler']
    model_types = ['GP', 'imputed'] #we distinguish between those two types of models
    preprocessing = False
    resubmit_failed_jobs = args.resubmit #True
    exclude_original = 'overrides.model__parameters__include_original=False'

    # In case we resubmit failed jobs, read dictionary listing the counts of completed jobs:
    if resubmit_failed_jobs:
        with open('scripts/completed_run_counts.json', 'r') as f:
            counts = json.load(f) #dict that contains {dataset: { model: count, .. }, .. } 
    else:
        count = n_total

    # Create command files for both model types seperately, GP and imputed models
    commands = []
    for model_type in model_types:
        for dataset in datasets:
            if preprocessing:
                outfile = os.path.join('scripts', 'commands', f'command_{dataset}_{model_type}_preprocessing.csv')
            else:
                outfile = os.path.join('scripts', 'commands', f'command_{dataset}_{model_type}_hypersearches.csv')
        
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
                    if preprocessing:
                        # only start all imputation schemes as preprocessing (for one model)
                        if model == "ImputedSignatureModel":
                            #we need additional loop over imputation strategies
                            for data_format in data_formats:
                                #write command to outfile, this one is just to run the preprocessing!
                                command = f'pipenv run python {fit_module_path} with {model} {dataset} {data_format} n_calls=1 n_random_starts=1 overrides.n_epochs=1'
                                commands.append(command)
                    else:
                        #determine if to exlude_original has to be set (only in signature models)
                        if 'Signature' in model:
                            suffix = exclude_original
                        else:
                            suffix = ''
                        #actual runs here
                        if model_type == 'imputed':
                            #for imputed models, we need additional loop over imputation strategies
                            for data_format in data_formats:
                                #define output directory of current hypersearch experiment
                                outdir = os.path.join('experiments', fit_module, dataset, data_format + model)
                                #count to resubmit:
                                if resubmit_failed_jobs:
                                    count = get_count_to_submit(counts, n_total, dataset, model, data_format=data_format)
                                if count == 0: #dont add invalid commands
                                    continue 
                                count_f = format_counts(count)
                                command = f'pipenv run python {fit_module_path} -F {outdir} with {model} {dataset} {data_format} {count_f} {suffix}' 
                                commands.append(command)
                        else:
                            #GP models
                            outdir = os.path.join('experiments', fit_module, dataset, model)
                            
                            #count to resubmit:
                            if resubmit_failed_jobs:
                                count = get_count_to_submit(counts, n_total, dataset, model)
                            if count == 0: #dont add invalid commands
                                    continue 
                            count_f = format_counts(count)
                            command = f'pipenv run python {fit_module_path} -F {outdir} with {model} {dataset} {count_f} {suffix}' 
                            commands.append(command)
 
            else: #UEA datasets here ..         
                for subsampler in subsamplers:
                    for model in models: 
                        if preprocessing:
                            # only start all imputation schemes as preprocessing (for one model)
                            if model == "ImputedSignatureModel":
                                #we need additional loop over imputation strategies
                                for data_format in data_formats:
                                    #write command to outfile, this one is just to run the preprocessing!
                                    if dataset == 'PenDigits':
                                        print('Using custom subsampling for PenDigits due to short lengths')
                                        #write pipenv run python command
                                        command = f'pipenv run python {fit_module_path} with {subsampler}{dataset} {model} {dataset} {data_format} n_calls=1 n_random_starts=1 overrides.n_epochs=1' 
                                    else:
                                        command = f'pipenv run python {fit_module_path} with {subsampler} {model} {dataset} {data_format} n_calls=1 n_random_starts=1 overrides.n_epochs=1'
                                    commands.append(command)        
                        else:
                            if model_type == 'imputed':
                                #for imputed models, we need additional loop over imputation strategies
                                for data_format in data_formats:
                                    #define output directory of current hypersearch experiment
                                    outdir = os.path.join('experiments', fit_module, dataset, subsampler, data_format + model)
                                    
                                    #count to resubmit:
                                    if resubmit_failed_jobs:
                                        count = get_count_to_submit(counts, n_total, dataset, model, subsampler, data_format)
                                    if count == 0: #dont add invalid commands
                                        continue 
                                    count_f = format_counts(count)

                                    if dataset == 'PenDigits':
                                        print('Using custom subsampling for PenDigits due to short lengths')
                                        #write pipenv run python command
                                        command = f'pipenv run python {fit_module_path} -F {outdir} with {subsampler}{dataset} {model} {dataset} {data_format} {eval_str} {count_f}' 
                                    else:
                                        #write pipenv run python command 
                                        command = f'pipenv run python {fit_module_path} -F {outdir} with {subsampler} {model} {dataset} {data_format} {eval_str} {count_f}' 
                                    commands.append(command)
                            else:
                                #define output directory of current GP hypersearch experiments
                                outdir = os.path.join('experiments', fit_module, dataset, subsampler, model)
                                
                                #count to resubmit:
                                if resubmit_failed_jobs:
                                    count = get_count_to_submit(counts, n_total, dataset, model, subsampler)
                                if count == 0: #dont add invalid commands
                                        continue 
                                count_f = format_counts(count)

                                if dataset == 'PenDigits':
                                    print('Using custom subsampling for PenDigits due to short lengths')
                                    #write pipenv run python command
                                    command = f'pipenv run python {fit_module_path} -F {outdir} with {subsampler}{dataset} {model} {dataset} {eval_str} {count_f}' 
                                else: 
                                    command = f'pipenv run python {fit_module_path} -F {outdir} with {subsampler} {model} {dataset} {eval_str} {count_f}' 
                                commands.append(command)
                
            #Write commands to outfile:
            commands_out = pd.Series(commands)
            #If preprocessing only write imputed ones:
            if not (preprocessing and model_type=='GP'): 
                commands_out.to_csv(outfile, index=False, header=False) 
            commands = [] #reset commands 
       
