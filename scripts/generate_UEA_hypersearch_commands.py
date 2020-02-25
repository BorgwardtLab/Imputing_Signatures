import csv
import json
import os
from IPython import embed
import sys
import argparse
import pandas as pd

#Set up paths:
exp_dir = 'exp'
fit_module = 'hyperparameter_search'
fit_module_path = os.path.join(exp_dir, fit_module + '.py')

datasets = ['PenDigits', 'LSST', 'CharacterTrajectories' ]
gp_models = ['GP_mc_SignatureModel', 'GP_mom_SignatureModel', 'GP_mc_GRUSignatureModel', 'GP_mom_GRUSignatureModel', 'GP_mom_GRUModel', 'GP_mc_GRUModel' ] #GP_mom_DeepSignatureModel GP_mc_DeepSignatureModel
imputed_models = ['ImputedSignatureModel', 'ImputedRNNSignatureModel', 'ImputedRNNModel' ]
 
data_formats = ['zero', 'linear', 'forwardfill', 'causal', 'indicator' ] #only for imputed models
subsamplers = ['LabelBasedSubsampler', 'MissingAtRandomSubsampler']
model_types = ['GP', 'imputed'] #we distinguish between those two types of models
preprocessing = False
resubmit_failed_jobs = True


# In case we resubmit failed jobs, read dictionary listing the counts of completed jobs:
if resubmit_failed_jobs:
    with open('scripts/completed_run_counts.json', 'r') as f:
        counts = json.load(f)
    embed();sys.exit() 

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
 
        for subsampler in subsamplers:
            for model in models:
                if preprocessing:
                    #only start all imputation schemes as preprocessing (for one model)
                    if model == "ImputedSignatureModel":
                        #we need additional loop over imputation strategies
                        for data_format in data_formats:
                            #write command to outfile, this one is just to run the preprocessing!
                            if dataset == 'PenDigits':
                                print('Using custom subsampling for PenDigits due to short lengths')
                                #write python command
                                command = f'python {fit_module_path} with {subsampler}{dataset} {model} {dataset} {data_format} n_calls=1 n_random_starts=1 overrides.n_epochs=1' 
                            else:
                                command = f'python {fit_module_path} with {subsampler} {model} {dataset} {data_format} n_calls=1 n_random_starts=1 overrides.n_epochs=1'
                            commands.append(command)        
                else:   
                    if model_type == 'imputed':
                        #for imputed models, we need additional loop over imputation strategies
                        for data_format in data_formats:
                            #define output directory of current hypersearch experiment
                            outdir = os.path.join('experiments', fit_module, dataset, subsampler, data_format + model)
                            
                            if dataset == 'PenDigits':
                                print('Using custom subsampling for PenDigits due to short lengths')
                                #write python command
                                command = f'python {fit_module_path} -F {outdir} with {subsampler}{dataset} {model} {dataset} {data_format} evaluation_metric=balanced_accuracy' 
                            else:
                                #write python command 
                                command = f'python {fit_module_path} -F {outdir} with {subsampler} {model} {dataset} {data_format} evaluation_metric=balanced_accuracy' 
                            commands.append(command)
                    else:
                        #define output directory of current GP hypersearch experiments
                        outdir = os.path.join('experiments', fit_module, dataset, subsampler, model)
                        if dataset == 'PenDigits':
                            print('Using custom subsampling for PenDigits due to short lengths')
                            #write python command
                            command = f'python {fit_module_path} -F {outdir} with {subsampler}{dataset} {model} {dataset} evaluation_metric=balanced_accuracy' 
                        else: 
                            command = f'python {fit_module_path} -F {outdir} with {subsampler} {model} {dataset} evaluation_metric=balanced_accuracy' 
                        commands.append(command)
        
        #Write commands to outfile:
        commands_out = pd.Series(commands)
        #If preprocessing only write imputed ones:
        if not (preprocessing and model_type=='GP'): 
            commands_out.to_csv(outfile, index=False, header=False) 
        commands = [] #reset commands 
   
