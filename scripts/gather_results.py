import os
import sys 
import json
import argparse
import pandas as pd
import numpy as np
from IPython import embed

from collections import defaultdict

def process_experiment(base_path, exp_path, heads):
    metrics_path = os.path.join(base_path, exp_path, heads[0])
    run_path = os.path.join(base_path, exp_path, heads[1])
    if not all(os.path.exists(path) for path in [metrics_path, run_path]):
        print(f'Either run or metrics missing in {exp_path}')
        return False, None
    try:
        valid_run = get_run_validity(run_path)
    except:
        print(f'Exception occured at: {base_path}/{exp_path}')
        return False, False
    
    if not valid_run:
        #print(f'Invalid run due to run status {run_path}')
        return False, None
    #print(f'Found valid run {run_path}')

    output, valid_results = get_results(metrics_path)
    if not valid_results:
        return False, None
    else:
        return True, output

def get_run_validity(path):
    with open(path, 'r') as f:
        data = json.load(f)
    status = data['status']
    if status == 'COMPLETED':
        return True
    else: 
        return False

def determine_metric(metrics, data):
    """
    Util function to check which of the provided eval metrics is available in the current data dictionary
    """
    found = False
    eval_metric = None
    for metric in metrics:
        if metric in data.keys():
            eval_metric = metric
            found = True
            break
    return found, eval_metric
    
def get_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    eval_metrics = ['validation.balanced_accuracy', 'validation.auprc'] 
    found_metric, eval_metric = determine_metric(eval_metrics, data)
    found_metric = False
    for metric in eval_metrics:
        if metric in data.keys():
            eval_metric = metric
            found_metric = True
            break
    if not found_metric:
        raise ValueError(f'None of the specified eval metrics available in the following path: {path}')
     
    test_metric_dict = { eval_metrics[0]: ['testing.balanced_accuracy', 'testing.auroc_weighted', 'testing.accuracy', eval_metrics[0] ], 
                         eval_metrics[1]: ['testing.auprc', 'testing.auroc', eval_metrics[1] ]
    }
    best_step = np.argmax(data[eval_metric]['values'])
    result_dict = defaultdict()
    
    valid = True
    for metric in test_metric_dict[eval_metric]:
        result_dict[metric] = data[metric]['values'][best_step]
        if result_dict[metric] is None:
            valid = False
            print(f'{metric} not found in {path}')
    return result_dict, valid


def gather_results_in_dict(base_paths, useful_heads):
    results = defaultdict(list)
    counter = 0            
    for base_path in base_paths:
        print(f'Searchin base path: {base_path}')
        for root, dirs, files in os.walk(base_path):
            for name in files:
                file_path = os.path.join(root, name)

                #only focus on useful files: 
                if name == useful_heads[0]:
                    #get full experiment path:
                    experiment_split = file_path.split('/')  
                    base_split = base_path.split('/')
                    #list of subdirs containing experiment (without basepath)
                    experiment = [e for e in experiment_split if e not in base_split]
                    
                    valid_depth = 6 if experiment[0] == 'Physionet2012' else 7
                    
                    #a valid experiment containing model run info has len of 6
                    if len(experiment) < valid_depth:
                        exp_path = '/'.join(experiment)
                        print(f'Skipping experiment: {exp_path}')
                        continue
                    elif len(experiment) > valid_depth:
                        print('Found experiment with more than the expected {valid_depth} levels!')
                    exp_path = '/'.join(experiment)
                    exp_path_split = os.path.split(exp_path)
                    exp_path_tail = exp_path_split[0] 
                    exp_path_head = exp_path_split[1] 
                    #determine if given experiment was completed
                    valid, output = process_experiment(base_path, exp_path_tail, useful_heads) 
                    if valid:
                        if exp_path_tail not in results.keys():
                            results[exp_path_tail] = [] #this list is simply a safety measure if for some reason two jobs receive the same path name
                        results[exp_path_tail].append(output)
    return results


def process_run_dict(result_dict):
    out_dict = defaultdict(list)  
    for key, value in result_dict.items():
        key_split = key.split('/')
        #in case we have subsampling, key_split has len 6, otherwise 5
        if len(key_split) == 6:
            dataset = os.path.join(key_split[0], key_split[1])
            method = key_split[2]
        else:
            dataset = key_split[0]
            method = key_split[1]

        if dataset not in out_dict.keys():
            out_dict[dataset] = []
        out_dict[dataset].append({method: value})
    return out_dict


def count_runs(out_dict):
    counts = defaultdict(dict)
    for dataset in out_dict.keys(): #dataset
        if dataset not in counts.keys():
            counts[dataset] = defaultdict()
        for run in out_dict[dataset]: #looping over list of runs
            for method, result in run.items(): #run is a dict with method as key and dictionary of results as value
                if method not in counts[dataset].keys():
                    counts[dataset][method] = 1
                else:
                    counts[dataset][method] += len(result)
    return counts


def get_best_runs(out_dict, n_counts=20, metrics = ['validation.balanced_accuracy', 'validation.auprc']):
    #pilot_test_methods = ['GP_mom_LSTMSignatureModel']     
 
    counts = defaultdict(dict) 
    for dataset in out_dict.keys(): #dataset
        if dataset not in counts.keys():
            counts[dataset] = defaultdict()
        for run in out_dict[dataset]: #looping over list of runs
            for method, result in run.items(): #run is a dict with method as key and dictionary of results as value
                #first determine current eval metric:
                found, metric = determine_metric(metrics, result[0])
                if not found:
                    raise ValueError(f'No valid eval metric found for the following job: {run}')
                if method not in counts[dataset].keys():
                    counts[dataset][method] = defaultdict()
                if 'count' not in counts[dataset][method].keys():
                    counts[dataset][method]['count'] = 0
                    best = 0
                    for res in result:
                        counts[dataset][method]['count'] += 1 
                        if res[metric] > best:
                            best = res[metric]
                            best_res = res
                    counts[dataset][method]['best'] = best_res  
                elif counts[dataset][method]['count'] > n_counts:
                    continue
                else: 
                    for res in result: #loop over possible multiple runs
                        counts[dataset][method]['count'] += 1
                        if res[metric] > counts[dataset][method]['best'][metric]:
                            counts[dataset][method]['best'] = res 
    return counts


     
if __name__ == "__main__":
    
    # Parameters:
    base_paths = ['experiments/hyperparameter_search'] 
    useful_heads = ['metrics.json', 'run.json']


    #raw results (only selecting the best stage of run --> without selecting the best run per method!)
    results = gather_results_in_dict(base_paths, useful_heads)
   
       
    out_dict = process_run_dict(results)

    #Simply count which method has how many completed runs
    counts = count_runs(out_dict)
   
    with open('scripts/completed_run_counts.json', 'w') as f:
        json.dump(counts, f)
 
    best_runs = get_best_runs(out_dict)

    embed()
    
    #Dump raw results: 
    with open('results/raw_results.json', 'w') as f:
        json.dump(results, f)
    
    #Dump best runs: 
    with open('results/best_runs.json', 'w') as f:
        json.dump(best_runs, f)
 
