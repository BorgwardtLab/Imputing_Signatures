import os
import sys 
import json
import argparse
import numpy as np
from IPython import embed

from collections import defaultdict

def process_experiment(base_path, exp_path, heads):
    metrics_path = os.path.join(base_path, exp_path, heads[0])
    run_path = os.path.join(base_path, exp_path, heads[1])
    if not all(os.path.exists(path) for path in [metrics_path, run_path]):
        print(f'Either run oder metrics missing in {exp_path}')
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

def get_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    eval_metric = 'validation.average_precision_score.macro' 
    test_metrics = ['testing.average_precision_score.macro', 'testing.roc_auc_score.macro']
    best_step = np.argmax(data[eval_metric]['values'])
    result_dict = defaultdict()
    
    valid = True
    for metric in test_metrics:
        result_dict[metric] = data[metric]['values'][best_step]
        if result_dict[metric] is None:
            valid = False
    return result_dict, valid


def gather_results_in_dict(base_paths, useful_heads):
    results = defaultdict(list)           
    for base_path in base_paths:

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
                    
                    #a valid experiment containing model run info has len of 6
                    if len(experiment) < 6:
                        continue
                    elif len(experiment) > 6:
                        print('Found experiment with more than 6 levels!')
                    exp_path = '/'.join(experiment)
                    exp_path_split = os.path.split(exp_path)
                    exp_path_tail = exp_path_split[0] 
                    exp_path_head = exp_path_split[1] 
                    #determine if given experiment was completed
                    valid, output = process_experiment(base_path, exp_path_tail, useful_heads) 
                    if valid:
                        if exp_path_tail not in results.keys():
                            results[exp_path_tail] = []
                        results[exp_path_tail].append(output)
    return results

if __name__ == "__main__":
    
    # Parameters:
    base_paths = ['exp_runs/hyperparameter_search', 
                  'exp_runs/batched_runs/batch0/hyperparameter_search',
                  'hypersearch_runs'] 
    useful_heads = ['metrics.json', 'run.json']


    results = gather_results_in_dict(base_paths, useful_heads)
    
    with open('gathered_runs.json', 'w') as f:
        json.dump(results, f)
 
