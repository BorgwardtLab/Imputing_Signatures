import os
import sys 
import json
import argparse
import pandas as pd
import numpy as np
from IPython import embed
from tabulate import tabulate
from collections import defaultdict

#Renaming dictionaries:

model_names = {
    'GP_mc_GRUModel': 'GP-RNN',
    'GP_mc_GRUSignatureModel': 'GP-RNNSig',
    'GP_mc_SignatureModel': 'GP-Sig',
    'GP_mc_DeepSignatureModel': 'GP-DeepSig',
    'GP_mom_GRUModel': 'GP-RNN (PoM)',
    'GP_mom_GRUSignatureModel': 'GP-RNNSig (PoM)',
    'GP_mom_SignatureModel': 'GP-Sig (PoM)',
    'GP_mom_DeepSignatureModel': 'GP-DeepSig (PoM)',
    'causalImputedRNNModel': 'causal-RNN',
    'causalImputedRNNSignatureModel': 'causal-RNNSig',
    'causalImputedSignatureModel': 'causal-Sig',
    'causalImputedDeepSignatureModel': 'causal-DeepSig',
    'forwardfillImputedRNNModel': 'ff-RNN',
    'forwardfillImputedRNNSignatureModel': 'ff-RNNSig',
    'forwardfillImputedSignatureModel': 'ff-Sig',
    'forwardfillImputedDeepSignatureModel': 'ff-DeepSig',
    'indicatorImputedRNNModel': 'ind-RNN',
    'indicatorImputedRNNSignatureModel': 'ind-RNNSig',
    'indicatorImputedSignatureModel': 'ind-Sig',
    'indicatorImputedDeepSignatureModel': 'ind-DeepSig',
    'linearImputedRNNModel': 'lin-RNN',
    'linearImputedRNNSignatureModel': 'lin-RNNSig',
    'linearImputedSignatureModel': 'lin-Sig',
    'linearImputedDeepSignatureModel': 'lin-DeepSig',
    'zeroImputedRNNModel': 'zero-RNN',
    'zeroImputedRNNSignatureModel': 'zero-RNNSig',
    'zeroImputedSignatureModel': 'zero-Sig',
    'zeroImputedDeepSignatureModel': 'zero-DeepSig'
}
subsampling_names = {
    'LabelBasedSubsampler': 'Label-based',
    'MissingAtRandomSubsampler': 'Random',
    '': ''
} 
metric_names = {
    'testing.accuracy': 'Accuracy',
    'testing.auroc_weighted': 'w-AUROC',
    'testing.balanced_accuracy': 'BAC',
    'validation.balanced_accuracy': 'val-BAC',
    'testing.auprc': 'Average Precision', #average precision 
    'testing.auroc': 'AUROC',
    'validation.auprc': 'val-AP'  
}

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

def get_best_runs_dict(data):
    """
    Extract a dictionary which can be directly processed to tex via pandas
    which is a nested dict of the following structure:
    {dataset1: 
        { method1: 
            { metric1: 0.99,
                .. 
             } 
        } .. 
    }
    """
    out = defaultdict()
    for dataset in data.keys():
        out[dataset] = defaultdict()
        for method in data[dataset].keys():
            out[dataset][method] = defaultdict()
            for metric in data[dataset][method]['best'].keys():
                out[dataset][method][metric] = data[dataset][method]['best'][metric] 
    return out

def pivot_df(df):
    df_out = df.pivot_table(index=['subsampling', 'model'], columns='metric', values='value')
    return df_out

def highlight_best(df, top=3): #actually operates on df series 
    formats = [ [r' \mathbf{ \underline{ ',    ' }}'],
        [r' \mathbf{ ',               ' }'],
        [r' \underline{ ',            ' }']]

    top_n = df.nlargest(top).index.tolist()
    #best = df.idxmax()
    #df[best] = f'$ \mathbf{{ {df[best]:.5g} }} $'
    rest = list(df.index)
    for i, best in enumerate(top_n):
        df[best] = '$ ' + formats[i][0] + f'{df[best]:.5g}' + formats[i][1] + ' $'
        rest.remove(best)
    #this manual step was trying to get conistently 5 decimals as df.round did not do it.
    #however, there is the same issue here as well.. 
    df[rest] = df[rest].apply(lambda x: '$ {:g} $'.format(float('{:.5g}'.format(float(x))))) 
    return df

def extract_and_tex_single_datasets(df):
    """ return dict of dataset-wise results in df format"""
    dfs = defaultdict()
    datasets = df['dataset'].unique()
    for dat in datasets:
        curr_df = df.query("dataset == @dat")
        #round values:
        #curr_df['value'] = curr_df['value'].round(5)
        #pivot df to compact format (metrics as columns)
        curr_piv = pivot_df(curr_df)
        if curr_piv.index[0][0] == '':
            curr_piv = curr_piv.reset_index(level='subsampling', drop=True)
            curr_piv = curr_piv.iloc[:,:].apply(highlight_best)
        else: #using subsampling, split dfs for bolding the winners
            curr_piv = pd.DataFrame() #initialize the concatenated df of all subsamplings
            subsamplings = curr_df['subsampling'].unique()
            for subsampling in subsamplings:
                df_sub = curr_df.query("subsampling == @subsampling")
                df_sub_piv = pivot_df(df_sub)
                df_sub_piv = df_sub_piv.iloc[:,:].apply(highlight_best)
                curr_piv = curr_piv.append(df_sub_piv)
        #drop validation metrics:
        cols = list(curr_piv.columns)
        cols_to_drop = [col for col in cols if 'val' in col]
        print(cols_to_drop)
        curr_piv = curr_piv.drop(columns=cols_to_drop)
        #rearrange columns (show hypersearch obj first)
        cols = list(curr_piv.columns)
        if 'Accuracy' in cols: #multivariate dataset
            new_order = [2,1,0]
            curr_piv = curr_piv[curr_piv.columns[new_order]] 
        dfs[dat] = curr_piv

        #Write table to result folder
        curr_piv.to_latex(f'results/tables/{dat}.tex', escape=False)
    return dfs

def convert_to_df(data):
    """Convert best runs dictionary to pd dataframe which can be transformed to tex table """
    out_df = pd.DataFrame() 
    for dataset in data.keys():
        if '/' in dataset:
            split = dataset.split('/')
            dataset_name = split[0]
            subsampling = split[1]
        else:
            subsampling = ''
            dataset_name = dataset
        #convert nested dictionary to df with redundant records (easier to group by)
        for model in data[dataset].keys():
            for metric in data[dataset][model].keys():
                record = {  'dataset':      dataset_name,
                            'subsampling':  subsampling_names[subsampling],
                            'model':        model_names[model],
                            'metric':       metric_names[metric], 
                            'value':       [ data[dataset][model][metric] ]
                         }
                record_df = pd.DataFrame(record)
                out_df = out_df.append(record_df) 
    # rename models:
     
    df = out_df.sort_values(['dataset','subsampling', 'metric']) 
    #if df.index[0][0] == '':
    #    df = df.drop(index=['subsampling'])
    #extract irregularly spaced and regularly spaced datasets:
    irregular = ['Physionet2012']
    is_irregular = df['dataset'].isin(irregular)
    df_irregular = df[is_irregular]
    df_regular = df[~is_irregular]
    
    dfs_irr = extract_and_tex_single_datasets(df_irregular)
    dfs_reg = extract_and_tex_single_datasets(df_regular)   
    
    #df_irregular = df_irregular.reset_index(drop=True)
    #df_irr_pivoted = pivot_df(df_irregular)
    #df_reg_pivoted = pivot_df(df_regular)  
    
    return dfs_irr, dfs_reg
 
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
 
    #Find the test performance of the best run per method
    best_runs = get_best_runs(out_dict)
    #Convert this output into a dictionary directly usable for tex via pandas
    best_runs_dict = get_best_runs_dict(best_runs)

    #Convert best run dict to df for tex table
    dfs = convert_to_df(best_runs_dict)

    #Write each dataset result to tex table:
    #embed()  
    #Dump raw results: 
    with open('results/raw_results.json', 'w') as f:
        json.dump(results, f)
    
    #Dump best runs: 
    with open('results/best_runs.json', 'w') as f:
        json.dump(best_runs, f)
 
