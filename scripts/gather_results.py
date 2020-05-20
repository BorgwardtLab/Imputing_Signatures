import os
import sys 
import json
import argparse
import pandas as pd
import numpy as np
from IPython import embed
from tabulate import tabulate
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt   



test_metrics = [
     'testing.balanced_accuracy',
     'testing.auroc_weighted',
     'testing.accuracy',
     'testing.auprc',
     'testing.auroc'  
    ] 

model_names = {
    'GP_mc_GRUModel': 'GP-RNN',
    'GP_mc_GRUSignatureModel': 'GP-RNNSig',
    'GP_mc_SignatureModel': 'GP-Sig',
    'GP_mc_DeepSignatureModel': 'GP-DeepSig',
    'GP_mom_GRUModel': 'GP_PoM-RNN',
    'GP_mom_GRUSignatureModel': 'GP_PoM-RNNSig',
    'GP_mom_SignatureModel': 'GP_PoM-Sig',
    'GP_mom_DeepSignatureModel': 'GP_PoM-DeepSig',
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
    'validation.auprc': 'val-AP',
    'n_params': 'Number of Parameters' 
}

def process_experiment(base_path, exp_path, heads):
    metrics_path = os.path.join(base_path, exp_path, heads[0])
    run_path = os.path.join(base_path, exp_path, heads[1])
    config_path = os.path.join(base_path, exp_path, heads[2])

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
        seed = get_seed(config_path)
        output['seed'] = seed #relevant to keep track of repetitions
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

def select_test_metrics(metrics, data):
    """
    Util function to check which subset of the provided test metrics is available in the current data dictionary
    """
    found = False
    eval_metrics = []
    for metric in metrics:
        if metric in data.keys():
            eval_metrics.append(metric)
            found = True
    return found, eval_metrics


def get_seed(path):
    with open(path, 'r') as f:
        data = json.load(f)
    seed = data['seed']
    return seed

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


def gather_results_in_dict(base_paths, useful_heads, path_depth=6):
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
                    
                    valid_depth = path_depth if experiment[0] == 'Physionet2012' else path_depth + 1
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
                        #    results[exp_path_tail] = [] #this list is simply a safety measure if for some reason two jobs receive the same path name
                        #results[exp_path_tail].append(output)
                            results[exp_path_tail] = output
                        else:
                            raise ValueError(f'Trying to overwrite existing result with same path! {exp_path_tail}')
    return results


def process_run_dict(result_dict, path_depth=6):
    out_dict = defaultdict(list)  
    for key, value in result_dict.items():
        key_split = key.split('/')
        #in case we have subsampling, key_split has len 6, otherwise 5
        if len(key_split) == path_depth: #4 for repetitions
            dataset = os.path.join(key_split[0], key_split[1])
            method = key_split[2]
        else:
            dataset = key_split[0]
            method = key_split[1]

        if dataset not in out_dict.keys():
            out_dict[dataset] = []
        value.update(path = key) #add base path to restart jobs for repetitions
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
                    counts[dataset][method] += 1 #if result is a list of runs: len(result)
    return counts

def count_seeds(out_dict):
    """ 
    Function to count all completed seeds, to determine which have yet to be submitted / resubmitted.
    """
    counts = defaultdict(dict)
    for dataset in out_dict.keys(): #dataset
        if dataset not in counts.keys():
            counts[dataset] = defaultdict()
        for run in out_dict[dataset]: #looping over list of runs
            for method, result in run.items(): #run is a dict with method as key and dictionary of results as value
                if method not in counts[dataset].keys():
                    counts[dataset][method] = [result['seed']]
                else:
                    counts[dataset][method].append( result['seed']) 
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
                found, metric = determine_metric(metrics, result) #result[0] 
                print(f'Using {metric}')
                if not found:
                    raise ValueError(f'No valid eval metric found for the following job: {run}')
                if method not in counts[dataset].keys():
                    counts[dataset][method] = defaultdict()
                if 'count' not in counts[dataset][method].keys():
                    counts[dataset][method]['count'] = 0
                    best = 0
                    #for res in result: res --> result 
                    counts[dataset][method]['count'] += 1 
                    if result[metric] > best:
                        best = result[metric]
                        best_res = result
                    counts[dataset][method]['best'] = best_res  
                elif counts[dataset][method]['count'] > n_counts:
                    continue
                else: 
                    #for res in result: #loop over possible multiple runs
                    counts[dataset][method]['count'] += 1
                    if result[metric] > counts[dataset][method]['best'][metric]:
                        counts[dataset][method]['best'] = result 
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

def convert_to_df(data, repetitions=False, n_repetitions=5, param_flag=False):
    """Convert best runs dictionary to pd dataframe which can be transformed to tex table or easy to plot.
        If repetitions=True, #n_repetitions results are also added as seperate records
        If param_flag=True (requires repetitions), param df is compiled
    """
    if param_flag and not repetitions:
        raise ValueError('param df only computed for reptitions! --> set repetitions=True')
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
                if metric in ['path', 'seed', 'count']:
                    continue
                if repetitions:
                    model_name = model_names[model]
                    imputation, model_name = model_name.split('-') 
                    if not param_flag:
                        found_reps = len(data[dataset][model][metric]['raw']) 
                        if found_reps < n_repetitions: 
                            print(f'For {dataset}{subsampling} {model} found {found_reps} completed repetitions!')    
                        for rep in np.arange(found_reps): # cave: this rep must not coincide with the exact rep nr that was used in the command!
                            record = {  'long_dataset': os.path.join(dataset_name, subsampling),
                                        'dataset':      dataset_name,
                                        'subsampling':  subsampling_names[subsampling],
                                        'model':        model_name,
                                        'imputation':   imputation,
                                        'metric':       metric_names[metric], 
                                        'value':        [ data[dataset][model][metric]['raw'][rep] ] ,
                                        'repetition':   rep + 1
                                 }
                            record_df = pd.DataFrame(record)
                            out_df = out_df.append(record_df)
                    else: #if param flag and repetitions
                        record = {  'long_dataset': os.path.join(dataset_name, subsampling),
                                    'dataset':      dataset_name,
                                    'subsampling':  subsampling_names[subsampling],
                                    'model':        model_name,
                                    'imputation':   imputation,
                                    'metric':       metric_names[metric], 
                                    'value':        [ data[dataset][model][metric] ]
                                 }
                        record_df = pd.DataFrame(record)
                        out_df = out_df.append(record_df) 
                else:
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
    if repetitions:
        return df
    else: 
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

def aggregate_repetitions(out_dict, n_counts=5, include_incomplete=False, completed=None):
    counts = defaultdict(dict) 
    for dataset in out_dict.keys(): #dataset
        if dataset not in counts.keys():
            counts[dataset] = defaultdict()
        for run in out_dict[dataset]: #looping over list of runs
            for method, result in run.items(): #run is a dict with method as key and dictionary of results as value
                #first determine available test metrics to aggregate over:
                found, metrics = select_test_metrics(test_metrics, result)
                #print(f'Using {metrics}')
                if not found:
                    raise ValueError(f'No valid test metric found for the following job: {run}')
                if method not in counts[dataset].keys():
                    counts[dataset][method] = defaultdict()
                    counts[dataset][method]['count'] = 0
                    for metric in metrics:
                        counts[dataset][method][metric] = defaultdict() 
                        counts[dataset][method][metric]['raw'] = [ result[metric] ]
                    counts[dataset][method]['count'] += 1
                elif counts[dataset][method]['count'] >= n_counts:
                    continue
                else: 
                    for metric in metrics:
                        counts[dataset][method][metric]['raw'].append(result[metric])
                    counts[dataset][method]['count'] += 1
                    if include_incomplete: #also aggregate incomplete runs
                        curr_n_counts = min( completed[dataset][method], n_counts)
                    else:
                        curr_n_counts = n_counts 
                    if counts[dataset][method]['count'] == curr_n_counts:
                        for metric in metrics:
                            #print(f'Aggregating {dataset} {method} {metric}.. ')
                            # we finished gathering raw repetition results, and aggregate now
                            counts[dataset][method][metric]['raw'] = np.array(counts[dataset][method][metric]['raw'])
                            counts[dataset][method][metric]['mean'] =  counts[dataset][method][metric]['raw'].mean()
                            counts[dataset][method][metric]['std'] =  counts[dataset][method][metric]['raw'].std()
    return counts


def plot_df(df, params=None, path='results/plots', hue='model', main=False):
    """
    - Create grouped barplot and save to path/barplot.pdf
    - hue: which concept should be inside the grouping of the barplot
    - main: indicator whether smaller plot (only showing hypersearch objective) for main paper

    """
    # do plotting here
    

    hue_order = ['Sig', 'RNNSig', 'DeepSig', 'RNN']
    order = ['zero', 'lin', 'ff', 'causal', 'ind', 'GP', 'GP_PoM']
    if hue == 'model':
        x = 'imputation'
    elif hue == 'imputation':
        x = 'model'
        tmp = hue_order
        hue_order = order
        order = tmp 

    for dataset in df['long_dataset'].unique():
        data = df[df['long_dataset'] == dataset]
        plt.clf() 
        figsize=(10,10) 
        plt.figure(figsize=figsize)        
        if main:
            #only plot metric that was hyperparam search objective:
            data = data[data['metric'].isin(['BAC', 'Average Precision'])]
            curr_metric = data['metric'].iloc[0]                              
            data = data.rename(columns={'value': curr_metric} )
            row = None 
        else:
            curr_metric = 'value'
            row = 'metric'
        
        g = sns.catplot(x=x, y=curr_metric, row=row, hue=hue, data=data,  
            kind="bar", height=2, aspect=2, order=order, hue_order=hue_order, palette=sns.color_palette("Paired", 8), errwidth=0.75, capsize=0.05) #muted #bar
 
        #if params is not None:
        #    if not main:
        #        raise ValueError('param plotting only implemented for main figure with one eval metric')
        #    curr_par = params[params['long_dataset'] == dataset]
        #    curr_metric = curr_par['metric'].iloc[0]
        #    curr_par = curr_par.rename(columns={'value': curr_metric} )
        #    row = None
        #    #add parameters to plot
        #    g.map(sns.catplot, x=x, y=curr_metric, row=row, hue=hue, kind="bar", height=2, aspect=2, order=order, hue_order=hue_order, palette=sns.color_palette("Paired", 8), data=curr_par) 
        if main:
            plot_path = os.path.join(path, f"barplot_main_{dataset.replace('/','-')}.pdf")
        else:
            plot_path = os.path.join(path, f"barplot_{dataset.replace('/','-')}.pdf")

        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plt.clf()

        if params is not None:
            plt.figure(figsize=figsize)        
            curr_par = params[params['long_dataset'] == dataset]
            curr = curr_par['metric'].iloc[0]
            curr_par = curr_par.rename(columns={'value': curr} )
            
            g = sns.catplot(x=x, y=curr, row=row, hue=hue, data=curr_par, 
                    kind="bar", height=2, aspect=2, order=order, hue_order=hue_order, palette=sns.color_palette("Paired", 8), errwidth=0.75, capsize=0.05) #muted #bar
            g.set(yscale="log")           
            plot_path = os.path.join(path, f"barplot_main_params_{dataset.replace('/','-')}.pdf")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            plt.clf()

def get_run_params(path):
    """ function that reads out the number of parameter for a given run path
        - input: path (relative, starting from: experiments/train_model/<path>
        (cout.txt is the file where the parameters are written to, this will 
        be appended to path here)
    """ 
    filepath = os.path.join('experiments', 'train_model', path, 'cout.txt')
    key = 'Number of trainable Parameters:'  
    with open(filepath, 'r') as f:
        found = False
        while not found:
            l = f.readline()
            if key in l:
                found = True    
                param_line = l
    if not found:
        return False, None
    else:
        p_string = param_line.split(':')[1].rstrip('\n')
        return True, int(p_string)
 
def gather_parameters(out_dict):
    counts = defaultdict(dict) 
    for dataset in out_dict.keys(): #dataset
        if dataset not in counts.keys():
            counts[dataset] = defaultdict()
        for run in out_dict[dataset]: #looping over list of runs
            for method, result in run.items(): #run is a dict with method as key and dictionary of results as value
                #first determine available test metrics to aggregate over:
                found, n_params = get_run_params(result['path'])
                if not found:
                    raise ValueError(f'Number of parameters not found for the following job: {run}')
                if method not in counts[dataset].keys():
                    counts[dataset][method] = defaultdict()
                    counts[dataset][method]['n_params'] = n_params
                else:
                    assert n_params == counts[dataset][method]['n_params']
    return counts



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', action='store_true', default=False, 
        help='if true, reptitions are gathered, otherwise hypersearch runs (default: false)')
    args = parser.parse_args()
        
    repetitions = args.repetitions
 
    # Parameters:
    if repetitions: #gather and aggregate results of repetitions
        base_paths = ['experiments/train_model']
        path_depth = 4
        run_name = 'repetitions' 
    else: #gather hyperparameter search runs
        base_paths = ['experiments/hyperparameter_search']
        path_depth = 6
        run_name = 'run' 
    useful_heads = ['metrics.json', 'run.json', 'config.json']

    #raw results (only selecting the best stage of run --> without selecting the best run per method!)
    results = gather_results_in_dict(base_paths, useful_heads, path_depth)
   
    ## TODO: got until here with repetitions! need to count completed repes and dump count dict for generation of repetition jobs

    out_dict = process_run_dict(results, path_depth)

    #Simply count which method has how many completed runs
    counts = count_runs(out_dict)
 
    with open(f'scripts/completed_{run_name}_counts.json', 'w') as f:
        json.dump(counts, f)
    #additionally, print it:
    print(json.dumps(counts, indent=4))

    if repetitions:
        #count seeds (for resubs, to ensure that each repetition has been completed)
        seed_counts = count_seeds(out_dict)
        with open(f'scripts/completed_{run_name}_seed_counts.json', 'w') as f:
            json.dump(seed_counts, f)
    
        #finish after this:
        agg = aggregate_repetitions(out_dict, n_counts=5, include_incomplete=True, completed=counts)
        params = gather_parameters(out_dict)
        df = convert_to_df(agg, repetitions=True)
        param_df = convert_to_df(params, repetitions=True, param_flag=True) 
        plot_df(df, hue ='imputation')
        plot_df(df, params=param_df, hue ='imputation', main=True)
        embed()
        sys.exit()
 
    #Find the test performance of the best run per method (in terms of validation performance)
    best_runs = get_best_runs(out_dict)
 
    #Convert this output into a dictionary directly usable for tex via pandas
    best_runs_dict = get_best_runs_dict(best_runs)
    
    #Convert best run dict to df for tex table
    dfs = convert_to_df(best_runs_dict)

    #Write each dataset result to tex table:
    #Dump raw results: 
    with open('results/raw_results.json', 'w') as f:
        json.dump(results, f)
    
    #Dump best runs: 
    with open('results/best_runs.json', 'w') as f:
        json.dump(best_runs, f)
 
