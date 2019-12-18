import numpy as np
import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from collections import defaultdict
from IPython import embed

nested_dict = lambda: defaultdict(nested_dict)


def generate_dataset(n_samples=5, n_tasks=3, noise=1):
    """
    Generating Variable Length, Irregularly observed Time Series (with sine and cosine)
    Returns:
    -data dict:
        'x':
            'task_0': list of tensors
            'task_1': ..
             .. (n_tasks-1)
        'y': 
            'task_0': list of tensors
            'task_1': ..
             .. (n_tasks-1)
    list elements represent samples / instances
    """
    #determine labels (balanced)
    n_cases = int(np.floor(n_samples/2))
    n_controls = int(n_samples - n_cases)
    labels = torch.cat([torch.ones(n_cases), torch.zeros(n_controls)])
    #add signals correlated with label:
    signal = 5 #affects the frequency of the sine/cosines
    
    #first draw number of observations per sample (uniform between 20 and 50)
    lengths = torch.randint(50, 150, (n_tasks,n_samples))
    data = nested_dict()
    for i, length in enumerate(lengths): #loop over tasks
        x_list = []
        y_list = []
        for j, l in enumerate(length): #loop over samples
            x = torch.rand(l)
            if i % 2 == 0: #even tasks with sin, uneven with cos
                y = torch.sin(torch.randn(1) + x * ((5 + (signal*labels[j])*torch.randn(1))* math.pi)) + noise*torch.randn(x.size()) * 0.1
            else:
                y = torch.cos(torch.randn(1) + x * ((5+ (signal*labels[j])*torch.randn(1))* math.pi)) + noise*torch.randn(x.size()) * 0.1
            x_list.append(x)
            y_list.append(y)
        data['x']['task_' + str(i)] = x_list
        data['y']['task_' + str(i)] = y_list
    return data, labels    


def generate_test_points(n_samples=5, n_tasks=3, n_query=51):
    inputs = torch.linspace(0, 1, n_query)
    
    def get_indices(data,fill=0):
        return torch.full_like(data, dtype=torch.long, fill_value=fill)
    indices = torch.stack(
                   [ torch.cat(
                        [get_indices(inputs, i) for i in np.arange(n_tasks)]
                    ) for j in np.arange(n_samples)
                   ]
    ) 
    
    inputs = inputs.repeat(n_samples,n_tasks)

    #add extra dimension for GPytorch:
    inputs = inputs.unsqueeze(-1)
    indices = indices.unsqueeze(-1)
    return inputs, indices

def plot_data(data):
    n_samples = len(data['x']['task_0'])
    n_tasks = len(list(data['x'].keys()))
    fig, ax = plt.subplots(n_samples, figsize=(10,10))
    for i in np.arange(n_samples):
        for j in np.arange(n_tasks):  
            x = data['x']['task_'+str(j)][i]
            y = data['y']['task_'+str(j)][i]
            ax[i].plot(x,y, 'o')

def pad_data(data_list, fill=0):
    """
    Pad tensors with fill-value such that list of variable shaped tensors can be stacked
    """
    n_samples = len(data_list)
    #first determine longest time series:
    lengths = [x.shape[0] for x in data_list]
    max_len = max(lengths)
    
    output = fill*torch.ones((n_samples, max_len))
    for i in np.arange(n_samples):
        output[i,:lengths[i]] = data_list[i]
    return output
        
def merge_tasks(data):
    """
    For GPyTorch MGP, the tasks need to be merged into one tensor, and a index tensor identifies the task at hand
    Inputs:
    - data: nested dictionary, from generate_dataset()
    Returns:
    - x: [n_samples, len_all_tasks]
    - y: [n_samples, len_all_tasks] 
    - task_indices: [n_samples, len_all_tasks] 
    
    """
    n_samples = len(data['x']['task_0'])
    n_tasks = len(list(data['x'].keys()))
    
    #first determine maximal length of all tasks
    inputs_samples = [] # as final output size is not known here, we append all data to list..
    values_samples = []
    indices_samples = [] 
   
    for i in np.arange(n_samples):
        indices_list = []
        inputs_list = []
        values_list = []
        for j in np.arange(n_tasks):
            x = data['x']['task_' + str(j)][i]
            y = data['y']['task_'+str(j)][i]
            task_index = torch.full_like(x, dtype=torch.long, fill_value=j)
            #Append data per task into one list, and then concatenate it
            inputs_list.append(x)
            values_list.append(y)
            indices_list.append(task_index)
        #Get tensor of current sample
        inputs = torch.cat(inputs_list)
        values = torch.cat(values_list)
        indices = torch.cat(indices_list)

        #Append all tensors to list over samples
        inputs_samples.append(inputs)
        values_samples.append(values)
        indices_samples.append(indices)
    
    return  pad_data(inputs_samples), pad_data(values_samples), pad_data(indices_samples,fill=n_tasks).to(dtype=torch.long)
    #we pad the task indices with 99 as an exotic number, to not mix it up with real tasks

def prepare_train_data(data):
    """
    Util function to rearrange data in MGP GPyTorch digestible (hadamard) format, whereas inputs and indices require dummy dimension
    """
    inputs, values, indices = merge_tasks(data)
    inputs = inputs.unsqueeze(-1)
    indices = indices.unsqueeze(-1)
    return inputs, values, indices

def create_synthetic_dataset(n_samples=10, n_tasks=3, n_query=51, noise=3):
    #Generate irregularly spaced, asynchronous multi-variate time series:
    data,labels = generate_dataset(n_samples=n_samples, n_tasks=n_tasks, noise=noise)
    #Reformat to hadamard format:
    inputs, values, indices = prepare_train_data(data)
    #Generate test points (here all between 0 and 1 therefore independent of training data)
    test_inputs, test_indices = generate_test_points(n_samples=n_samples, n_tasks=n_tasks, n_query=n_query)
    return labels, inputs, values, indices, test_inputs, test_indices
 
if __name__ in "__main__":
    
    labels, inputs, values, indices, test_inputs, test_indices = create_synthetic_dataset() 
    embed() 

