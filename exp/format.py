from collections import defaultdict
from functools import partial
import numpy as np
import torch

def to_gpytorch_format(d, grid_spacing=1.0):
    """Convert dictionary with data into the gpytorch format.

    Args:
        d: Dictionary with at least the following keys: time, values

    Returns:
        Dictionary where time and values are replaced with inputs, values and
        indices.
    """
    time = d['time']
    del d['time']
    values = d['values']
    valid_indices = np.where(np.isfinite(values))

    inputs = time[valid_indices[0]]
    values_compact = values[valid_indices]
    indexes = valid_indices[1]
    d['inputs'] = inputs
    d['values'] = values_compact
    d['indices'] = indexes[..., np.newaxis]

    # Compute test points
    max_input = np.max(inputs[:, 0])
    min_input = np.min(inputs[:, 0])

    n_tasks = values.shape[-1]
    d['n_tasks'] = n_tasks  # is channel_dim
    test_inputs = np.arange(min_input, max_input + grid_spacing, grid_spacing)
    len_test_grid = len(test_inputs)
    test_inputs = np.tile(test_inputs, n_tasks)
    test_indices = np.repeat(np.arange(n_tasks), len_test_grid)
    d['test_inputs'] = test_inputs[:, np.newaxis].astype(np.float32)
    d['test_indices'] = test_indices[:, np.newaxis].astype(np.int64)
    return d


def get_max_shape(l):
    """Get maximum shape for all numpy arrays in list.

    Args:
        l: List of numpy arrays.

    Returns:
        Shape containing the max shape along each axis.

    """
    shapes = np.array([el.shape for el in l])
    return np.max(shapes, axis=0)


def dict_collate_fn(instances, padding_values=None):
    """Collate function for a list of dictionaries.

    Args:
        instances: List of dictionaries with same keys.
        padding_values: Dict with a subset of keys from instances, mapping them
            to the values that should be used for padding. If not defined 0 is
            used.

    Returns:
        Dictionary with instances padded and combined into tensors.

    """

    # Convert list of dicts to dict of lists
    dict_of_lists = {
        key: [d[key] for d in instances]
        for key in instances[0].keys() if key != 'n_tasks'
    }
    if 'n_tasks' in instances[0].keys():
        n_tasks = instances[0]['n_tasks']

    # Pad instances to max shape
    max_shapes = {key: get_max_shape(value) for key, value in dict_of_lists.items()}
    padded_output = defaultdict(list)
    # Pad with 0 in case not otherwise defined
    padding_values = padding_values if padding_values else {}
    padding_values = defaultdict(lambda: 0., padding_values.items())
    for key, max_shape in max_shapes.items():
        for instance in dict_of_lists[key]:
            instance_shape = np.array(instance.shape)
            padding_shape = max_shape - instance_shape
            # Numpy wants the padding in the form before, after so we need to
            # prepend zeros
            if key == 'values':
                # determine and append valid length
                instance_len = instance_shape[0]

            if 'test_' in key:
                # for GP format test inputs, test indices, put padded/invalid points
                # directly in the time series between the channel switches for easier reshaping later
                to_pad = padding_shape[0]  # number of values to pad
                if to_pad > 0:
                    instance_len = instance_shape[0]
                    # temporarily get rid of extra dim (for GP format):
                    instance = instance[:, 0]
                    time_len = int(instance_len / n_tasks)  # time_length of valid observation of current instance
                    padding = np.repeat(padding_values[key], to_pad / n_tasks)  # as to_pad counts all
                    # padded values per multi variate time series
                    total_padding = np.repeat(padding, n_tasks)
                    padded_chunk_len = time_len + len(
                        padding)  # length of instance time series channel which was padded
                    final_len = len(total_padding) + instance_len

                    padded = np.zeros(final_len)
                    for i in np.arange(n_tasks):  # for each chunk to be padded
                        instance_part = instance[i * time_len: (i + 1) * time_len]
                        padded_chunk = np.concatenate([instance_part, padding])
                        padded[i * padded_chunk_len: (i + 1) * padded_chunk_len] = padded_chunk
                        # add again the additional dim
                    padded = padded[:, np.newaxis]
                    if key == 'test_inputs':  # make sure that format is correct
                        padded = padded.astype(np.float32)
                    else:
                        padded = padded.astype(int)
                else:
                    padded = instance
            else:
                # perform standard padding
                padding = np.stack(
                    [np.zeros_like(padding_shape), padding_shape], axis=1)
                padded = np.pad(
                    instance,
                    padding,
                    mode='constant',
                    constant_values=padding_values[key]
                )
            padded_output[key].append(padded)
            if key == 'values':
                padded_output['valid_lengths'].append(instance_len)

    # Combine instances into individual arrays
    combined = {
        key: torch.tensor(np.stack(values, axis=0))
        for key, values in padded_output.items()
    }

    return combined


def get_input_transform(data_format, grid_spacing):
    """
    Util function to return input transform of dataset, depending on data format
    Args:
        - data_format: which data format to use depends on model
            'GP' for models using gpytorch
            'imputed' for imputed baselines
        - grid_spacing: number of hours between each query point / or imputed point depending on format
    """
    if data_format == 'GP':
        return partial(to_gpytorch_format, grid_spacing=grid_spacing)
    elif data_format == 'imputed':
        raise NotImplementedError('Pre-Imputed dataset not implemented yet!')
    else:
        raise ValueError('No valid data format provided!')


def get_collate_fn(data_format, n_input_dims):
    """
    Util function to return collate_fn which might depend on data format / used model
    Args:
        - data_format: which data format to use depends on model
            'GP' for models using gpytorch
            'imputed' for imputed baselines
        - n_input_dims: number of input dims, the gpytorch implementation uses a dummy task for padded
            values in the batch tensor (due to zero indexing it's exactly n_input_dims)
    """
    if data_format == 'GP':
        return partial(dict_collate_fn, padding_values={'indices': n_input_dims, 'test_indices': n_input_dims})
    elif data_format == 'imputed':
        raise NotImplementedError('Pre-Imputed dataset not implemented yet!')
    else:
        raise ValueError('No valid data format provided!')
