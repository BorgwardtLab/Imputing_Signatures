"""Module with util functions."""
import os
from typing import List
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

# import tensorflow as tf
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.util import nest

# from ..utils import n_parallel


DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

DATA_DIR = os.environ.get('DATA_DIR', DEFAULT_DATA_DIR)


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
        for key in instances[0].keys()
    }

    # Pad instances to max shape
    max_shapes = {
        key: get_max_shape(value) for key, value in dict_of_lists.items()
    }
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
            padding = np.stack(
                [np.zeros_like(padding_shape), padding_shape], axis=1)
            padded = np.pad(
                instance,
                padding,
                mode='constant',
                constant_values=padding_values[key]
            )
            padded_output[key].append(padded)

    # Combine instances into individual arrays
    combined = {
        key: torch.tensor(np.stack(values, axis=0))
        for key, values in padded_output.items()
    }

    return combined


def add_measurement_masking(measurements):
    """Add an additional tensor containing 1s when a value was measured."""
    # Mask NaN values in measurement tensors and create multi-hot vector
    measurement_indicators = tf.math.is_finite(measurements)
    measurements = tf.where(
        measurement_indicators, measurements, tf.zeros_like(measurements))
    return measurements, measurement_indicators


def get_delta_t(times, measurements, measurement_indicators):
    """Add a delta t tensor which contains time since previous measurement.

    Args:
        times: The times of the measurements (tp,)
        measurements: The measured values (tp, measure_dim)
        measurement_indicators: Indicators if the variables was measured or not
            (tp, measure_dim)

    Returns:
        delta t tensor of shape (tp, measure_dim)

    """
    scattered_times = times * tf.cast(measurement_indicators, tf.float32)
    dt_array = tf.TensorArray(tf.float32, tf.shape(measurement_indicators)[0])
    # First observation has dt = 0
    first_dt = tf.zeros(tf.shape(measurement_indicators)[1:])
    dt_array = dt_array.write(0, first_dt)

    def compute_dt_timestep(i, last_dt, dt_array):
        last_dt = tf.where(
            measurement_indicators[i-1],
            tf.fill(tf.shape(last_dt), tf.squeeze(times[i] - times[i-1])),
            times[i] - times[i-1] + last_dt
        )
        dt_array = dt_array.write(i, last_dt)
        return i+1, last_dt, dt_array

    n_observations = tf.shape(scattered_times)[0]
    _, last_dt, dt_array = tf.while_loop(
        lambda i, a, b: i < n_observations,
        compute_dt_timestep,
        loop_vars=[tf.constant(1), first_dt, dt_array]
    )
    dt_tensor = dt_array.stack()
    dt_tensor.set_shape(measurements.get_shape())
    return dt_tensor
