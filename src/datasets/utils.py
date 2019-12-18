"""Module with util functions."""
import os
from typing import List
import pandas as pd
import numpy as np

# import tensorflow as tf
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.util import nest

# from ..utils import n_parallel


DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

DATA_DIR = os.environ.get('DATA_DIR', DEFAULT_DATA_DIR)


def to_gpytorch_format(d):
    """Convert dictionary with data into the gpytorch format.

    Args:
        d: Dictionary with at least the following keys: time, values

    Returns:
        Dictionary where time and values are replaced with inputs, values and
        indexes.

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
    d['indexes'] = indexes
    return d


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
