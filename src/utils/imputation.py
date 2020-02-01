import torch


def forward_fill_imputation(batch):
    """Simple forward-fill imputation. Every NaN value is replaced with the most recent non-NaN value. (And is left at
    NaN if there are no preceding non-NaN values.)"""

    old_values = batch['values']  # tensor of shape (batch, stream, channels)

    # Now forward-fill impute the missing values
    values = old_values.clone()
    time_slices = iter(values.unbind(dim=1))
    prev_time_slice = next(time_slices)
    for time_slice in time_slices:
        nan_mask = torch.isnan(time_slice)
        time_slice.masked_scatter_(nan_mask, prev_time_slice.masked_select(nan_mask))
        prev_time_slice = time_slice

    return {'time': batch['time'], 'values': values, 'label': batch['label']}


def causal_imputation(batch):
    """Performs causal imputation on the batch.

    Suppose we have a sequence of observations, (t_1, x_1, y_1, z_1), ..., (t_n, x_n, y_n, z_n), where t_i are the
    timestamps, and x_i, y_i, z_i are three channels that are observed. The different timestamps mean that this data is
    potentially irregularly sampled. Furthermore each x_i, y_i or z_i may be NaN, to represent no observation in that
    channel at that time, so the data may potentially also be partially observed. Suppose for example that the t_i, x_i
    pairs look like this:

    t_1 t_2 t_3
    x_1 NaN x_3

    (Where t_2 is presumably included because there is an observation for y_2 or z_2, but we don't show that here.)

    Then the causal imputation scheme first does simple forward-fill imputation:

    t_1 t_2 t_3
    x_1 x_1 x_3

    and then duplicates and interleaves the time and channel observations, to get:

    t_1 t_2 t_2 t_3 t_3
    x_1 x_1 x_1 x_1 x_3

    The forward-fill imputation preserves causality. The interleaving of time and channel changes means that when a
    change does occur, it does so instantaneously. In the example above, x changes from x_1 to x_3 without the value of
    t increasing.

    So for example if multiple channels are present:

    t_1 t_2 t_3 t_4
    x_1 NaN x_3 x_4
    NaN y_2 NaN y_4

    this becomes:

    t_1 t_2 t_2 t_3 t_3 t_4 t_4
    x_1 x_1 x_1 x_1 x_3 x_3 x_4
    NaN NaN y_2 y_2 y_2 y_2 y_4

    Note that we don't try to impute any NaNs at the start. Practically speaking you might want to back-fill those, but
    exactly what you want to do with those isn't something the causal imputation scheme determines; that's a choice we
    leave up to you.

    Input:
        batch: Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] should be a tensor of shape (batch, stream, 1)
            (b) batch['values'] should be a tensor of shape (batch, stream, channels)
            (c) batch['labels'] should be a tensor of shape (batch, 1)
            As each batch element may have a different length, then the shorter ones are assumed to be zero-padded out
            to the length of the longest element.
            e.g. if the number of channels is 1, then this could look like:
                 batch['time'][0] = tensor([[0.0], [1.0], [1.5], [2.0], [0.0], [0.0], [0.0], [0.0]])
               batch['values'][0] = tensor([[1.2], [2.5], [3.1], [2.3], [0.0], [0.0], [0.0], [0.0]])
               ...
                 batch['time'][4] = tensor([[0.0], [1.0], [1.5], [2.0], [2.2], [2.7], [4.5], [5.0]])

    Returns:
        A Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] will be a tensor of shape (batch, 2 * stream - 1, 1)
            (b) batch['values'] will be a tensor of shape (batch, 2 * stream - 1, channels)
            (c) batch['labels'] will be a tensor of shape (batch, 1)
    """

    old_time = batch['time']

    # Start off by forward-fill imputing the missing values
    values = forward_fill_imputation(batch)['values']

    # For the times, we want to repeat every time twice, and then drop the first (repeated) time.
    time = old_time.repeat_interleave(2, dim=1)
    time = time[:, 1:]

    # For the values, we want to repeat every value twice, and then drop the last (repeated) value.
    # This is a bit finickity because of the zero-padding of the shorter batch elements.
    values = values.repeat_interleave(2, dim=1)
    indices = old_time.squeeze(2).argmax(dim=1)
    indices *= 2
    indices += 1
    values.scatter_(1, indices.unsqueeze(1).unsqueeze(2).expand(values.size(0), 1, values.size(2)), 0)
    values = values[:, :-1]

    return {'time': time, 'values': values, 'label': batch['label']}
