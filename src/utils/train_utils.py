import torch
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def augment_labels(labels, n_samples):
    """Expand labels for multiple MC samples in the GP Adapter.

    Args:
         Takes tensor of size [n]

    Returns:
        expanded tensor of size [n_mc_samples, n]

    """
    return labels.expand(labels.shape[0], n_samples).transpose(1, 0)

def build_regularly_spaced_grid(inputs, indices, n_samples, n_tasks, grid_spacing):
    """Build a regularly spaced grid based on the GP inputs.

    Args:
        inputs: Values where the GP gets data.
        indices: Indices corresponding to the tasks. Here indices == n_tasks
            will be interpreted as the padded auxilary task.
        n_samples: Number of samples to draw for each instance.
        n_tasks: Number of tasks (not including the auxillary task!)
        grid_spacing: Spacing in between grid points.

    Returns:
        inputs, indices of a regularly spaced grid for each instance.

    """
    non_padded_values = indices != n_tasks
    masked_max = torch.where(
        non_padded_values, inputs, torch.full_like(inputs, float('-inf')))
    masked_min = torch.where(
        non_padded_values, inputs, torch.full_like(inputs, float('inf')))
    max_inputs, _ = torch.max(masked_max, 1)
    min_inputs, _ = torch.min(masked_min, 1)
    max_diff, _ = torch.max(max_inputs - min_inputs, 0)
    # Add one because we want the borders to be included
    max_len = ((torch.ceil(max_diff / grid_spacing) + 1) * n_tasks).long()
    print(max_len)

    out_inputs = []
    out_indices = []
    for min_val, max_val in zip(min_inputs, max_inputs):
        # Build array of structure: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] (this
        # is in the case of 3 tasks)
        cur_inputs = torch.arange(
            min_val.item(), (max_val + grid_spacing).item(), grid_spacing)
        n_inputs = cur_inputs.size()[0]
        cur_inputs = torch.flatten(
            cur_inputs.unsqueeze(-1).repeat([1, n_tasks]))

        # Build array of structure [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        cur_indices = torch.arange(
            0, n_tasks, dtype=torch.int).repeat([n_inputs])

        # Padding
        padding_len = (max_len - cur_indices.size()[0]).item()
        padded_inputs = F.pad(cur_inputs, [0, padding_len], value=0.)
        padded_indices = F.pad(cur_indices, [0, padding_len], value=n_tasks)

        # Append to list
        out_inputs.append(padded_inputs)
        out_indices.append(padded_indices)

    out_inputs = torch.stack(out_inputs, dim=0)
    out_indices = torch.stack(out_indices, dim=0)
    return out_inputs, out_indices























