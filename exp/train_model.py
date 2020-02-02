"""Module to train a model with a dataset configuration."""

import gpytorch
import os
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch

sys.path.append(os.getcwd())

from exp.callbacks import Callback, Progressbar, LogDatasetLoss, LogTrainingLoss
from exp.format import get_input_transform, get_collate_fn
from exp.ingredients import dataset_config, model_config
from exp.utils import augment_labels, count_parameters, plot_losses


# Test for debugging sacred read-only error
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

EXP = Experiment('training', ingredients=[model_config.ingredient, dataset_config.ingredient])
EXP.captured_out_filter = apply_backspaces_and_linefeeds


@EXP.config
def cfg():
    n_epochs = 50
    batch_size = 32
    virtual_batch_size = None
    learning_rate = 5e-4
    weight_decay = 1e-3
    early_stopping = 20
    data_format = 'GP'
    grid_spacing = 1.  # determines n_hours between query points
    max_root = 25  # max_root_decomposition_size for MGP lanczos iters
    device = 'cuda'
    quiet = False
    evaluation = {'active': False, 'evaluate_on': 'validation'}
    n_mc_smps = 10 


@EXP.named_config
def rep1():
    seed = 249040430


@EXP.named_config
def rep2():
    seed = 621965744


@EXP.named_config
def rep3():
    seed = 771860110


@EXP.named_config
def rep4():
    seed = 775293950


@EXP.named_config
def rep5():
    seed = 700134501


class NewlineCallback(Callback):
    """Add newline between epochs for better readability."""
    def on_epoch_end(self, **kwargs):
        print()


@EXP.automain
def train(n_epochs, batch_size, virtual_batch_size, learning_rate, weight_decay, early_stopping, data_format,
          grid_spacing, max_root, n_mc_smps, device, quiet, evaluation, _run, _log, _seed, _rnd):
    """Sacred wrapped function to run training of model."""

    torch.manual_seed(_seed)

    try:
        rundir = _run.observers[0].dir
    except IndexError:
        rundir = None

    # Check if virtual batch size is defined and valid:
    if virtual_batch_size is not None:
        if virtual_batch_size % batch_size != 0:
            raise ValueError(f'Virtual batch size {virtual_batch_size} has to be a multiple of batch size {batch_size}') 
    
    # Define dataset transform:
    input_transform = get_input_transform(data_format, grid_spacing)

    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    train_dataset = dataset_config.get_instance(split='training', transform=input_transform)
    validation_dataset = dataset_config.get_instance(split='validation', transform=input_transform)
    test_dataset = dataset_config.get_instance(split='testing', transform=input_transform)
    
    # Determine number of input dimensions as GP-Sig models requires this parameter for initialisation
    n_input_dims = train_dataset.measurement_dims
    out_dimension = train_dataset.n_classes
    collate_fn = get_collate_fn(data_format, n_input_dims) 
    
    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120

    model = model_config.get_instance(n_input_dims, out_dimension)
    print(f'Number of trainable Parameters: {count_parameters(model)}')
    model.to(device)
   
    # Safety guard, ensure that if mc_sampling is inactive that n_mc_smps are 1 to
    # prevent unwanted label augmentation
    if not hasattr(model, 'sampling_type'):
        n_mc_smps = 1     
    elif model.sampling_type != 'monte_carlo':
        n_mc_smps = 1

    loss_fn = torch.nn.BCELossWithLogits()

    callbacks = [
        LogTrainingLoss(_run, print_progress=quiet),
        LogDatasetLoss('validation', validation_dataset, data_format, collate_fn, loss_fn, _run, batch_size, max_root,
                       n_mc_smps, early_stopping=early_stopping, save_path=rundir, device=device, print_progress=True),
        LogDatasetLoss('testing', test_dataset, data_format, collate_fn, loss_fn, _run, batch_size, max_root, n_mc_smps,
                       save_path=rundir, device=device, print_progress=False)
    ]
    if quiet:
        # Add newlines between epochs
        callbacks.append(NewlineCallback())
    else:
        callbacks.append(Progressbar())

    train_loop(model, train_dataset, data_format, loss_fn, collate_fn, n_epochs, batch_size, virtual_batch_size,
               learning_rate, n_mc_smps, max_root, weight_decay, device, callbacks)

    if rundir:
        # Save model state (and entire model)
        print('Loading model checkpoint prior to evaluation...')
        state_dict = torch.load(os.path.join(rundir, 'model_state.pth'))
        model.load_state_dict(state_dict)
    model.eval()

    logged_averages = callbacks[0].logged_averages
    logged_stds = callbacks[0].logged_stds
    loss_averages = {key: value for key, value in logged_averages.items() if 'loss' in key}
    loss_stds = {key: value for key, value in logged_stds.items() if 'loss' in key}
    if rundir:
        plot_losses(loss_averages, loss_stds, save_file=os.path.join(rundir, 'batch_monitoring.png'))
    monitoring_measures = callbacks[1].logged_averages
    print(monitoring_measures)
    monitoring_measures.update(loss_averages)
    print(monitoring_measures)
    if rundir:
        plot_losses(monitoring_measures, save_file=os.path.join(rundir, 'epoch_monitoring.png'))

    result = {key: values[-1] for key, values in logged_averages.items()}

    if evaluation['active']:
        evaluate_on = evaluation['evaluate_on']
        if evaluate_on == 'validation':
            eval_measures = callbacks[1].eval_measures
        else:
            eval_measures = callbacks[2].eval_measures
        
        result.update(eval_measures)

    return result


def execute_callbacks(callbacks, hook, local_variables):
    stop = False
    for callback in callbacks:
        # Convert return value to bool --> if callback doesn't return
        # anything we interpret it as False
        stop |= bool(getattr(callback, hook)(**local_variables))
    return stop


def train_loop(model, dataset, data_format, loss_fn, collate_fn, n_epochs, batch_size, virtual_batch_size,
               learning_rate, n_mc_smps=1, max_root=25, weight_decay=1e-5, device='cuda', callbacks=None):
    if callbacks is None:
        callbacks = []

    virtual_scaling = 1
    if virtual_batch_size is not None:
        virtual_scaling = virtual_batch_size / batch_size

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                                               pin_memory=True, num_workers=8)
    n_instances = len(dataset)
    n_batches = len(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epoch = 1
    for epoch in range(1, n_epochs + 1):
        if execute_callbacks(callbacks, 'on_epoch_begin', locals()):
            break
        optimizer.zero_grad()
        for batch, d in enumerate(train_loader):
            # if we use mc sampling, expand labels to match multiple predictions
            if n_mc_smps > 1:
                y_true = augment_labels(d['label'], n_mc_smps)
            else:
                y_true = d['label']

            if data_format == 'GP':
                # GP format of data:
                inputs = d['inputs'].to(device)
                indices = d['indices'].to(device)
                values = d['values'].to(device)
                test_inputs = d['test_inputs'].to(device)
                test_indices = d['test_indices'].to(device)
            else:
                raise NotImplementedError('Trainloop for other data formats not implemented yet.')

            execute_callbacks(callbacks, 'on_batch_begin', locals())

            model.train()

            if data_format == 'GP':
                with gpytorch.settings.fast_pred_var(), \
                     gpytorch.settings.max_root_decomposition_size(max_root):
                    logits = model(inputs, indices, values, test_inputs, test_indices)

            y_true = y_true.flatten().to(device)

            loss = loss_fn(logits, y_true)

            loss = loss / virtual_scaling
            loss.backward()
            if (batch + 1) % virtual_scaling == 0:
                optimizer.step()
                optimizer.zero_grad()
                execute_callbacks(callbacks, 'on_batch_end', locals())

        if execute_callbacks(callbacks, 'on_epoch_end', locals()):
            break
    execute_callbacks(callbacks, 'on_train_end', locals())
