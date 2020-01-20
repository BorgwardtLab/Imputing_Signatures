"""Module to train a model with a dataset configuration."""
import numpy as np
import os
import pandas as pd
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import torch

import sys
sys.path.append(os.getcwd())

from src.callbacks import Callback, Progressbar

from src.datasets import get_input_transform, get_collate_fn
from src.training import TrainingLoop
from src.visualization import plot_losses

from exp.callbacks import LogDatasetLoss, LogTrainingLoss
from exp.ingredients import model as model_config
from exp.ingredients import dataset as dataset_config



EXP = Experiment(
    'training',
    ingredients=[model_config.ingredient, dataset_config.ingredient]
)
EXP.captured_out_filter = apply_backspaces_and_linefeeds

@EXP.config
def cfg():
    n_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-5
    early_stopping = 10
    data_format = 'GP'
    grid_spacing = 1. #determines n_hours between query points
    max_root = 25 #max_root_decomposition_size for MGP matrix rank
    device = 'cuda'
    quiet = False
    evaluation = {
        'active': False,
        'evaluate_on': 'validation'
    }


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
def train(n_epochs, batch_size, learning_rate, weight_decay,
          early_stopping, data_format, grid_spacing, max_root, device, quiet, 
            evaluation, _run, _log, _seed, _rnd):
    """Sacred wrapped function to run training of model."""
    torch.manual_seed(_seed)
    rundir = None
    try:
        rundir = _run.observers[0].dir
    except IndexError:
        pass

    #Define dataset transform:
    input_transform = get_input_transform(data_format, grid_spacing)

    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    train_dataset = dataset_config.get_instance(split='training', 
                                                transform=input_transform)
    validation_dataset = dataset_config.get_instance(split='validation', 
                                                    transform=input_transform)
    test_dataset = dataset_config.get_instance(split='testing', 
                                                transform=input_transform)
    
    #Determine number of input dimensions as GP-Sig models requires this parameter for initialisation
    n_input_dims = train_dataset.measurement_dims
    collate_fn = get_collate_fn(data_format, n_input_dims) 
    
    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120
    model = model_config.get_instance(n_input_dims)
    model.to(device)
    
    # Loss function:
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    callbacks = [
        LogTrainingLoss(_run, print_progress=quiet),
        LogDatasetLoss('validation', validation_dataset, data_format, collate_fn,
                        loss_fn, _run, batch_size, max_root, n_mc_smps, 
                        early_stopping=early_stopping, save_path=rundir, 
                        device=device, print_progress=True),
        LogDatasetLoss('testing', test_dataset, data_format, collate_fn,
                        loss_fn, _run, batch_size, max_root, n_mc_smps, 
                        early_stopping=early_stopping, save_path=rundir, 
                        device=device, print_progress=False)
    ]

    if quiet:
        # Add newlines between epochs
        callbacks.append(NewlineCallback())
    else:
        callbacks.append(Progressbar(print_loss_components=True))

    training_loop = TrainingLoop(
        model, 
        dataset,
        data_format, 
        loss_fn,
        collate_fn,
        n_epochs, 
        batch_size, 
        learning_rate, 
        n_mc_smps, 
        max_root,
        weight_decay,
        device, 
        callbacks
    )
    # Run training
    training_loop()

    if rundir:
        # Save model state (and entire model)
        print('Loading model checkpoint prior to evaluation...')
        state_dict = torch.load(os.path.join(rundir, 'model_state.pth'))
        model.load_state_dict(state_dict)
    model.eval()

    logged_averages = callbacks[0].logged_averages
    logged_stds = callbacks[0].logged_stds
    loss_averages = {
        key: value for key, value in logged_averages.items() if 'loss' in key
    }
    loss_stds = {
        key: value for key, value in logged_stds.items() if 'loss' in key
    }
    if rundir:
        plot_losses(
            loss_averages,
            loss_stds,
            save_file=os.path.join(rundir, 'loss.png')
        )

    result = {
        key: values[-1] for key, values in logged_averages.items()
    }

    if evaluation['active']:
        evaluate_on = evaluation['evaluate_on']
        if evaluate_on == 'validation':
            eval_measures = callbacks[1].eval_measures
        else:
            eval_measures = callbacks[1].eval_measures
        
        result.update(eval_measures)

    return result
