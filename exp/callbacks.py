"""Callbacks specific to sacred."""
import os
from collections import defaultdict

import gpytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as auprc

from src.callbacks import Callback
from src.utils.train_utils import augment_labels


def convert_to_base_type(value):
    """Convert a value into a python base datatype.

    Args:
        value: numpy or torch value

    Returns:
        Python base type
    """
    if isinstance(value, (torch.Tensor, np.generic)):
        return value.item()
    else:
        return value


class LogTrainingLoss(Callback):
    """Logging of loss during training into sacred run."""

    def __init__(self, run, print_progress=False):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            run: Sacred run
        """
        self.run = run
        self.print_progress = print_progress
        self.epoch_losses = None
        self.logged_averages = defaultdict(list)
        self.logged_stds = defaultdict(list)
        self.iterations = 0

    def _description(self):
        all_keys = self.logged_averages.keys()
        elements = []
        for key in all_keys:
            last_average = self.logged_averages[key][-1]
            last_std = self.logged_stds[key][-1]
            elements.append(
                f'{key}: {last_average:3.3f} +/- {last_std:3.3f}')
        return ' '.join(elements)

    def on_epoch_begin(self, **kwargs):
        self.epoch_losses = defaultdict(list)

    def on_batch_end(self, loss, **kwargs):
        loss = convert_to_base_type(loss)
        self.iterations += 1
        self.epoch_losses['training.loss'].append(loss)
        self.run.log_scalar('training.loss.batch', loss, self.iterations)

    def on_epoch_end(self, epoch, **kwargs):
        for key, values in self.epoch_losses.items():
            mean = np.mean(values)
            std = np.std(values)
            self.run.log_scalar(key + '.mean', mean, self.iterations)
            self.logged_averages[key].append(mean)
            self.run.log_scalar(key + '.std', std, self.iterations)
            self.logged_stds[key].append(std)
        self.epoch_losses = defaultdict(list)
        if self.print_progress:
            print(f'Epoch {epoch}:', self._description())


class LogDatasetLoss(Callback):
    """Logging of loss and other eval measures during and after training into sacred run."""

    def __init__(self, dataset_name, dataset, data_format, collate_fn, loss_fn, run, 
                 batch_size=64, max_root=25, n_mc_smps=1, early_stopping=None, save_path=None,
                 device='cpu', print_progress=True):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            dataset_name: Name of dataset
            dataset: Dataset to use
            collate_fn: dataset-specific collate function (depends on input dimension)
            loss_fn: loss function object to use
            run: Sacred run
            print_progress: Print evaluated loss
            batch_size: Batch size
            max_root: max_root_decomposition_size (rank param for GP)
            n_mc_smps: number of mc samples (1 if no additional sampling used)
            early_stopping: if int the number of epochs to wait befor stopping
                training due to non-decreasing loss, if None dont use
                early_stopping
            save_path: Where to store model weigths
        """
        self.prefix = dataset_name
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size,
                                      collate_fn=collate_fn, pin_memory=True)
        self.data_format = data_format
        self.loss_fn = loss_fn
        self.run = run
        self.print_progress = print_progress
        self.max_root = max_root
        self.early_stopping = early_stopping
        self.save_path = save_path
        self.device = device
        self.iterations = 0
        self.patience = 0
        self.best_loss = np.inf

    def _compute_eval_measures(self, model, full_eval=False):
        losses = defaultdict(list)
        model.eval()

        if full_eval:
            y_true_total = []
            y_score_total = []

        for d in self.data_loader:
            
            #Augment labels depending on whether mc sampling is used
            if self.n_mc_smps > 1:
                    y_true = augment_labels(d['label'], self.n_mc_smps)
            else:
                    y_true = d['label']
            if data_format == 'GP':
                #Unpack GP format data: 
                inputs = d['inputs']
                indices = d['indices'] 
                values = d['values']
                test_inputs = d['test_inputs']
                test_indices = d['test_indices'] 
                
                if self.device == 'cuda':
                    inputs  = inputs.cuda(non_blocking = True)
                    indices = indices.cuda(non_blocking = True)
                    values  = values.cuda(non_blocking = True)
                    test_inputs = test_inputs.cuda(non_blocking = True)
                    test_indices = test_indices.cuda(non_blocking = True) 

                with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(
                    self.max_root):
    
                    logits = model( inputs, 
                                    indices, 
                                    values, 
                                    test_inputs, 
                                    test_indices )
            else: 
                raise NotImplementedError('Data formats other than GP not implemented yet!')

            #Compute loss
            if self.device == 'cuda':
                y_true = y_true.long().flatten().cuda( non_blocking=True )
            else: 
                y_true = y_true.long().flatten()
            
            loss = loss_fn(logits, y_true)

            loss = convert_to_base_type(loss)

            # Rescale the losses as batch_size might not divide dataset
            # perfectly, e.g. in case drop_last is set to True in
            # the constructor.
            n_instances = len(data)
            losses['loss'].append(loss*n_instances)
            
            if full_eval:
                with torch.no_grad():
                    y_true = y_true.detach().cpu().numpy()
                    y_score = logits[:,1].flatten().detach().cpu().numpy()  
                    y_true_total.append(y_true)
                    y_score_total.append(y_score)
        return_dict = {}
        average_loss = sum(losses['loss']) / len(self.dataset)
        return_dict['average_loss'] = average_loss  
        if full_eval: 
            y_true_total = np.concatenate(y_true_total)
            y_score_total = np.concatenate(y_score_total)
            for measures in [auc, auprc]:
                for mode in ['macro', 'micro', 'weighted']:
                    return_dict[measure.__name__ + '__' + mode] = measure(y_true_total, y_score_total, average=mode)
        return return_dict
 
    def _progress_string(self, epoch, losses):
        progress_str = " ".join([
            f'{self.prefix}.{key}: {value:.3f}'
            for key, value in losses.items()
        ])
        return f'Epoch {epoch}: ' + progress_str

    def on_batch_end(self, **kwargs):
        self.iterations += 1

    def on_epoch_begin(self, model, epoch, **kwargs):
        """Store the loss on the dataset prior to training."""
        if epoch == 1:  # This should be prior to the first training step
            losses = self._compute_eval_measures(model)
            if self.print_progress:
                print(self._progress_string(epoch - 1, losses))

            for key, value in losses.items():
                self.run.log_scalar(
                    f'{self.prefix}.{key}',
                    value,
                    self.iterations
                )

    def on_epoch_end(self, model, epoch, **kwargs):
        """Score evaluation metrics at end of epoch."""
        losses = self._compute_eval_measures(model)
        print(self._progress_string(epoch, losses))
        for key, value in losses.items():
            self.run.log_scalar(
                f'{self.prefix}.{key}',
                value,
                self.iterations
            )
        if self.early_stopping is not None:
            if losses['loss'] < self.best_loss:
                self.best_loss = losses['loss']
                if self.save_path is not None:
                    save_path = os.path.join(self.save_path, 'model_state.pth')
                    print('Saving model to', save_path)
                    torch.save(
                        model.state_dict(),
                        save_path
                    )
                self.patience = 0
            else:
                self.patience += 1

            if self.early_stopping <= self.patience:
                print(
                    'Stopping training due to non-decreasing '
                    f'{self.prefix} loss over {self.early_stopping} epochs'
                )
                return True

    def on_train_end(self, model, epoch, **kwargs):
        """Score evaluation metrics at end of training."""
        self.eval_measures = self._compute_eval_measures(model, full_eval=True)
        for key, value in self.eval_measures.items():
            self.run.log_scalar(
                f'{self.prefix}.{key}',
                value,
                self.iterations
            )

