"""Training classes."""
import gc
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.utils.train_utils import augment_labels


class TrainingLoop():
    """Training a model using a dataset."""

    def __init__(self, model, dataset, data_format, loss_fn, collate_fn, n_epochs, batch_size, learning_rate,
                 n_mc_smps=1, max_root=25, weight_decay=1e-5, device='cuda', callbacks=None):
        """Training of a model using a dataset and the defined callbacks.

        Args:
            model: GP_Sig, competitor
            dataset: Dataset
            n_epochs: Number of epochs to train
            batch_size: Batch size
            learning_rate: Learning rate
            callbacks: List of callbacks
        """
        self.model = model
        self.dataset = dataset
        self.data_format = data_format
        self.loss_fn = loss_fn
        self.collate_fn = collate_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_mc_smps = n_mc_smps
        self.max_root = max_root
        self.weight_decay = weight_decay
        self.device = device
        self.callbacks = callbacks if callbacks else []

    def _execute_callbacks(self, hook, local_variables):
        stop = False
        for callback in self.callbacks:
            # Convert return value to bool --> if callback doesn't return
            # anything we interpret it as False
            stop |= bool(getattr(callback, hook)(**local_variables))
        return stop

    def on_epoch_begin(self, local_variables):
        """Call callbacks before an epoch begins."""
        return self._execute_callbacks('on_epoch_begin', local_variables)

    def on_epoch_end(self, local_variables):
        """Call callbacks after an epoch is finished."""
        return self._execute_callbacks('on_epoch_end', local_variables)

    def on_batch_begin(self, local_variables):
        """Call callbacks before a batch is being processed."""
        self._execute_callbacks('on_batch_begin', local_variables)

    def on_batch_end(self, local_variables):
        """Call callbacks after a batch has be processed."""
        self._execute_callbacks('on_batch_end', local_variables)

    def on_train_end(self, local_variables):
        """Call callbacks after training is finished."""
        self._execute_callbacks('on_train_end', local_variables)

    # pylint: disable=W0641
    def __call__(self):
        """Execute the training loop."""
        model = self.model
        dataset = self.dataset
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        n_mc_smps = self.n_mc_smps
        collate_fn = self.collate_fn

        n_instances = len(dataset)
        train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, 
                                  shuffle=True, pin_memory=True) #drop_last=True)
        n_batches = len(train_loader)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=self.weight_decay)

        epoch = 1
        for epoch in range(1, n_epochs+1):
            if self.on_epoch_begin(remove_self(locals())):
                break

            for batch, d in enumerate(train_loader):
                #if we use mc sampling, expand labels to match multiple predictions
                if n_mc_smps > 1:
                    y_true = augment_labels(d['label'], n_mc_smps)
                else:
                    y_true = d['label']

                if self.data_format == 'GP':
                    #GP format of data:
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
                else:
                    raise NotImplementedError('Trainloop for other data formats not implemented yet.')
                 
                self.on_batch_begin(remove_self(locals()))

                # Set model into training mode and feed forward
                model.train()
                
                if self.data_format == 'GP':
                    with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(self.max_root):
                        logits = model( inputs, 
                                        indices, 
                                        values, 
                                        test_inputs, 
                                        test_indices)

                #Compute Loss:
                if self.device == 'cuda':
                    y_true = y_true.long().flatten().cuda(non_blocking=True) 
                else: 
                    y_true = y_true.long().flatten()
                loss = self.loss_fn(logits, y_true)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Call callbacks
                self.on_batch_end(remove_self(locals()))
                
                # Clean memory
                torch.cuda.empty_cache()
                gc.collect()

            if self.on_epoch_end(remove_self(locals())):
                break
        self.on_train_end(remove_self(locals()))
        return epoch


def remove_self(dictionary):
    """Remove entry with name 'self' from dictionary.

    This is useful when passing a dictionary created with locals() as kwargs.

    Args:
        dictionary: Dictionary containing 'self' key

    Returns:
        dictionary without 'self' key

    """
    del dictionary['self']
    return dictionary

