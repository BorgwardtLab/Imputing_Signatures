"""Callbacks for training loop."""
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# Hush the linter, child callbacks will always have different parameters than
# the overwritten method of the parent class. Further kwargs will mostly be an
# unused parameter due to the way arguments are passed.
# pylint: disable=W0221,W0613

class Callback():
    """Callback for training loop."""

    def on_epoch_begin(self, **local_variables):
        """Call before an epoch begins."""

    def on_epoch_end(self, **local_variables):
        """Call after an epoch is finished."""

    def on_batch_begin(self, **local_variables):
        """Call before a batch is being processed."""

    def on_batch_end(self, **local_variables):
        """Call after a batch has be processed."""
    
    def on_train_end(self, **local_variables):
        """Call after training is finished."""


class Progressbar(Callback):
    """Callback to show a progressbar of the training progress."""

    def __init__(self):
        """Show a progressbar of the training progress.

        Args:
            print_loss_components: Print all components of the loss in the
                progressbar
        """
        self.total_progress = None
        self.epoch_progress = None

    def on_epoch_begin(self, n_epochs, n_instances, **kwargs):
        """Initialize the progressbar."""
        if self.total_progress is None:
            self.total_progress = tqdm(
                position=0, total=n_epochs, unit='epochs')
        self.epoch_progress = tqdm(
            position=1, total=n_instances, unit='instances')

    def _description(self, loss):
        description = f'Loss: {loss:3.3f}'
        return description

    def on_batch_end(self, batch_size, loss, virtual_batch_size, **kwargs):
        """Increment progressbar and update description."""
        if virtual_batch_size is not None:
            batch_size = virtual_batch_size
        self.epoch_progress.update(batch_size)
        description = self._description(loss)
        self.epoch_progress.set_description(description)

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        """Increment total training progressbar."""
        self.epoch_progress.close()
        self.epoch_progress = None
        self.total_progress.update(1)
        if epoch == n_epochs:
            self.total_progress.close()
