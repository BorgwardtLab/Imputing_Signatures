"""Functions for visualizing stuff."""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_losses(losses, losses_std=None, save_file=None):
    """Plot a dictionary with per epoch losses.

    Args:
        losses: Mean of loss per epoch
        losses_std: stddev of loss per epoch

    """
    for key, values in losses.items():
        if losses_std is not None:
            plt.errorbar(range(len(values)), values, yerr=losses_std[key], label=key)
        else: 
            plt.plot(range(len(values)), values, label=key)
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    if save_file:
        plt.savefig(save_file, dpi=200)
        plt.close()
