#!/usr/bin/env python3
#
# Example script for demonstrating how to use subsampling and loading
# for UEA data sets.

from datasets.uea_datasets import UEADataset
from datasets.subsampling import MissingAtRandomSubsampler
from datasets.subsampling import LabelBasedSubsampler


if __name__ == '__main__':

    # MAR subsampling; no additional parameters except for the threshold
    # have to be specified. Optionally, a random seed can be used.
    dataset_train = UEADataset(
            'PenDigits',
            'training',
            transform=MissingAtRandomSubsampler(0.25))

    print('Missing-at-random subsampling')

    # The subsampling is transparent and automatically applied when
    # requesting each instance.
    print(dataset_train[0])

    # The subsampling is also *consistent* and will always yield the
    # same results.
    print(dataset_train[0])

    # Subsampling with per-class probabilities is slightly more
    # involved. The number of classes has to be specified from the
    # outside, unfortunately, as the reader can only report this
    # information *after* having read the whole data set.

    dataset_train = UEADataset(
            'PenDigits',
            'training',
            transform=LabelBasedSubsampler(10, (0.3, 0.6)))

    print('Label-based subsampling:')

    print(dataset_train[0])
    print(dataset_train[0])
