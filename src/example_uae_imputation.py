#!/usr/bin/env python3
#
# Example script for demonstrating how to use imputation and subsampling
# for working with UEA data sets.

from datasets.uea_datasets import UEADataset
from datasets.subsampling import MissingAtRandomSubsampler
from datasets.subsampling import LabelBasedSubsampler

from imputation import ImputationStrategy


if __name__ == '__main__':

    transforms = [
        MissingAtRandomSubsampler(0.25),
        ImputationStrategy(strategy='zero')
    ]

    dataset_train = UEADataset(
            'PenDigits',
            'training',
            transform=transforms
    )

    print('Missing-at-random subsampling')

    # Check that subsampling and imputation strategy behave exactly the
    # same...
    print(dataset_train[0])
    print(dataset_train[0])
