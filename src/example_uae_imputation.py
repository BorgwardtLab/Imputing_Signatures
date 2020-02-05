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
        LabelBasedSubsampler(10, (0.3, 0.6)),
        ImputationStrategy(strategy='forward_fill')
    ]

    print(transforms)

    dataset_train = UEADataset(
            'PenDigits',
            'training',
            transform=transforms,
            use_disk_cache=True,
    )

    print('Missing-at-random subsampling')

    # Check that subsampling and imputation strategy behave exactly the
    # same...
    print(dataset_train[0])
    print(dataset_train[0])
