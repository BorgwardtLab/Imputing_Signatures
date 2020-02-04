#!/usr/bin/env python3
#
# Test script for checking whether UEA data sets can be loaded and
# transformed accordingly.

from datasets.uea_datasets import UEADataset


if __name__ == '__main__':
    dataset_train = UEADataset(
            'PenDigits',
            'training')

    for X in dataset_train:
        print(X)
