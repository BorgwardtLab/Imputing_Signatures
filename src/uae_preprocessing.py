#!/usr/bin/env python3
#
# Performs subsampling and imputation for all selected UAE data sets and
# all scenarios.

from datasets.uea_datasets import UEADataset
from datasets.subsampling import MissingAtRandomSubsampler
from datasets.subsampling import LabelBasedSubsampler

from imputation import ImputationStrategy


if __name__ == '__main__':

    # Professionally specified classes :)
    datasets = [
        ('CharacterTrajectories', 20),
        ('LSST', 14)
        ('PenDigits', 10)
    ]

    splits = ['training', 'validation', 'testing']

    for dataset_name, _ in datasets:

        for t in [0.5, 0.75]:
            mar = MissingAtRandomSubsampler(t)

        for name in ImputationStrategy().available_strategies:
            transforms = [
                    mar,
                    ImputationStrategy(name)
            ]

            for split in splits:
                dataset = UEADataset(
                            dataset_name,
                            split,
                            transform=transforms,
                            use_disk_cache=True
                )

                for instance, index in enumerate(dataset):
                    print(f'{dataset_name}: {split}: {index}')
