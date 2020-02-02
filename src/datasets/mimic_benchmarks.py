from os.path import dirname, exists, join

import numpy as np
from tqdm import trange

import src.tasks as tasks
from src.datasets.dataset import Dataset, DataSplit
from src.datasets.mimic_benchmarks_utils import \
    FeatureTransform, \
    DecompensationReader, \
    InHospitalMortalityReader, \
    LengthOfStayReader, \
    PhenotypingReader, \
    Normalizer

from .utils import DATA_DIR

DATASET_BASE_PATH = join(DATA_DIR, 'mimic_raw')


class MIMICBenchmarkDataSplit(DataSplit):
    def __init__(self, ReaderClass, data_path, listfile, normalizer_config,
                 length_quantiles):
        self.reader = ReaderClass(data_path, listfile)
        self.feature_transform = FeatureTransform()
        self.normalizer = Normalizer(self._get_continious_feature_indices())
        self.length_quantiles = length_quantiles

        self._set_properties()

        if not exists(normalizer_config):
            print(f'Normalizer config {normalizer_config} not found!')
            print('Generating normalizer config...')
            for i in trange(len(self)):
                instance = self.reader.read_example(i)['X']
                transformed = self.feature_transform.transform(instance)
                self.normalizer._feed_data(transformed[1])
            self.normalizer._save_params(normalizer_config)
        else:
            self.normalizer.load_params(normalizer_config)

    def _set_properties(self):
        self.has_unaligned_measurements = True
        self.statics = None
        self.n_statics = 0
        instance = self.reader.read_example(0)
        try:
            self._n_classes = len(instance['y'])
        except TypeError:
            self._n_classes = 1

        times, features = self.feature_transform.transform(instance['X'])
        self._measurement_dims = features.shape[1]

    def _get_continious_feature_indices(self):
        instance = self.reader.read_example(0)
        t, features, header = self.feature_transform.transform(
            instance['X'], return_header=True)
        self._header = header
        cont_channels = [i for (i, h) in enumerate(header) if "->" not in h]
        return cont_channels

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def measurement_dims(self):
        return self._measurement_dims

    def __len__(self):
        return self.reader.get_number_of_examples()

    def __getitem__(self, index):
        instance = self.reader.read_example(index)
        t, features = self.feature_transform.transform(instance['X'])
        features = self.normalizer.transform(features)
        label = instance['y']
        if self.n_classes == 1:
            # Add an additional dimension to the label if it is only a scalar.
            # This makes it confirm more to the treatment of multi-class
            # targets.
            label = [label]
        label = np.array(label, dtype=np.float32)
        time = np.array(t, dtype=np.float32)[:, None]
        features = np.array(features, dtype=np.float32)
        return (time, features), label


class MIMICDatasetBase(Dataset):
    def __init__(self, reader_class, data_path, normalizer_config,
                 length_quantiles):
        self.reader_class = reader_class
        self.data_path = data_path
        self.normalizer_config = normalizer_config
        self.length_quantiles = length_quantiles
        # Build data split on training data to ensure it is used for the
        # compuation of normalization statistics.
        d = self.get_data('training')

    def get_data(self, split):
        if split == 'training':
            return MIMICBenchmarkDataSplit(
                self.reader_class,
                join(self.data_path, 'train'),
                join(self.data_path, 'train_listfile.csv'),
                self.normalizer_config,
                self.length_quantiles
            )
        if split == 'validation':
            return MIMICBenchmarkDataSplit(
                self.reader_class,
                join(self.data_path, 'train'),
                join(self.data_path, 'val_listfile.csv'),
                self.normalizer_config,
                None
            )
        if split == 'testing':
            return MIMICBenchmarkDataSplit(
                self.reader_class,
                join(self.data_path, 'test'),
                join(self.data_path, 'test_listfile.csv'),
                self.normalizer_config,
                None
            )
        raise ValueError(f'Unknown split: {split}')


class MIMICDecompensation(MIMICDatasetBase):
    def __init__(self, data_path=join(DATASET_BASE_PATH, 'decompensation')):
        dataset_name = self.__class__.__name__
        normalizer_config = join(
            dirname(__file__), 'resources',
            dataset_name + '_normalization.json')
        super().__init__(DecompensationReader, data_path, normalizer_config)

    @property
    def task(self):
        return tasks.BinaryClassification()


class MIMICInHospitalMortality(MIMICDatasetBase):
    def __init__(self,
                 data_path=join(DATASET_BASE_PATH, 'in-hospital-mortality')):
        dataset_name = self.__class__.__name__
        normalizer_config = join(
            dirname(__file__), 'resources',
            dataset_name + '_normalization.json')
        super().__init__(
            InHospitalMortalityReader,
            data_path,
            normalizer_config,
            [1, 63, 76, 91, 292]
        )

    @property
    def task(self):
        return tasks.BinaryClassification()


class MIMICPhenotyping(MIMICDatasetBase):
    def __init__(self,
                 data_path=join(DATASET_BASE_PATH, 'phenotyping')):
        dataset_name = self.__class__.__name__
        normalizer_config = join(
            dirname(__file__), 'resources',
            dataset_name + '_normalization.json')
        super().__init__(
            PhenotypingReader,
            data_path,
            normalizer_config,
            [1, 41, 69, 128, 1968]
        )

    @property
    def task(self):
        return tasks.MultilabelClassification(self.n_classes)


class MIMICLengthOfStay(MIMICDatasetBase):
    def __init__(self,
                 data_path=join(DATASET_BASE_PATH, 'length-of-stay')):
        dataset_name = self.__class__.__name__
        normalizer_config = join(
            dirname(__file__), 'resources',
            dataset_name + '_normalization.json')
        super().__init__(LengthOfStayReader, data_path, normalizer_config)

    @property
    def task(self):
        return tasks.Regression(n_dimensions=1, is_positive=True)
