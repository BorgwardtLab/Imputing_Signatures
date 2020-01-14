# noqa: ignore: D101,D102
import os

import pandas as pd
import numpy as np
from tqdm import trange

# from ..tasks import BinaryClassification
from .dataset import Dataset
from .mimic_benchmarks_utils import Normalizer
from .utils import DATA_DIR

DATASET_BASE_PATH = os.path.join(DATA_DIR, 'physionet_2012')


class PhysionetDataReader():
    valid_columns = [
        # Statics
        'Age', 'Gender', 'Height', 'ICUType', 'Weight',
        # Time series variables
        'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH'
    ]

    def __init__(self, data_path, endpoint_file):
        self.data_path = data_path
        self.endpoint_data = pd.read_csv(endpoint_file, header=0, sep=',')

    def convert_string_to_decimal_time(self, values):
        return values.str.split(':').apply(
            lambda a: float(a[0]) + float(a[1])/60
        )

    def read_example(self, index):
        example_row = self.endpoint_data.iloc[index, :]
        record_id = example_row['RecordID']
        data = self.read_file(str(record_id))
        data['Time'] = self.convert_string_to_decimal_time(data['Time'])
        return {'X': data, 'y': example_row['In-hospital_death']}

    def read_file(self, record_id):
        filename = os.path.join(self.data_path, record_id + '.txt')
        df = pd.read_csv(filename, sep=',', header=0)
        # Sometimes the same value is observered twice fot the same time, in
        # this case simply take the first occurance.
        duplicated_entries = df[['Time', 'Parameter']].duplicated()
        df = df[~duplicated_entries]
        pivoted = df.pivot(index='Time', columns='Parameter', values='Value')
        return pivoted.reindex(columns=self.valid_columns).reset_index()

    def get_number_of_examples(self):
        return len(self.endpoint_data)


class PhysionetFeatureTransform():
    ignore_columns = ['Time', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']

    def __call__(self, dataframe):
        times = dataframe['Time'].values
        values = dataframe[[
            col for col in dataframe.columns if col not in self.ignore_columns]].values
        return times, values


class Physionet2012Dataset(Dataset):
    """Dataset of the PhysioNet 2012 Computing in Cardiology challenge."""

    normalizer_config = os.path.join(
        os.path.dirname(__file__),
        'resources',
        'Physionet2012Dataset_normalization.json'
    )

    def __init__(self, split, transform=None, data_path=DATASET_BASE_PATH):
        """Initialize dataset.

        Args:
            split: Name of split. One of `training`, `validation`, `testing`.
            data_path: Path to data. Default:
                {project_root_dir}/data/physionet_2012

        """
        self.data_path = data_path
        split_dir, split_file = self._get_split_path(split)

        self.reader = PhysionetDataReader(split_dir, split_file)
        self.feature_transform = PhysionetFeatureTransform()
        self.normalizer = Normalizer()
        self._set_properties()

        if not os.path.exists(self.normalizer_config):
            print(f'Normalizer config {self.normalizer_config} not found!')
            print('Generating normalizer config...')
            if split != 'training':
                # Only allow to compute normalization statics on training split
                raise ValueError(
                    'Not allowed to compute normalization data '
                    'on other splits than training.'
                )
            for i in trange(len(self)):
                instance = self.reader.read_example(i)['X']
                transformed = self.feature_transform(instance)
                self.normalizer._feed_data(transformed[1])
            self.normalizer._save_params(self.normalizer_config)
        else:
            self.normalizer.load_params(self.normalizer_config)

        self.maybe_transform = transform if transform else lambda a: a

    def _get_split_path(self, split):
        split_paths = {
            'training': (
                os.path.join(self.data_path, 'train'),
                os.path.join(self.data_path, 'train_listfile.csv')
            ),
            'validation': (
                os.path.join(self.data_path, 'train'),
                os.path.join(self.data_path, 'val_listfile.csv')
            ),
            'testing': (
                os.path.join(self.data_path, 'test'),
                os.path.join(self.data_path, 'test_listfile.csv')
            )
        }
        return split_paths[split]

    def _set_properties(self):
        self.has_unaligned_measurements = True
        self.statics = None
        self.n_statics = 0
        instance = self.reader.read_example(0)
        times, features = self.feature_transform(instance['X'])
        self._measurement_dims = features.shape[1]

    @property
    def n_classes(self):
        return 1

    @property
    def measurement_dims(self):
        return self._measurement_dims

    def __len__(self):
        return self.reader.get_number_of_examples()

    def __getitem__(self, index):
        instance = self.reader.read_example(index)
        t, features = self.feature_transform(instance['X'])
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
        return self.maybe_transform(
            {'time': time, 'values': features, 'label': label})

    # @property
    # def task(self):
    #     return BinaryClassification()
