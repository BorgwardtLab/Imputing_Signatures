import json
import os
import pickle

import numpy as np


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'resources/discretizer_config.json'
)


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = None

    def _feed_data(self, x):
        # x = np.array(x)
        if self._count is None:
            # Initialize variables
            self._count = np.zeros(x.shape[-1])
            self._sum_x = np.zeros(x.shape[-1])
            self._sum_sq_x = np.zeros(x.shape[-1])

        self._count += np.sum(np.isfinite(x), axis=0)
        self._sum_x += np.nansum(x, axis=0)
        self._sum_sq_x += np.nansum(x**2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        N = self._count
        self._means = self._sum_x / N
        self._stds = np.sqrt(
            1.0/(N - 1) *
            (
                self._sum_sq_x
                - 2.0 * self._sum_x * self._means
                + N * self._means**2
            )
        )
        self._stds[self._stds < eps] = eps
        with open(save_file_path, "w") as save_file:
            json.dump(
                {
                    'means': self._means.tolist(),
                    'stds': self._stds.tolist()
                },
                save_file
            )

    def load_params(self, load_file_path):
        with open(load_file_path, "r") as load_file:
            dct = json.load(load_file)
            self._means = np.array(dct['means'])
            self._stds = np.array(dct['stds'])

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret


class FeatureTransform:
    """Code based on code from  Harutyunyan et al.

    Converts categorical features into one hot vectors etc.
    Compared to the orignial code this implementation does not contain any
    imputation steps though.
    """
    def __init__(self, start_time='zero', config_path=CONFIG_PATH):
        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel,
                                           range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._start_time = start_time

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None, return_header=False):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = len(X)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.full(shape=(N_bins, cur_len), fill_value=np.nan,
                       dtype=float)
        total_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                data[bin_id, begin_pos[channel_id] + category_id] = 1.
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        times = []
        for bin_id, row in enumerate(X):
            t = float(row[0]) - first_time
            # bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins
            times.append(float(t))

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                total_data += 1
                write(data, bin_id, channel, row[j], begin_pos)

        self._done_count += 1
        times = np.array(times)

        # create new header
        if return_header:
            new_header = []
            for channel in self._id_to_channel:
                if self._is_categorical_channel[channel]:
                    values = self._possible_values[channel]
                    for value in values:
                        new_header.append(channel + "->" + value)
                else:
                    new_header.append(channel)
            return times, data, new_header

        # new_header = ",".join(new_header)

        return times, data


class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class LengthOfStayReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for length of stay prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), float(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.
        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : float
                Remaining time in ICU.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class PhenotypingReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for phenotype classification task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        :param max_length: Discard sequences with more than 1000 hours of data
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(mas[0], float(mas[1]), list(map(int, mas[2:]))) for mas in self._data]
        self._blacklist = [
            # Criterion for exclusion: more than 2000 distinct timepoints
            # Train data
            '50883_episode1_timeseries.csv', '70492_episode1_timeseries.csv',
            '711_episode3_timeseries.csv', '24915_episode1_timeseries.csv',
            '73129_episode2_timeseries.csv', '3932_episode1_timeseries.csv',
            '24597_episode1_timeseries.csv', '31123_episode1_timeseries.csv',
            '99383_episode1_timeseries.csv', '16338_episode1_timeseries.csv',
            '48123_episode2_timeseries.csv', '1785_episode1_timeseries.csv',
            '56854_episode1_timeseries.csv', '76151_episode2_timeseries.csv',
            '72908_episode1_timeseries.csv', '26277_episode3_timeseries.csv',
            '77614_episode1_timeseries.csv', '6317_episode3_timeseries.csv',
            '82609_episode1_timeseries.csv', '79645_episode1_timeseries.csv',
            '12613_episode1_timeseries.csv', '77617_episode1_timeseries.csv',
            '41861_episode1_timeseries.csv', '55205_episode1_timeseries.csv',
            '45910_episode1_timeseries.csv', '80927_episode1_timeseries.csv',
            '49555_episode1_timeseries.csv', '19911_episode3_timeseries.csv',
            '43459_episode1_timeseries.csv', '21280_episode2_timeseries.csv',
            '90776_episode1_timeseries.csv', '51078_episode2_timeseries.csv',
            '65565_episode1_timeseries.csv', '41493_episode1_timeseries.csv',
            '10694_episode2_timeseries.csv', '54073_episode1_timeseries.csv',
            '12831_episode2_timeseries.csv', '89223_episode1_timeseries.csv',
            '46156_episode1_timeseries.csv', '58242_episode4_timeseries.csv',
            '55639_episode1_timeseries.csv', '89840_episode1_timeseries.csv',
            # Validation data
            '67906_episode1_timeseries.csv', '59268_episode1_timeseries.csv',
            '78251_episode1_timeseries.csv', '32476_episode1_timeseries.csv',
            '96924_episode2_timeseries.csv', '96686_episode10_timeseries.csv',
            '5183_episode1_timeseries.csv', '58723_episode1_timeseries.csv',
            '78515_episode1_timeseries.csv', '40187_episode1_timeseries.csv',
            '62239_episode2_timeseries.csv', '79337_episode1_timeseries.csv',
            # Testing data
            '29105_episode2_timeseries.csv', '69745_episode4_timeseries.csv',
            '59726_episode1_timeseries.csv', '81786_episode1_timeseries.csv',
            '12805_episode1_timeseries.csv', '6145_episode1_timeseries.csv',
            '54353_episode2_timeseries.csv', '58854_episode1_timeseries.csv',
            '98994_episode1_timeseries.csv', '19223_episode2_timeseries.csv',
            '80345_episode1_timeseries.csv', '48935_episode1_timeseries.csv',
            '48380_episode1_timeseries.csv', '70698_episode1_timeseries.csv',
            '51177_episode1_timeseries.csv'
        ]
        self._data = list(filter(
            lambda a: a[0] not in self._blacklist, self._data))

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : array of ints
                Phenotype labels.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class DecompensationReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for decompensation prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Read the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Directory with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                Mortality within next 24 hours.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._blacklist = [
            # Criterion for exclusion: more than 1000 distinct timepoints
            # In training data
            '73129_episode2_timeseries.csv', '48123_episode2_timeseries.csv',
            '76151_episode2_timeseries.csv', '41493_episode1_timeseries.csv',
            '65565_episode1_timeseries.csv', '55205_episode1_timeseries.csv',
            '41861_episode1_timeseries.csv', '58242_episode4_timeseries.csv',
            '54073_episode1_timeseries.csv', '46156_episode1_timeseries.csv',
            '55639_episode1_timeseries.csv', '89840_episode1_timeseries.csv',
            '43459_episode1_timeseries.csv', '10694_episode2_timeseries.csv',
            '51078_episode2_timeseries.csv', '90776_episode1_timeseries.csv',
            '89223_episode1_timeseries.csv', '12831_episode2_timeseries.csv',
            '80536_episode1_timeseries.csv',
            # In validation data
            '78515_episode1_timeseries.csv', '62239_episode2_timeseries.csv',
            '58723_episode1_timeseries.csv', '40187_episode1_timeseries.csv',
            '79337_episode1_timeseries.csv',
            # In testing data
            '51177_episode1_timeseries.csv', '70698_episode1_timeseries.csv',
            '48935_episode1_timeseries.csv', '54353_episode2_timeseries.csv',
            '19223_episode2_timeseries.csv', '58854_episode1_timeseries.csv',
            '80345_episode1_timeseries.csv', '48380_episode1_timeseries.csv'
        ]
        self._data = list(filter(
            lambda a: a[0] not in self._blacklist, self._data))
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}
