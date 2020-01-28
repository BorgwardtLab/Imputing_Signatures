from IPython import embed
import os
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load

import uea_ucr_datasets

#requires: EXPORT UEA_UCR_DATA_DIR=path/to/data/UEA_UCR

uea_ucr_datasets.list_datasets()
dataset = 'PenDigits'  #'PhonemeSpectra' #'SpokenArabicDigits' #'UWaveGestureLibrary'
d = uea_ucr_datasets.Dataset(dataset, train=True)
first_instance = d[0]
instance_x, instance_y = first_instance

embed()

