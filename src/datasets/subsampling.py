'''
Subsampling methods for time series. Implements several ways of
enforcing missingness. Each of these will merely update tensors
to include NaN values---data will *not* be removed.
'''

import numpy as np

import warnings


class MissingAtRandomSubsampler:
    '''
    Performs MAR (missing at random) subsampling using a pre-defined
    threshold.
    '''

    def __init__(self, probability=0.1, random_state=2020):
        self.probability = probability
        self.random_state = np.random.RandomState(random_state)

    def __call__(self, X, y=None):
        '''
        Applies the MAR subsampling to a given instance. The input label
        is optional because it will be ignored.

        Parameters
        ----------
            X: Input tensor or array. The first dimension, i.e. the
            rows, will be taken as the length of the time series.

            y: Input label. Will be ignored and returned as-is. The
            class thus provides a consistent interface.
        '''

        # Full tensor size, i.e. the product of instances and channels
        # of the time series.
        n = np.prod(X.shape)

        # Get the number of samples that we need to mask in the end;
        # we fully ignore interactions between different channels.
        m = int(np.floor(n * self.probability))

        # Nothing to do here, move along!
        if m == 0:
            warnings.warn(
                    f'The current subsampling threshold will *not* result '
                    f'in any instances being subsampled. Consider using a '
                    f'larger probability than {self.probability}.'
            )
        else:

            # Get the indices that we are masking in the original time
            # series and update `X` accordingly.
            indices = self.random_state.choice(n, m, replace=False)
            X.ravel()[indices] = np.nan

        return X, y
