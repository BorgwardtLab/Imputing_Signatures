'''
Subsampling methods for time series. Implements several ways of
enforcing missingness. Each of these will merely update tensors
to include NaN values---data will *not* be removed.
'''

import numpy as np

import warnings


def _mask_tensor(X, p, random_state):
    '''
    Masks values in a tensor with a given probability. The masking will
    set certain values to `np.nan`, making it easy to ignore them for a
    downstream processing task. Given an unravelled tensor of size $m$,
    this function will mask up to $p \cdot m$ of the entries. 

    Parameters
    ----------

        X: Input tensor or `np.ndarray`.
        p: Probability for masking any one entry in the tensor.

        random_state: Random number generator for performing the actual
        sampling prior to the masking. This should be specified from an
        external class instance.
    '''

    # Full tensor size, i.e. the product of instances and channels
    # of the time series.
    n = np.prod(X.shape)

    # Get the number of samples that we need to mask in the end;
    # we fully ignore interactions between different channels.
    m = int(np.floor(n * p))

    # Nothing to do here, move along!
    if m == 0:
        warnings.warn(
                f'The current subsampling threshold will *not* result '
                f'in any instances being subsampled. Consider using a '
                f'larger probability than {p}.'
        )
    else:

        # Get the indices that we are masking in the original time
        # series and update `X` accordingly.
        indices = random_state.choice(n, m, replace=False)
        X.ravel()[indices] = np.nan

    return X


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
            X: Input tensor or `np.ndarray`. 
            y: Input label. Will be ignored and returned as-is in order
            to provide a consistent interface with other schemes.
        '''

        return _mask_tensor(X, self.probability, self.random_state), y


class LabelBasedSubsampler:
    '''
    Performs subsampling conditional on class labels by using
    a pre-defined set of thresholds for each class.
    '''

    def __init__(self, n_classes, probability_ranges, random_state=2020):
        self.random_state = np.random.RandomState(random_state)

        prob_l = probability_ranges[0]
        prob_r = probability_ranges[1]

        # Generate dropout probabilities for each of the classes. This
        # assumes that labels are forming a contiguous range between 0
        # and `n_classes`.
        self.probabilities = self.random_state.uniform(
            prob_l, prob_r, n_classes
        )

    def __call__(self, X, y=None):
        '''
        Applies the label-based subsampling to a given instance.

        Parameters
        ----------
            X: Input tensor or array.
            y: Input label. Will be used to look up a *pre-defined*
            label-based subsampling probability.
        '''

        # Get probability for the particular instance. This call looks
        # idiosyncratic because the UEA reader class wraps labels into
        # an additional dimension.
        p = self.probabilities[y[0]]
        
        return _mask_tensor(X, p, self.random_state), y
