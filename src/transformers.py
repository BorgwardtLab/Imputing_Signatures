from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sig_utils import compute_signatures
from tslearn.metrics import cdist_dtw
from sklearn.neighbors import KNeighborsClassifier



class SignatureTransform(TransformerMixin, BaseEstimator):
    def __init__(self, truncation=5):
        self.truncation = truncation
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Computes signatures of entire data array (n samples, d time steps)
        Currently implemented for univariate time series data
        Returns Signature array (n_samples, n_sig_components)
        """
        return compute_signatures(X, trunc=self.truncation)


class DTW_KNN(BaseEstimator, ClassifierMixin):
    """ DTW_KNN model in sklearn style (for pipeline transforms)
    """
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
    def fit(self, X, y=None):
        #add train data to object for later predictions:
        self.X_train = X
        #Compute DTW distance matrix of train data:
        D = cdist_dtw(X)
        #Fit KNN on train DTW matrix
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='precomputed', n_jobs=4)
        self.knn.fit(D, y)
    def predict(self, X, y=None):
        D_test = cdist_dtw(X, self.X_train)
        return self.knn.predict(D_test)

 
        
