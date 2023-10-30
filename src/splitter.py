
from sklearn.model_selection import  StratifiedShuffleSplit
import pandas as pd

class StratifiedRegressionSplit(StratifiedShuffleSplit):
    def __init__(self, n_splits=5, n_bins = 10, test_size = None, train_size = None, random_state=None):
        super().__init__(n_splits=n_splits, test_size = test_size, train_size=train_size, random_state=random_state)
        self.n_bins = n_bins
    def get_n_splits(self, X=None, y=None, groups=None):
        if type(y) == pd.core.series.Series:
            y_discr = pd.cut(y, bins=self.n_bins)
            return super().get_n_splits(X, y_discr,groups)
        else:
            raise NotImplementedError()
        
    def split(self, X, y=None, groups=None):
        if type(y) == pd.core.series.Series:
            y_discr = pd.cut(y, bins=self.n_bins)
            return super().split(X, y_discr,groups)
        else:
            raise NotImplementedError()
        