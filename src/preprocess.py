import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, KBinsDiscretizer
import numpy as np
import pickle
import os

class OneHotEncoderOnlyCategorical(OneHotEncoder):
    def __init__(self, categories = 'auto', drop = None, sparse_output = False, dtype = float, handle_unknown = 'error'):
        super().__init__(categories=categories, drop = drop, sparse_output=sparse_output, dtype=dtype, handle_unknown=handle_unknown)
        self.categorical_features = {}
        self.numerical_features = {}
        self.num_cols = []
        self.cat_cols = []
        self.feature_names_in_ = []
        self.n_features_in_ = 0
    def fit(self, X, y=None):        
        self.categorical_features = { n: t for n, t in X.dtypes.to_dict().items() if t in [ np.dtype('O') ]}
        self.numerical_features = { n: t for n, t in X.dtypes.to_dict().items() if t not in [ np.dtype('O') ]}
        self.num_cols = list(self.numerical_features.keys())
        self.cat_cols = list(self.categorical_features.keys())
        super().fit(X[self.cat_cols])
        self.feature_names_in_ = np.array( self.cat_cols, dtype = object)
        self.n_features_in_ = len( self.cat_cols)
        return self
        
    def transform(self, X, y=None):
        return pd.DataFrame(data = 
                            np.hstack(
                                (X[self.num_cols].values, super().transform(X[self.cat_cols]))
                                ), 
                            columns=self.get_feature_names_out()
                        )
           
    def get_feature_names_out(self, input_features = None):
        return np.array( self.num_cols +  list(super().get_feature_names_out() ),  dtype=object)

class PolynomialFeaturesDF(PolynomialFeatures):
    def __init__(self, degree = 2, interaction_only = False, include_bias = True, order = 'C'):
        super().__init__(degree = degree, interaction_only = interaction_only, include_bias = include_bias, order = order)
        self.feature_names_in_ = []
        self.n_features_in_ = 0
    def fit(self, X, y=None):
        super().fit(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self
        
    def transform(self, X, y=None):
        super().fit(X)
        X_t = pd.DataFrame(columns=list(super().get_feature_names_out() ))
        X_t[list(super().get_feature_names_out() )] = super().transform(X)
        return X_t
           
    def get_feature_names_out(self, input_features = None):
        return super().get_feature_names_out()
    
 
class  CreateInteractions(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.feature_names_in_ = []
        self.n_features_in_ = 0
        
        self.n_cat_f = 0
        self.n_num_f = 0
        self.n_bin_num_f = 0
        self.n_nonbin_num_f = 0
        
        self.categorical_features = {}
        self.numerical_features = {}
        self.binary_numerical_features = {}
        self.nonbinary_numerical_features = {}
        
        self.cat_cols = []
        self.num_cols = []
        self.bin_num_cols = []
        self.nonbin_num_cols = []
    def fit(self, X, y=None):
        self.categorical_features = { n: t for n, t in X.dtypes.to_dict().items() if t in [ np.dtype('O') ]}
        self.numerical_features = { n: t for n, t in X.dtypes.to_dict().items() if t not in [ np.dtype('O') ]}
 
        self.cat_cols = list(self.categorical_features.keys())
        self.num_cols = list(self.numerical_features.keys())
        self.n_cat_f = len(self.cat_cols)
        self.n_num_f = len(self.num_cols)
        
        for num_col in self.num_cols:
            is_bin = True
            for val in X[num_col]:
                if (np.abs(1-val) > 1e-3) and (np.abs(val) > 1e-3):
                    is_bin = False
            if is_bin:
                self.bin_num_cols.append(num_col)
            else:
                self.nonbin_num_cols.append(num_col)
        
        self.n_bin_num_f = len(self.bin_num_cols)
        self.n_nonbin_num_f = len(self.nonbin_num_cols)
        
        self.binary_numerical_features = { n: t for n, t in X.dtypes.to_dict().items() if n in self.bin_num_cols}
        self.nonbinary_numerical_features = { n: t for n, t in X.dtypes.to_dict().items() if n in self.nonbin_num_cols}
        
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self
        
    def transform(self, X, y=None):
        X_t = pd.DataFrame(data=np.zeros((len(X),self.n_cat_f + self.n_num_f + self.n_nonbin_num_f * self.n_bin_num_f + int((self.n_bin_num_f*(self.n_bin_num_f - 1))/2))), columns=self.get_feature_names_out())

        X_t[self.cat_cols] = X[self.cat_cols]
        X_t[self.num_cols] = X[self.num_cols]
        for num_col in self.nonbin_num_cols:
            for cat_col in self.bin_num_cols:
                X_t[f"{num_col} {cat_col}"] = X[num_col]*X[cat_col]

        for i in range(len(self.bin_num_cols)):
            for j in range(i+1,len(self.bin_num_cols)):
                X_t[f"{self.bin_num_cols[i]} {self.bin_num_cols[j]}"] = X[self.bin_num_cols[i]]*X[self.bin_num_cols[j]]
        
        return X_t
           
    def get_feature_names_out(self, input_features = None):
        # num with categorical
        # categorical with categorical
        cols = []
        for num_col in self.nonbin_num_cols:
            for cat_col in self.bin_num_cols:
                cols.append(f"{num_col} {cat_col}")
        for i in range(len(self.bin_num_cols)):
            for j in range(i+1,len(self.bin_num_cols)):
                cols.append(f"{self.bin_num_cols[i]} {self.bin_num_cols[j]}")
        return np.array(self.cat_cols + self.num_cols + cols, dtype=object)
    
class TypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, variables = []):
        self.variables = variables
        self.convert_dict = []
        self.feature_names_in_ = []
        self.n_features_in_ = 0
    def fit(self, X, y=None):
        self.convert_dict = {variable['name'] : variable['type'] for variable in self.variables}
        self.feature_names_in_ = np.array( X.columns, dtype = object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self
    def transform(self, X, y=None):
        return X.astype(self.convert_dict)[ [variable['name'] for variable in self.variables] ]
    
    def get_feature_names_out(self, input_features = None):
        return np.array([variable['name'] for variable in self.variables], dtype = object)
        
class SparseModelFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model_file = None):
        self.gs = None
        if os.path.isfile(model_file):
            with open(model_file, 'rb') as f:
                self.gs = pickle.load(f)
        # else:
        #     raise FileNotFoundError(f'File {model_file} does not exist.')
        self.selected_features = []
        self.feature_names_in_ = []
        self.n_features_in_ = 0
        
    def fit(self, X, y=None):
        features = list(self.gs.feature_names_in_)
        coefs = list(self.gs.best_estimator_.steps[1][1].coef_)
        self.feature_names_in_ = np.array( X.columns, dtype = object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.selected_features = [feature for i, feature in enumerate(features) if (np.abs(coefs[i]) > 1e-3)]
    def transform(self, X, y=None):
        return X[self.selected_features]
    def get_feature_names_out(self, input_features = None):
        return np.array(self.selected_features, dtype = object)

class KBinsDiscretizerWithNames(KBinsDiscretizer):
    def __init__(self, n_bins = 10,  encode='onehot', strategy='uniform', random_state=0, dtype=np.float32):
        super().__init__(n_bins=n_bins, encode = encode, strategy=strategy, random_state=random_state, subsample=None)
        self.categorical_features = {}
        self.numerical_features = {}
        self.num_cols = []
        self.cat_cols = []
        self.feature_names_in_ = []
        self.n_features_in_ = 0
        #self.bin_edges_ = []

        # self.bin_edges_ = None
    def fit(self, X, y=None):
        
        # get categorical features by type
        self.categorical_features = { n: t for n, t in X.dtypes.to_dict().items() if t in [ np.dtype('O') ]}
        # get numerical features by type
        self.numerical_features = { n: t for n, t in X.dtypes.to_dict().items() if t not in [ np.dtype('O') ]}
        
        self.num_cols = list(self.numerical_features.keys())
        self.cat_cols = list(self.categorical_features.keys())
        
        # if set(self.num_feature_names).issubset(set(self.num_cols)):
        #     raise ValueError(f"Feature names {list(set(self.num_feature_names).difference(set(self.num_cols)))} are not present in column list of data matrix: {self.num_cols}")
        self.feature_names_in_ = np.array( X.columns, dtype = object)
        self.n_features_in_ = len( self.feature_names_in_)
        super().fit(X[self.num_cols])
        return self
        
    def transform(self, X, y=None):
        type_dict = {}
        type_dict.update(self.categorical_features)
        type_dict.update(self.numerical_features)
        type_dict.update(
            {f: np.dtype('float32') for f in self.get_dicr_names_out()}
        )
        disc_tt = super().transform(X[self.num_cols])
        return pd.DataFrame(data = 
                            np.hstack(
                                (X[self.num_cols + self.cat_cols].values, disc_tt.toarray() if type(disc_tt) != np.ndarray else disc_tt )
                                ), 
                            columns=self.get_feature_names_out()
                        ).astype(
                            type_dict
                        )
    def get_dicr_names_out(self, input_features = None):
        dis_out_names=[]
        for i in range(len(self.num_cols)):
            dis_out_names += list(map(
                lambda x: 
                    f"{self.num_cols[i]}_[{x[0]}, {x[1]})",
                zip(self.bin_edges_[i], self.bin_edges_[i][1:])
            ))
        return dis_out_names
    
    def get_feature_names_out(self, input_features = None):
        
        return np.array(self.num_cols + self.cat_cols + self.get_dicr_names_out(),  dtype=object)



class KBinsDiscretizerWithNamesOnlyCategorical(KBinsDiscretizer):
    def __init__(self, n_bins = 10,  encode='onehot', strategy='uniform', random_state=0, dtype=np.float32):
        super().__init__(n_bins=n_bins, encode = encode, strategy=strategy, random_state=random_state, subsample=None)
        self.categorical_features = {}
        self.numerical_features = {}
        self.num_cols = []
        self.cat_cols = []
        self.feature_names_in_ = []
        self.n_features_in_ = 0
        #self.bin_edges_ = []

        # self.bin_edges_ = None
    def fit(self, X, y=None):
        
        # get categorical features by type
        self.categorical_features = { n: t for n, t in X.dtypes.to_dict().items() if t in [ np.dtype('O') ]}
        # get numerical features by type
        self.numerical_features = { n: t for n, t in X.dtypes.to_dict().items() if t not in [ np.dtype('O') ]}
        
        self.num_cols = list(self.numerical_features.keys())
        self.cat_cols = list(self.categorical_features.keys())
        
        # if set(self.num_feature_names).issubset(set(self.num_cols)):
        #     raise ValueError(f"Feature names {list(set(self.num_feature_names).difference(set(self.num_cols)))} are not present in column list of data matrix: {self.num_cols}")
        self.feature_names_in_ = np.array( X.columns, dtype = object)
        self.n_features_in_ = len( self.feature_names_in_)
        super().fit(X[self.num_cols])
        return self
        
    def transform(self, X, y=None):
        type_dict = {}
        type_dict.update(self.categorical_features)
        type_dict.update(
            {f: np.dtype('float32') for f in self.get_dicr_names_out()}
        )
        disc_tt = super().transform(X[self.num_cols])
        return pd.DataFrame(data = 
                            np.hstack(
                                (X[self.cat_cols].values, disc_tt.toarray() if type(disc_tt) != np.ndarray else disc_tt )
                                ), 
                            columns=self.get_feature_names_out()
                        ).astype(
                            type_dict
                        )
    def get_dicr_names_out(self, input_features = None):
        dis_out_names=[]
        for i in range(len(self.num_cols)):
            dis_out_names += list(map(
                lambda x: 
                    f"{self.num_cols[i]}_[{x[0]}, {x[1]})",
                zip(self.bin_edges_[i], self.bin_edges_[i][1:])
            ))
        return dis_out_names
    
    def get_feature_names_out(self, input_features = None):
        
        return np.array(self.cat_cols + self.get_dicr_names_out(),  dtype=object)