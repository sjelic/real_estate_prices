import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, ensemble, kernel_ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from splitter import StratifiedRegressionSplit
from scoring import scoring
from preprocess  import PolynomialFeaturesDF, KBinsDiscretizerWithNames
np.random.seed(0)
from preprocess import OneHotEncoderOnlyCategorical, PolynomialFeaturesDF, SparseModelFeatureSelector
import os
DATA_DIR = './data'
MODEL_PATH = './models'
RESULT_PATH = './results'


models = {
    'Linear Regression' : {
        'fitting_pipline': GridSearchCV(
            estimator=make_pipeline(
                OneHotEncoderOnlyCategorical(),
                StandardScaler(),
                linear_model.LinearRegression()
            ),
            param_grid={},
            scoring=scoring,
            refit='r2',
            return_train_score = True,
            cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
            n_jobs=-1
        )
    },
    'Ridge Regression' : {
        'fitting_pipline': GridSearchCV(
        estimator=make_pipeline(
                OneHotEncoderOnlyCategorical(),
                StandardScaler(),
                linear_model.Ridge(max_iter=1000)
        ),
        param_grid={'ridge__alpha': np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))) },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Lasso Regression': {
        'preprocessing_pipline': make_pipeline(
        ),
        'fitting_pipline': GridSearchCV(
        estimator=make_pipeline(
                OneHotEncoderOnlyCategorical(),
                StandardScaler(),
                linear_model.Lasso(max_iter=1000)
        ),
        param_grid={'lasso__alpha': np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))) },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'ElasticNet Regression': {
        'fitting_pipline': GridSearchCV(
         estimator=make_pipeline(
                OneHotEncoderOnlyCategorical(),
                StandardScaler(),
                linear_model.ElasticNet(max_iter=1000)
        ),
        param_grid={'elasticnet__alpha': np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
                    'elasticnet__l1_ratio':  np.arange(0,1.1,0.1)},
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Random Forest Regression': {
        'fitting_pipline': GridSearchCV(
         estimator=make_pipeline(
                OneHotEncoderOnlyCategorical(),
                StandardScaler(),
                ensemble.RandomForestRegressor(
                                        bootstrap=True,
                                        max_features= 16,
                                        min_samples_split = 4,
                                        min_samples_leaf=2,
                                        oob_score=True,
                                        n_jobs=-1,
                                        random_state = 0,
                                        n_estimators=500,
                                        max_samples = 0.7
            )
        ),
        param_grid={},
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Polynomial Regression' : {
        'fitting_pipline': GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            PolynomialFeaturesDF(interaction_only=True, include_bias = False),
            StandardScaler(),
            linear_model.LinearRegression()
        ),
        param_grid={
            "ridge__alpha": np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
            "kbinsdiscretizerwithnames__n_bins": np.arange(2, 10),
            "kbinsdiscretizerwithnames__strategy": ['uniform', 'quantile', 'kmeans'],
            "polynomialfeaturesdf__degree" : [2,3]
        },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Ridge Polynomial Regression' : {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
                KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
                OneHotEncoderOnlyCategorical(),
                PolynomialFeaturesDF(interaction_only=True, include_bias = False),
                StandardScaler(),
                linear_model.Ridge(max_iter=1000)
        ),
         param_grid={"ridge__alpha": np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
            "kbinsdiscretizerwithnames__n_bins": np.arange(2, 10),
            "kbinsdiscretizerwithnames__strategy": ['uniform', 'quantile', 'kmeans'],
            "polynomialfeaturesdf__degree" : [2,3]},
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Lasso Polynomial Regression' : 
    {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            PolynomialFeaturesDF(interaction_only=True, include_bias = False),
            StandardScaler(),
            linear_model.Lasso(max_iter=1000)
        ),
        param_grid={"lasso__alpha": np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,10),
                    "kbinsdiscretizerwithnames__strategy": ['uniform', 'quantile', 'kmeans'],
                    "polynomialfeaturesdf__degree" : [2,3]
                    },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'ElasticNet Polynomial Regression' : {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            PolynomialFeaturesDF(interaction_only=True, include_bias = False),
            StandardScaler(),
            linear_model.ElasticNet(max_iter=1000)
        ),
        param_grid={"elasticnet__alpha": np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
                    "elasticnet__l1_ratio": np.arange(0,1.1,0.1),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,10),
                    "kbinsdiscretizerwithnames__strategy": ['uniform', 'quantile', 'kmeans'],
                    "polynomialfeaturesdf__degree" : [2,3]
                    },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Kernel Ridge Regression' : {

        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            StandardScaler(),
            kernel_ridge.KernelRidge(kernel='polynomial')),
        param_grid={'kernelridge__alpha': list(np.arange(0.1,3,0.1)),
                    'kernelridge__degree': list(np.arange(2,5,1)),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,10),
                    "kbinsdiscretizerwithnames__strategy": ['uniform', 'quantile', 'kmeans']
                    },
        scoring=scoring,
        refit='r2',
        return_train_score=True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
        
    }
}