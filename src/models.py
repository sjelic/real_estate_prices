import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, ensemble, kernel_ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from splitter import StratifiedRegressionSplit
from scoring import scoring
from preprocess  import PolynomialFeaturesDF, KBinsDiscretizerWithNames, CreateInteractions, KBinsDiscretizerWithNamesOnlyCategorical
np.random.seed(0)
from preprocess import OneHotEncoderOnlyCategorical, PolynomialFeaturesDF, SparseModelFeatureSelector
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lars
from sklearn.decomposition import PCA

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
                PCA(),
                RandomForestRegressor(
                                        bootstrap=True,
                                        max_features= 16,
                                        min_samples_split = 4,
                                        min_samples_leaf=2,
                                        oob_score=True,
                                        n_jobs=-1,
                                        random_state = 0,
                                        n_estimators=40,
                                        max_samples = 0.7
            )
        ),
        param_grid={'pca__n_components': np.arange(5,20,5)},
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
            "kbinsdiscretizerwithnames__n_bins": np.arange(2, 6),
            "kbinsdiscretizerwithnames__strategy": ['uniform', 'quantile', 'kmeans'],
            "polynomialfeaturesdf__degree" : [2]
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
                linear_model.Ridge(max_iter=10000)
        ),
        # 3364
         param_grid={"ridge__alpha": np.arange(3364,3364,1),
            "kbinsdiscretizerwithnames__n_bins": np.arange(2, 3),
            "kbinsdiscretizerwithnames__strategy": ['quantile'],
            "polynomialfeaturesdf__degree" : [2]},
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
     'Ridge Interaction Regression' : {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
                KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
                OneHotEncoderOnlyCategorical(),
                CreateInteractions(),
                StandardScaler(),
                linear_model.Ridge(max_iter=10000)
        ),
        # 3364
         param_grid={"ridge__alpha": np.arange(2000,5000,1),
            "kbinsdiscretizerwithnames__n_bins": np.arange(2, 3),
            "kbinsdiscretizerwithnames__strategy": ['quantile']
            },
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
        param_grid={
                    #"lasso__alpha": np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
                    "lasso__alpha": np.arange(1,2,1),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,6),
                    "kbinsdiscretizerwithnames__strategy": ['uniform', 'quantile', 'kmeans'],
                    "polynomialfeaturesdf__degree" : [2]
                    },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Lasso Interaction Regression' : 
    {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            CreateInteractions(),
            StandardScaler(),
            linear_model.Lasso(max_iter=1000)
        ),
        param_grid={
                    "lasso__alpha": np.arange(3,5,0.1),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,3,1),
                    "kbinsdiscretizerwithnames__strategy": ['quantile']
                    },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'Lasso Interaction Only Regression' : 
    {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNamesOnlyCategorical(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            PolynomialFeaturesDF(degree = 2, interaction_only=True, include_bias = False),
            StandardScaler(),
            linear_model.Lasso(max_iter=300)
        ),
        param_grid={
                    "lasso__alpha": np.arange(4,5,1),
                    "kbinsdiscretizerwithnamesonlycategorical__n_bins": np.arange(1,11,1),
                    "kbinsdiscretizerwithnamesonlycategorical__strategy": ['quantile']
                    },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    },
    'ElasticNet Interaction Regression' : {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            CreateInteractions(),
            StandardScaler(),
            linear_model.ElasticNet(max_iter=1000)
        ),
        param_grid={"elasticnet__alpha": np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
                    "elasticnet__l1_ratio": np.arange(0,1.1,0.1),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,3,1),
                    "kbinsdiscretizerwithnames__strategy": ['quantile']
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
        param_grid={'kernelridge__alpha': np.concatenate((np.arange(0.1,1,0.1),np.arange(1,20,1))),
                    'kernelridge__degree': list(np.arange(1,5,1)),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,3,1),
                    "kbinsdiscretizerwithnames__strategy": ['quantile']
                    },
        scoring=scoring,
        refit='r2',
        return_train_score=True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
        
    },
    'AdaBoost Lasso Interaction Regression': {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            CreateInteractions(),
            StandardScaler(),
            AdaBoostRegressor(
                estimator=linear_model.Lasso(alpha=4.2, max_iter=1000),
                random_state = 0
            )
        ),
        param_grid={
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,3,1),
                    "kbinsdiscretizerwithnames__strategy": ['quantile'],
                    "adaboostregressor__n_estimators": np.arange(7,8,1),
                    'adaboostregressor__learning_rate': np.arange(0.005,0.006,0.001),
                    'adaboostregressor__loss' : ['exponential']
                    # 'linear', 'square', 
                    },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
        
    },
    'GradientBoosting Regression': {
        'fitting_pipline': GridSearchCV(
         estimator=make_pipeline(
                OneHotEncoderOnlyCategorical(),
                CreateInteractions(),
                StandardScaler(),
                GradientBoostingRegressor(
                                        loss='squared_error',
                                        learning_rate= 0.05,
                                        n_estimators=500
                                        
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
    'Support Vector Regression' : {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            StandardScaler(),
            SVR(kernel='poly')
            ),
        param_grid={'svr__epsilon': np.arange(0.05,0.2,0.05),
                    'svr__C': np.arange(1,4,1)
                    },
        scoring=scoring,
        refit='r2',
        return_train_score=True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
        
    },
    'LARS' : {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            CreateInteractions(),
            StandardScaler(),
            Lars()
            ),
        param_grid={'lars__n_nonzero_coefs': np.arange(500,700,100)
                    },
        scoring=scoring,
        refit='r2',
        return_train_score=True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
        
    },
    'Lasso Interaction Regression with PCA' : 
    {
        'fitting_pipline':  GridSearchCV(
        estimator=make_pipeline(
            KBinsDiscretizerWithNames(
                            encode='onehot-dense',
                            random_state=0),
            OneHotEncoderOnlyCategorical(),
            CreateInteractions(),
            StandardScaler(),
            PCA(),
            linear_model.Lasso(max_iter=1000)
        ),
        param_grid={
                    "lasso__alpha": np.arange(4.2,4.3,0.1),
                    "kbinsdiscretizerwithnames__n_bins": np.arange(2,3,1),
                    "kbinsdiscretizerwithnames__strategy": ['quantile'],
                    "pca__n_components": np.arange(10,11,1)
                    },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=-1
        )
    }
}