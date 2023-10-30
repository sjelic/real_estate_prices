import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import make_scorer
from matplotlib.widgets import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
import numpy as np
import json
from functools import reduce
from custom import StratifiedRegressionSplit


variables = pd.read_excel('../data/covariates.xlsx')
variables = variables.to_dict(orient='records')
targets = []
features = []
next = 0
for feature in variables:
    feature['type'] = int if feature['type'] == 'int' else float if feature['type'] == 'float' else str
    if feature['group'] == 'target':
        targets.append(feature)
        continue
    else:
        features.append(feature)
        

data = pd.read_excel('../data/hp_ljubljana_new_with_rooms.xlsx')
feature_names = [x['name'] for x in features]
target_names = [x['name'] for x in targets]
data = data[feature_names + target_names]


exclude_features_names = []
categorical_features = [x for x in features if  (x['type'] == str) and (x not in exclude_features_names)]
categorical_features_names = [x['name'] for x in categorical_features]
numerical_features = [x for x in features if  (x['type'] in (int, float)) and (x not in exclude_features_names)]
numerical_features_names = [x['name'] for x in numerical_features]

# target = {'name': 'price_m2', 'type': float}

onehot = OneHotEncoder(sparse=False)
onehot.fit(data[categorical_features_names])
encoded_features = list(reduce(lambda x,y: x + y, [[{"name": f"{feature['name']}_{cat}", "type": int, "group": feature['group']} for cat in cats] for cats, feature in zip(onehot.categories_, categorical_features)]))
encoded_features_names = [x['name'] for x in encoded_features]

X_encoded = onehot.transform(data[categorical_features_names])
X_numerical = data[numerical_features_names]
X = pd.DataFrame(columns=numerical_features_names + encoded_features_names)


X[numerical_features_names] = X_numerical
X[encoded_features_names] = X_encoded
X = X.astype({x['name'] : x['type'] for x in numerical_features + encoded_features})
if len(target_names)==1:
    y = data[target_names[0]]
    y = y.astype({targets[0]['name'] : targets[0]['type']})
else:
    y = data[target_names]
    y = y.astype({x['name'] : x['type'] for x in targets})
# # y = (y - y.std())/ y.mean()


# ridge = linear_model.Ridge(alpha = 2)
def cod(y_true, y_pred):
        ratios = np.array(y_pred)/np.array(y_true)
        median = np.median(ratios)
        abs_dev = np.abs(ratios - median)
        return  (100*np.mean(abs_dev))/np.abs(median)

def mape(y_true, y_pred):
        rel_errors = np.abs((np.array(y_true) - np.array(y_pred))/(1e-10 + np.array(y_pred)))
        return  100*np.mean(rel_errors)

scoring = {'r2': 'r2',
           'cod': make_scorer(cod, greater_is_better=False),
           'mape': make_scorer(mape, greater_is_better=False)}

gs = GridSearchCV(
        estimator=make_pipeline(StandardScaler(), linear_model.Ridge(max_iter=10000)),
        param_grid={'ridge__alpha': list(np.arange(1,20,0.2)) },
        scoring=scoring,
        refit='r2',
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
        n_jobs=8
)

gs_all_metric_results = gs.fit(X,y)


gs_crossvalidation_results = pd.DataFrame(gs_all_metric_results.cv_results_)
reg_coefs = pd.DataFrame({feature: [coef] for feature, coef in zip(['intercept'] + list(gs_all_metric_results.feature_names_in_) , [gs_all_metric_results.best_estimator_.steps[1][1].intercept_] + list(gs_all_metric_results.best_estimator_.steps[1][1].coef_))})
reg_coefs.to_excel('../results/ridge_reg_coefs.xlsx', index=False)
gs_crossvalidation_results.to_excel('../results/ridge_cv_results.xlsx', index=False)

print(
    f"""
    EVALUATION METRICS:
        Best alpha: {gs_all_metric_results.best_params_['ridge__alpha']},
       Train mean r2: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_train_r2']},
        Train std r2: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_train_r2']},
        Train mean cod: {-gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_train_cod']},
        Train std cod: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_train_cod']},
        Train mean mape: {-gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_train_mape']},
        Train std mape: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_train_mape']}
        Test mean r2: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_test_r2']},
        Test std r2: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_test_r2']},
        Test mean cod: {-gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_test_cod']},
        Test std cod: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_test_cod']},
        Test mean mape: {-gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_test_mape']},
        Test std mape: {gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_test_mape']}
    """
)
lasso_metrics = {
    'best_alpha': gs_all_metric_results.best_params_['ridge__alpha'],
    'train_mean_r2': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_train_r2'],
    'train_std_r2': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_train_r2'],
    'train_mean_cod': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_train_cod'],
    'train_std_cod': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_train_cod'],
    'train_mean_mape': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_train_mape'],
    'train_std_mape': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_train_mape'],
    'test_mean_r2': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_test_r2'],
    'test_std_r2': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_test_r2'],
    'test_mean_cod': -gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_test_cod'],
    'test_std_cod': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_test_cod'],
    'test_mean_mape': -gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'mean_test_mape'],
    'test_std_mape': gs_crossvalidation_results.loc[gs_all_metric_results.best_index_,'std_test_mape'],
}
with open('../results/ridge_metrics.json', 'w') as f:
    json.dump(lasso_metrics, f)