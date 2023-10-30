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

# y = (y - y.std())/ y.mean()


# ridge = linear_model.Ridge(alpha = 2)
from sklearn.model_selection import cross_val_score


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


estimator = make_pipeline(StandardScaler(), linear_model.LinearRegression())
estimator.fit(X,y)

cv_res = cross_validate(
        estimator=estimator,
        X = X,
        y = y,
        scoring=scoring,
        return_train_score = True,
        cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0)
)

crossvalidation_results = pd.DataFrame(cv_res)
reg_coefs = pd.DataFrame({feature: [coef] for feature, coef in zip(['intercept'] + list(estimator.feature_names_in_) , [estimator[1].intercept_] + list(estimator[1].coef_))})
reg_coefs.to_excel('../results/reg_coefs.xlsx', index=False)
crossvalidation_results.to_excel('../results/regression_cv_results.xlsx', index=False)

print(
    f"""
    EVALUATION METRICS:
        Train mean r2: {crossvalidation_results['train_r2'].mean()},
        Train std r2: {crossvalidation_results['train_r2'].std()},
        Train mean cod: {-crossvalidation_results['train_cod'].mean()},
        Train std cod: {crossvalidation_results['train_cod'].std()},
        Train mean mape: {-crossvalidation_results['train_mape'].mean()},
        Train std mape: {crossvalidation_results['train_mape'].std()},
        Test mean r2: {crossvalidation_results['test_r2'].mean()},
        Test std r2: {crossvalidation_results['test_r2'].std()},
        Test mean cod: {-crossvalidation_results['test_cod'].mean()},
        Test std cod: {crossvalidation_results['test_cod'].std()},
        Test mean mape: {-crossvalidation_results['test_mape'].mean()},
        Test std mape: {crossvalidation_results['test_mape'].std()}
    """
)
lasso_metrics = {
    'train_mean_r2': crossvalidation_results['train_r2'].mean(),
    'train_std_r2': crossvalidation_results['train_r2'].std(),
    'train_mean_cod': -crossvalidation_results['train_cod'].mean(),
    'train_std_cod': crossvalidation_results['train_cod'].std(),
    'train_mean_mape': -crossvalidation_results['train_mape'].mean(),
    'train_std_mape': crossvalidation_results['train_mape'].std(),
    'test_mean_r2': crossvalidation_results['test_r2'].mean(),
    'test_std_r2': crossvalidation_results['test_r2'].std(),
    'test_mean_cod': -crossvalidation_results['test_cod'].mean(),
    'test_std_cod': crossvalidation_results['test_cod'].std(),
    'test_mean_mape': -crossvalidation_results['test_mape'].mean(),
    'test_std_mape': crossvalidation_results['test_mape'].std()
}
with open('../results/regression_metrics.json', 'w') as f:
    json.dump(lasso_metrics, f)