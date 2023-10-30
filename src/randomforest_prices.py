import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model, ensemble
from sklearn.metrics import make_scorer
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
n_features = len(numerical_features) + len(encoded_features)

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

# gs = GridSearchCV(
#         estimator=make_pipeline(StandardScaler(), ensemble.RandomForestRegressor(bootstrap=True, max_features=int(n_features), oob_score=True, n_jobs=-1, random_state = 0)),
#         param_grid={'randomforestregressor__n_estimators': list(np.arange(80,110,10))},
#         scoring=scoring,
#         refit='r2',
#         cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0)
# )

strat_split = StratifiedRegressionSplit(n_splits=1, n_bins = 10, test_size=0.3, random_state=0)



train, test = next(iter(strat_split.split(X,y)))

Xtrain = pd.DataFrame(X.loc[train], columns=X.columns)
ytrain = pd.Series(y.loc[train])

scaler = StandardScaler().fit(Xtrain)

Xtrain = scaler.transform(Xtrain)
Xtrain = pd.DataFrame(Xtrain, columns=X.columns)

Xtest = pd.DataFrame(X.loc[test], columns=X.columns)
ytest = pd.Series(y.loc[test])

Xtest = scaler.transform(Xtest)
Xtest = pd.DataFrame(Xtest, columns=X.columns)


estimator= ensemble.RandomForestRegressor(bootstrap=True, max_features=int(n_features/3), min_samples_split = 4, min_samples_leaf=2, oob_score=True, n_jobs=-1, random_state = 0, n_estimators=500,  max_samples = 0.7)

# make_pipeline(, )

gs_all_metric_results = estimator.fit(Xtrain, ytrain)

rf_results = [{'feature': key, 'importance': value} for key, value in zip(estimator.feature_names_in_, estimator.feature_importances_)]
rf_results = sorted(rf_results, key=lambda x: -x['importance'])
feature_importances = pd.DataFrame(rf_results)
feature_importances.to_excel('../results/randomforest_feature_importances.xlsx', index = False)

print(
    f"""
    EVALUATION METRICS:
        Train mean r2: {estimator.score(Xtrain, ytrain)},
        Test  mean r2: {estimator.score(Xtest, ytest)},
        Train mean cod: {cod(estimator.predict(Xtrain), ytrain)},
        Test  mean cod: {cod(estimator.predict(Xtest), ytest)},
        Train mean mape: {mape(estimator.predict(Xtrain), ytrain)},
        Test mean mape: {mape(estimator.predict(Xtest), ytest)},
        
    """
)
metrics = {
    'train_mean_r2': estimator.score(Xtrain, ytrain),
    'test_mean_r2': estimator.score(Xtest, ytest),
    'train_mean_cod': cod(estimator.predict(Xtrain), ytrain),
    'test_mean_cod': cod(estimator.predict(Xtest), ytest),
    'train_mean_mape': mape(estimator.predict(Xtrain), ytrain),
    'test_mean_mape': mape(estimator.predict(Xtest), ytest)
}
with open('../results/randomforest_metrics.json', 'w') as f:
    json.dump(metrics, f)