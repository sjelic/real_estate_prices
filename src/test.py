import models
import pandas as pd
from sklearn.pipeline import make_pipeline
from preprocess import TypeConverter
from splitter import StratifiedRegressionSplit
from sklearn.metrics import r2_score

variables = pd.read_excel('/root/workspace/clanci/realestate_prices/data/covariates.xlsx')
data = pd.read_excel('/root/workspace/clanci/realestate_prices/data/hp_ljubljana_new_with_rooms.xlsx')

variables = variables.to_dict(orient='records')
target = []
features = []
for feature in variables:
    feature['type'] = int if feature['type'] == 'int' else float if feature['type'] == 'float' else str
    if feature['group'] == 'target':
        target = feature
        continue
    else:
        features.append(feature)

X = data[[feature['name'] for feature in features]]
y = data[target['name']]

# PREPROCESS PIPLINE
preprocc_pip = make_pipeline(TypeConverter(variables=features))
preprocc_pip.fit(X)
X = preprocc_pip.transform(X)

models.models['ElasticNet Polynomial Regression']['preprocessing_pipline'].fit(X)
X = models.models['ElasticNet Polynomial Regression']['preprocessing_pipline'].transform(X)

cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0)
model = models.models['ElasticNet Polynomial Regression']['fitting_pipline'].estimator
model.set_params(**{'elasticnet__alpha': 1, 'elasticnet__l1_ratio': 0.9})
for train, test in cv.split(X,y):
    model.fit(X.loc[train,:], y[train])
    y_pred = model.predict(X.loc[test,:])
    score = r2_score(y[test], y_pred)
    print(score)