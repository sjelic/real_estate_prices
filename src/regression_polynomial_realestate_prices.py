import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from scoring import scoring
import models
from splitter import StratifiedRegressionSplit
from preprocess import OneHotEncoderOnlyCategorical, PolynomialFeaturesDF, TypeConverter


variables = pd.read_excel('./clanci/realestate_prices/data/covariates.xlsx')
data = pd.read_excel('./clanci/realestate_prices/data/hp_ljubljana_new_with_rooms.xlsx')


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


preprocc_pip = make_pipeline(TypeConverter(variables=features))
preprocc_pip.fit(X)
X = preprocc_pip.transform(X)

# gs = GridSearchCV(
#         estimator=make_pipeline(
#                 StandardScaler(),
#                 linear_model.LinearRegression()
#         ),
#         param_grid={},
#         scoring=scoring,
#         refit='r2',
#         return_train_score = True,
#         cv=StratifiedRegressionSplit(n_splits=10, n_bins = 10, test_size=0.3, random_state=0),
#         n_jobs=-1
#     )

gs_res = models.models['Least Angle Regression'].fit(X,y)


gs_crossvalidation_results = pd.DataFrame(gs_res.cv_results_)
reg_coefs = pd.DataFrame({feature: [coef] for feature, coef in zip(['intercept'] + list(gs_res.feature_names_in_) , [gs_res.best_estimator_.steps[1][1].intercept_] + list(gs_res.best_estimator_.steps[1][1].coef_))})
reg_coefs.to_excel('./clanci/realestate_prices/results/test_coef.xlsx', index=False)
gs_crossvalidation_results.to_excel('./clanci/realestate_prices/results/test.xlsx', index=False)
