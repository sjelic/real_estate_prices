import pandas as pd
from sklearn.pipeline import make_pipeline
import models
from preprocess import TypeConverter
import pickle
import sys
import os

DATA_DIR = './data'
MODEL_PATH = './models'
RESULT_PATH = './results'

def fit_model(model_name, data_path = os.path.join(DATA_DIR,'hp_ljubljana_new_with_rooms.xlsx'), variable_path = os.path.join(DATA_DIR,'covariates.xlsx')):
    variables = pd.read_excel(variable_path)
    data = pd.read_excel(data_path)
    
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
    
    # models.models[model_name]['preprocessing_pipline'].fit(X)
    # X = models.models[model_name]['preprocessing_pipline'].transform(X)
    
    
    return models.models[model_name]['fitting_pipline'].fit(X,y)


def fit_and_save_model(model_name, data_path = os.path.join(DATA_DIR,'hp_ljubljana_new_with_rooms.xlsx'), variable_path = os.path.join(DATA_DIR,'covariates.xlsx')):
    gs = fit_model(model_name, data_path, variable_path)
    with open(os.path.join(MODEL_PATH,f"{model_name.replace(' ', '_')}.pickle"), 'wb' ) as f:
        pickle.dump(gs,f)

def fit_and_save_all_models(data_path = os.path.join(DATA_DIR,'hp_ljubljana_new_with_rooms.xlsx'), variable_path = os.path.join(DATA_DIR,'covariates.xlsx')):
    for model_name in models.models:
        gs = fit_model(model_name, data_path, variable_path)
        with open(os.path.join(MODEL_PATH,f"{model_name.replace(' ', '_')}.pickle"), 'wb' ) as f:
            pickle.dump(gs,f)


# fit_and_save_model("Lasso Polynomial Regression")
if __name__ == '__main__':
    fit_and_save_model(sys.argv[1])


    