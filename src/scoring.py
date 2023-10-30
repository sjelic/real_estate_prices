import metrics
from sklearn.metrics import make_scorer
import numpy as np

# custom or well known metrics
def mpe(y_true, y_pred):
    rel_errors = (np.array(y_true) - np.array(y_pred))/(1e-10 + np.array(y_pred))
    return  np.mean(rel_errors)

def mae(y_true, y_pred):
    abs_errors = (np.array(y_true) - np.array(y_pred))
    return  np.mean(abs_errors)

def mape(y_true, y_pred):
        rel_errors = np.abs((np.array(y_true) - np.array(y_pred))/(1e-10 + np.array(y_pred)))
        return  np.mean(rel_errors)

def cod(y_true, y_pred):
        ratios = np.array(y_true)/(1e-10 + np.array(y_pred))
        median = np.median(ratios)
        abs_dev = np.abs((ratios - median) / (1e-10 + median))
        return  np.mean(abs_dev)

def rmse(y_true, y_pred):
    sq_errors = (np.array(y_true) - np.array(y_pred))**2
    return  np.sqrt(np.mean(sq_errors))

# true - greater is better, false = otherwise
sense = {
        'mpe': False,
        'mae': False,
        'mape': False,
        'cod': False,
        'rmse': False,
        'r2': True
}


scoring = {
           'mpe': make_scorer(metrics.mpe, greater_is_better=sense['mpe']),
           'mae': make_scorer(metrics.mae, greater_is_better=sense['mae']),
           'mape': make_scorer(metrics.mape, greater_is_better=sense['mape']),
           'cod': make_scorer(metrics.cod, greater_is_better=sense['cod']),
           'rmse': make_scorer(metrics.rmse, greater_is_better=sense['rmse']),
           'r2': 'r2',
        }