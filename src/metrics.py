import numpy as np

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




    



    
    