import pandas as pd
import pickle
from scoring import scoring, sense
import json
import sys

def evaluate(model_name):

    with open(f'/workspaces/workspace/clanci/realestate_prices/models/{model_name}.pickle', 'rb') as f:
        gs_res = pickle.load(f)
        f.close()
        

    gs_crossvalidation_results = pd.DataFrame(gs_res.cv_results_)
    gs_crossvalidation_results.to_excel(f'/workspaces/workspace/clanci/realestate_prices/results/{model_name}_results.xlsx', index=False)
    try:
        reg_coefs = {feature: coef for feature, coef in zip(['intercept'] + list(gs_res.feature_names_in_) , [gs_res.best_estimator_.steps[1][1].intercept_] + list(gs_res.best_estimator_.steps[1][1].coef_))}
        
        reg_coefs_ser = pd.Series(reg_coefs, index = reg_coefs.keys())
        reg_coefs_ser.to_excel(f'/workspaces/workspace/clanci/realestate_prices/results/{model_name}_coefs.xlsx')
    except Exception as e:
        print('No regression coefficients...')
   

    best_params = gs_res.best_params_
    metrics = {}
    for metric in scoring:
        for dataset in ['train', 'test']:
            for aggr in ['mean', 'std']:
                metrics[f'{aggr}_{dataset}_{metric}'] = gs_crossvalidation_results.loc[gs_res.best_index_,f'{aggr}_{dataset}_{metric}']
                if aggr=='std' or sense[metric]:
                    continue
                metrics[f'{aggr}_{dataset}_{metric}'] = -metrics[f'{aggr}_{dataset}_{metric}']
    best_params.update(metrics)

    print(json.dumps(best_params, indent=4, default=int))
    with open(f'/workspaces/workspace/clanci/realestate_prices/results/{model_name}_metrics.json', 'w') as f:
        json.dump(best_params, f, default=int)
        f.close()
        
if __name__ == '__main__':
    evaluate(sys.argv[1])
