import os
import glob
import sys
import pandas as pd


def get_report(model_report,metric,filename_model):
    model_report_rmse = pd.DataFrame([model_report[metric].values.T],columns=model_report['type'].values)
    model_report_rmse.reset_index(drop=True, inplace=True)
    model_report_rmse = model_report_rmse.add_prefix(metric+'_')

    model_report_rmse['model']=filename_model
    first_column = model_report_rmse.pop('model')
    model_report_rmse.insert(0, 'model', first_column)
    return model_report_rmse


def make_report(TARGET,PATH_RESULTS,PATH_RESULTS_MODELS,SUFFIX=''):
    
    filename_models = glob.glob(os.path.join(PATH_RESULTS_MODELS,'*[!md]'))
    
    list_rmse = []
    list_r2   = []

    for i in filename_models:
        filename_model = i.split('/')[-1]
        filename_report = os.path.join(PATH_RESULTS_MODELS,filename_model,filename_model+'_results.csv')

        try:
            print(filename_report)
            model_report = pd.read_csv(filename_report)

            df_rmse = get_report(model_report,'rmse',filename_model)
            df_r2 = get_report(model_report,'r2',filename_model)
            list_rmse.append(df_rmse)
            list_r2.append(df_r2)
        except:
            pass


    rmse_report = pd.concat(list_rmse)
    r2_report = pd.concat(list_r2)
    
    report = rmse_report.merge(r2_report,on='model',how='right')
    
    report.to_csv(os.path.join(PATH_RESULTS,'total_report_'+TARGET+'_'+SUFFIX+'.csv'),index=None)