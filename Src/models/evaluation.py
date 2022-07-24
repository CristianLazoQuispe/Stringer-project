from sklearn import metrics

def get_metrics_regression(y_true,y_pred):
    
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    r2   = metrics.r2_score(y_true, y_pred)
    
    return rmse,r2