from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import PoissonRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor




total_models  = {
    
    "LinearRegression": LinearRegression(positive=False),
    "ElasticNet"      : ElasticNet(alpha=0.1,l1_ratio=0.01,random_state=0),
    "BayesianRidge"   : BayesianRidge(),
    "Ridge"           : Ridge(alpha=0.001,solver='sag'),
    "Lasso"           : Lasso(alpha=0.001),
    "PoissonRegressor": PoissonRegressor(alpha=0.1),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=0,max_depth=20),
    "XGBRegressor"    : xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, 
                         learning_rate = 0.1, max_depth = 10, n_estimators =128),
    "LGBMRegressor"   : LGBMRegressor(objective='regression', n_estimators=1000,learning_rate=0.1,subsample=0.8,
                                      colsample_bytree=0.8,max_depth=25,num_leaves=200,min_child_weight=300)
}


