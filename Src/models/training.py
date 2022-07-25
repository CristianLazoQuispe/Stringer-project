from . import evaluation
import numpy as np
import pickle

from sklearn.model_selection import KFold
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 20
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True


def cross_validate(x_train, y_train, X_test, y_test,n_splits,model,model_name="",path_model="",n_precision=5):
    print('*'*10,' Training ',model_name,'*'*10)
    
    # Initialize the split method
    kfold_iterator = KFold(n_splits)
    
    train_rmse = []
    val_rmse = []
    
    train_r2 = []
    val_r2 = []    
    
    y_test_pred_total   = []
    
    results_fold = []

    for i_split, (train_indices, val_indices) in enumerate(kfold_iterator.split(x_train)):

        print('*'*5,' Fold ',i_split+1,'*'*5)
        x_cv_train = x_train[train_indices]
        y_cv_train = y_train[train_indices]
        x_cv_val = x_train[val_indices]
        y_cv_val = y_train[val_indices]
        
        #Training

        if model_name == "XGBRegressor":
            model.fit(x_cv_train, y_cv_train, eval_set=[(x_cv_train, y_cv_train), (x_cv_val, y_cv_val)],
                      early_stopping_rounds=10, verbose=50,eval_metric=['rmse'])       
        elif model_name =="LGBMRegressor":
            
            model.fit(x_cv_train,  y_cv_train, eval_set=[(x_cv_train, y_cv_train), (x_cv_val, y_cv_val)],
                      early_stopping_rounds=10, verbose=50,eval_metric='rmse')

        else:
            
            model.fit(x_cv_train,y_cv_train)
        
        y_cv_train_pred = model.predict(x_cv_train) 
        y_cv_val_pred   = model.predict(x_cv_val) 
        y_test_pred     = model.predict(X_test)
        
        # allow only positive values
        y_cv_train_pred[y_cv_train_pred<=0] = 0  
        y_cv_val_pred[y_cv_val_pred<=0]     = 0 
        y_test_pred[y_test_pred<=0]        = 0 
        
        #evaluation
        rmse_cv_train,r2_cv_train = evaluation.get_metrics_regression(y_cv_train,y_cv_train_pred)
        rmse_cv_val  ,r2_cv_val   = evaluation.get_metrics_regression(y_cv_val,y_cv_val_pred)
        rmse_test    ,r2_test     = evaluation.get_metrics_regression(y_test,y_test_pred)

        
        rmse_cv_train = np.round(rmse_cv_train,n_precision)
        r2_cv_train   = np.round(r2_cv_train,n_precision)

        rmse_cv_val = np.round(rmse_cv_val,n_precision)
        r2_cv_val   = np.round(r2_cv_val,n_precision)

        rmse_test = np.round(rmse_test,n_precision)
        r2_test   = np.round(r2_test,n_precision)
    
        
        print("Train fold : RMSE =",rmse_cv_train,"R2 =",r2_cv_train)
        print("Val  fold  : RMSE =",rmse_cv_val,"R2 =",r2_cv_val)
        print("Test total : RMSE =",rmse_test,"R2 =",r2_test)
                                   
        results_fold.append([i_split+1,'training fold',rmse_cv_train,r2_cv_train])
        results_fold.append([i_split+1,'validation fold',rmse_cv_val,rmse_cv_val])
        results_fold.append([i_split+1,'testing fold',rmse_test,r2_test])
                                 
        train_rmse.append(rmse_cv_train)
        train_r2.append(r2_cv_train)
        val_rmse.append(rmse_cv_val)
        val_r2.append(r2_cv_val)
        
        # save predictions
        y_test_pred_total.append(y_test_pred)

        create_folder(os.path.join(path_model))
        create_folder(os.path.join(path_model,model_name))

        # save
        filename = os.path.join(path_model,model_name,model_name+'_fold_'+str(i_split+1)+'.pkl')
        with open(filename,'wb') as f:
            print('Saving model ...')
            pickle.dump(model,f)
            print('Model saved !  in ',filename)
    
    y_test_pred_total = np.array(y_test_pred_total)
    y_test_pred_cv =  y_test_pred_total.mean(axis=0)
    y_test_pred_cv[y_test_pred_cv<=0] = 0 

    rmse_test_cv_total    ,r2_test_cv_total    = evaluation.get_metrics_regression(y_test,y_test_pred_cv)


    mean_train_rmse = np.round(np.mean(train_rmse),n_precision)
    mean_train_r2   = np.round(np.mean(train_r2),n_precision)

    mean_val_rmse = np.round(np.mean(val_rmse),n_precision)
    mean_val_r2   = np.round(np.mean(val_r2),n_precision)
    
    mean_test_rmse = np.round(rmse_test_cv_total,n_precision)
    mean_test_r2   = np.round(r2_test_cv_total,n_precision)
    
    results =  "Train mean cv : RMSE = "+str(mean_train_rmse)+" |  R2 ="+str(mean_train_r2)+'\n'
    results += "Val   mean cv : RMSE = "+str(mean_val_rmse)  +" |  R2 ="+str(mean_val_r2)+'\n'
    results += "Test  mean cv : RMSE = "+str(mean_test_rmse) +" |  R2 ="+str(mean_test_r2)
    
    print(results)
    
    
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(len(y_test)),y_test,'-bo',alpha=0.5,label='real values')
    plt.plot(range(len(y_test_pred_cv)),y_test_pred_cv,'-ro',alpha=0.5,label='predicted values')
    plt.legend(loc="upper left")
    plt.title(model_name+ '\n'+results)
    #plt.show()
    filename = os.path.join(path_model,model_name,model_name+'_test_prediction.png')
    print('Saving plot in ',filename)
    fig.savefig(filename, bbox_inches='tight',facecolor='white')
    plt.close(fig)    # close the figure window

    
    
    print('*'*30)
    metrics = pd.DataFrame(results_fold,columns=['Fold','type','rmse','r2'])
    filename = os.path.join(path_model,model_name,model_name+'_results_by_fold.csv')
    print('Saving folds report in ',filename)
    metrics.to_csv(filename,index=False)

    metrics = pd.DataFrame()
    metrics['type'] = ['training mean cv','validation mean cv','testing mean cv']
    metrics['rmse'] = [mean_train_rmse,mean_val_rmse,mean_test_rmse]
    metrics['r2'] = [mean_train_r2,mean_val_r2,mean_test_r2]
    
    filename = os.path.join(path_model,model_name,model_name+'_results.csv')
    print('Saving summary cv report in ',filename)                           
    metrics.to_csv(filename,index=False)
    
    return y_test_pred_cv

def create_folder(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
        
def training_model_cv(X_train,y_train,X_test,y_test,n_splits,total_models,path_model):
    
    for model_name in total_models.keys():
        
        model = total_models[model_name]
        
        y_test_pred_cv = cross_validate(X_train, y_train, X_test, y_test,
                                                 n_splits=n_splits,
                                                 model = model,model_name=model_name,
                                                 path_model=path_model)
