from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import os
import sys




def get_dataset(PATH_DATA,FNMAME,FEATURES,TARGET):
    fname =os.path.join(PATH_DATA,FNMAME)
    dat = np.load(fname, allow_pickle=True).item()
    print(dat.keys())

    X = dat[FEATURES].T
    
    if TARGET == "pupilCOM_x":
        y = dat['pupilCOM'][:,0]
    elif TARGET == "pupilCOM_y":
        y = dat['pupilCOM'][:,1]
    else:
        y = dat[TARGET]    
    
    print(" X size :",X.shape)
    print(" y size :",y.shape)
    return X,y


def simple_temporal_splitting(X,y,test_size):
    n = int(len(X)*0.25)

    X_train = X[:-n]
    y_train = y[:-n]
    


    X_test  = X[-n:]
    y_test  = y[-n:]
    
    X_train,y_train = shuffle(X_train,y_train)
    X_test,y_test   = shuffle(X_test,y_test)    
    
    print('X train : ',X_train.shape,"y train :",y_train.shape)
    print('X test  : ',X_test.shape ,"y test  :",y_test.shape)
    
    return X_train,X_test,y_train,y_test


def thresholding(X_train,X_test,thresholing_max=95):
    total_sresp = X_train.flatten()
    for percentile in [25,50,75,90,100]:
        print('percentile '+str(percentile)+': ',np.percentile(total_sresp, percentile))
    print("Applying thresholding ..")
    threshold_sresp = np.percentile(total_sresp, thresholing_max)
    print('threshold neural activity at '+str(thresholing_max)+
          '%:',threshold_sresp)
    
    print("After    thresholding ..")
    X_train[X_train>threshold_sresp] = threshold_sresp
    X_test[X_test>threshold_sresp] = threshold_sresp
    
    total_sresp = X_train.flatten()
    for percentile in [25,50,75,90,100]:
        print('percentile '+str(percentile)+': ',np.percentile(total_sresp, percentile))
    
    print("Thresholding completed!")

    return X_train, X_test


def normalization(X_train,X_test):
    print("Normalization process ..")
    
    std_scale = StandardScaler().fit(X_train)
    
    print('X train : mean [0]=',X_train.mean(axis=0)[0],"std [0]=",X_train.mean(axis=0)[0])
    print('X test  : mean [0]=',X_test.mean(axis=0)[0] ,"std [0]=",X_test.std(axis=0)[0])
    
    X_train_std = std_scale.transform(X_train)
    X_test_std  = std_scale.transform(X_test)

    '''
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    X_train_std = (X_train-mean)/std
    X_test = (X_test-mean)/std
    
    '''
    
    print("Normalization completed!")
    print('X train : mean [0]=',X_train_std.mean(axis=0)[0],"std [0]=",X_train_std.mean(axis=0)[0])
    print('X test  : mean [0]=',X_test_std.mean(axis=0)[0] ,"std [0]=",X_test_std.std(axis=0)[0])
    
    return X_train_std,X_test_std




def my_PCA(X_train,X_test,max_variance_explanation, n_components=3000):
    print("PCA processing ..")
    print("X_train size :",X_train.shape)
    print("X_test size  :",X_test.shape)

    my_model = PCA(n_components=3000)
    my_model.fit_transform(X_train)


    explained_variance_ratio_ = my_model.explained_variance_ratio_
    
    accumulate_explained_variance_ratio_ = np.cumsum(explained_variance_ratio_)
    
    n_principal_components = n_components
    
    for i in range(len(accumulate_explained_variance_ratio_)):
                   
        aux_variance = accumulate_explained_variance_ratio_[i]
        if aux_variance>=max_variance_explanation:
            n_principal_components = i
                   
    print("PCA completed!")

    print('# principal components :',n_principal_components+1)
    print('max variance explained :',accumulate_explained_variance_ratio_[n_principal_components])
    
    X_train_pca = my_model.transform(X_train)[:,:n_principal_components]
    X_test_pca  = my_model.transform(X_test)[:,:n_principal_components]
    
    print("X_train_pca size :",X_train_pca.shape)
    print("X_test_pca size  :",X_test_pca.shape)

    return X_train_pca,X_test_pca


def auto_PCA(X_train,X_test,max_variance_explanation):
    print("PCA processing ..")
    print("X_train size :",X_train.shape)
    print("X_test size  :",X_test.shape)
    
    pca = PCA(max_variance_explanation)
    pca.fit(X_train)
    print("PCA completed!")
    print("# principal components :",pca.n_components_)
    print('max variance explained :',np.sum(pca.explained_variance_ratio_))

    X_train_pca = pca.transform(X_train)
    X_test_pca =  pca.transform(X_test)
    
    print("X_train_pca size :",X_train_pca.shape)
    print("X_test_pca size  :",X_test_pca.shape)

    return X_train_pca,X_test_pca
