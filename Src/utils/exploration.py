import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 20
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True




def get_dataset(PATH_DATA,FNMAME,FEATURES,TARGET):
    fname =os.path.join(PATH_DATA,FNMAME)
    dat = np.load(fname, allow_pickle=True).item()
    print(dat.keys())

    X = dat[FEATURES].T
    y = dat[TARGET]
    
    return X,y
