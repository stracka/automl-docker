
import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics         import roc_auc_score, roc_curve, auc    
from sklearn.inspection      import permutation_importance


def plot_inputs ( df, lims , target, nbins=50, ncols=4) : 

    branch_names = list(lims.keys())
    
    plt.figure(figsize=(12,16))
    nrows = int(len(branch_names)/ncols+1)

    i = 0
    for b in branch_names:
        plt.subplot(nrows,ncols,i+1)
        plt.subplots_adjust(hspace=.3)

        cs = df[df[target]>0.5][b]
        cb = df[df[target]<0.5][b]
        myrange = (lims[b][0],lims[b][1])
        
        if (myrange[0] > myrange[1]):
            myrange = None

        _n, _bins, _patches = plt.hist(cs,bins=nbins,histtype='step',label='Signal',density=True,range=myrange)

        plt.hist(cb,bins=_bins,histtype='step',label='BG',density=True)

        i = i+1
        
        plt.legend()
        plt.title(b)
        
    return plt

def plot_clf ( df, clf , target, nbins=50) : 

    plot_range = (df[clf].min(), df[clf].max())
    plt.hist(df.loc[df[target]==0,clf],range=plot_range,density=True,histtype='step',bins=nbins,label='BG')
    plt.hist(df.loc[df[target]==1,clf],range=plot_range,density=True,histtype='step',bins=nbins,label='Signal')

    plt.legend()
    plt.title('BDT output')
        
    return plt

def plot_roc ( df, clf, target ) : 

    fpr, tpr, thresholds = roc_curve(df[target], df[clf])
    roc_auc = auc(fpr, tpr)                
    
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.3f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()

    return plt

        

def plot_importances ( df, clf, features, target ) :

    result = permutation_importance(estimator = clf,
                                    X = df[features],
                                    y = df[target] ,
                                    n_repeats = 10,
                                    random_state = 0)

    for i in np.argsort(result.importances_mean):
        logger.info('%20s  %6.4f +/- %6.4f'%(features[i],result.importances_mean[i],result.importances_std[i]))

    n_features = len(features)
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features),result.importances_mean, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    # https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance

    return plt
