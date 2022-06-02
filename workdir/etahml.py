#!/usr/bin/env python
# coding: utf-8

from datasets import *
from plots import *
from optimization import optuna_gbdt, randomsearch_gbdt, tpot_gbdt, autogluon_gbdt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import joblib
import pkg_resources
        
from sklearn.model_selection import StratifiedKFold

import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

    
#def etahml():
""" 
    config.json contains configuration parameters (add weight vars and configure usage)
    selection.json contains selection (the specified intervals are _dropped_!); events not selected will not be retained in the output dataset
    lims.json contains plot limits on variables; these will be also the training variables, but extra and aux vars listed in lims will be removed before training
    """ 

# load configuration 

with open('config.json', 'r') as f:
    config = json.load(f)
    
with open(config['limits'], 'r') as f:
    lims = json.load(f)

with open(config['selection'], 'r') as f:
    selection = json.load(f)

n_trials = config['n_trials'] 
timeout = config['timeout']  # in seconds
retrain = config['retrain'] 
prefix = config['prefix']
aux, target, extra, weightvars = config['aux'], config['target'], config['extra'], config['weightvars']
inputdata = list(zip(config['filenames'],config['treenames'],config['labels'],config['norm']))

branch_names = list(lims.keys())
features = branch_names.copy()

if aux not in branch_names:
    branch_names.append(aux)
if aux in features:
    features.remove(aux)
    
for i in extra:
    if i in features:
        features.remove(i)
    if i not in branch_names:
        branch_names.append(i)

logger.info(features)
logger.info(branch_names)

'''
data = pd.DataFrame([])
for filename,treename,label,norm in inputdata:
    tmp = open_root(fname = filename, tname = treename, branch_names=branch_names, label = label, norm = norm , selection=selection)
    data = data.append(tmp)
    logger.info(tmp)
    logger.info(len(data))
'''
input_dfs = []
for filename,treename,label,norm in inputdata:
    input_dfs.append( open_root(fname = filename, tname = treename, branch_names=branch_names, label = label, norm = norm , selection=selection) )
    
data = pd.concat(input_dfs, ignore_index=True)
logger.info(data)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
    
plot_inputs(df=data, lims=lims, target=target, nbins=50, ncols=4)
plt.savefig(prefix+'input.pdf')
plt.clf()



if retrain : 
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    itrain,itest = next(outer_cv.split(np.zeros(len(data)),data[target]))

    #clf, best_value = optuna_gbdt(data=data.iloc[itrain], features=features, target=target, aux=aux, n_trials=n_trials, timeout=timeout)
    clf, best_value = autogluon_gbdt(data=data.iloc[itrain], features=features, target=target, aux=aux, timeout=timeout)


    
    # apply to input data sample
    
    clsig=np.where(np.array(clf.classes_)==1)[0][0]
    data['clf_proba'] = clf.predict_proba(data[features])[:,int(clsig)] 
    data['is_train']  = 0
    data.iloc[itrain, data.columns.get_loc('is_train')] = 1 
    #data.loc[data.index[itrain], 'is_train'] = 1
    
    # persist model
    
    filename = prefix+'clf.pkl'
    with open(filename, 'wb') as f:  
        pickle.dump(clf, f)    
        
    filename = prefix+'clf.joblib'
    joblib.dump(clf, filename)

    # persist training data
    
    data.iloc[itrain].to_hdf(prefix+'data_train.h5',key='df',mode='w')
    data.iloc[itest].to_hdf(prefix+'data_test.h5',key='df',mode='w')
    
    data.iloc[itrain].to_csv(prefix+'data_train.csv')
    data.iloc[itest].to_csv(prefix+'data_test.csv')
    

    # persist list of packages
    installed_packages = pkg_resources.working_set
    installed_packages_tuple = sorted( [ (i.key, i.version) for i in installed_packages ] )
    installed_packages_df = pd.DataFrame(installed_packages_tuple, columns=['key','version'])
    installed_packages_df.to_csv(prefix+'packages.csv')
    
    # plot (this persists ROC AUC)
        
    plot_clf( df=data.iloc[itest] , clf='clf_proba' , target=target, nbins = 50)  
    plt.savefig(prefix+'output.pdf',bbox_inches='tight')
    plt.clf()

    plot_roc( df=data.iloc[itest] , clf='clf_proba' , target=target)  
    plt.savefig(prefix+'ROC.pdf',bbox_inches='tight')
    plt.clf()
    
    plot_importances(df=data.iloc[itest], clf=clf, features=features, target=target ) 
    plt.savefig(prefix+'importance.pdf',bbox_inches="tight")
    plt.clf()

else :
    logger.info('TBD')
       
    # Load the Model back from file
    with open(prefix+'clf.pkl', 'rb') as f:  
        clf = pickle.load(f)
        
    data_test = pd.read_hdf(prefix+'data_test.h5')
    clf3 = joblib.load(prefix+'clf.joblib')
    
    clsig=np.where(np.array(clf.classes_)==1)[0][0]
    data_test['clf_proba_2'] = clf.predict_proba(data_test[features])[:,int(clsig)] 

    


    
            
    '''
name = 'etaPi_2016_2018_MC_new.root'
data_fn = 'etaPi_2016_2018_MC.root' 
output_fn = 'bdt_vflat_'+name

signal_file = up.open(data_fn)
signal_tree = signal_file["DecayTree"]
branches = signal_tree.arrays(library='pd')
data = branches[features]

output = bdt.decision_function(data)
output.dtype = [('y', np.float64)]

from root_numpy import array2root
array2root(output,'' , "BDT",mode="RECREATE")
        
print('\nDone!')        
    '''

    '''
    result = permutation_importance(estimator = clf,
                                    X = data.iloc[itest, [data.columns.get_loc(c) for c in features] ] ,
                                    y = data.iloc[itest, data.columns.get_loc(target)] ,
                                    n_repeats = 10,
                                    random_state = 0)

    for i in np.argsort(result.importances_mean):
        logger.info('%20s  %6.4f +/- %6.4f'%(features[i],result.importances_mean[i],result.importances_std[i]))
    '''
    
#if __name__ == "__main__":
#    etahml()
            
    
'''

# also PMML and ONNX
# https://scikit-learn.org/stable/model_persistence.html
# 

Pickled models are often deployed in production using containers, like Docker, in order to freeze the environment and dependencies.

'''

