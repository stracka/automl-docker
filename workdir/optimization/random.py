#!/usr/bin/env python
# coding: utf-8

"""

Optimization with various strategies: 
For illustration, currently limited to tree ensemble classifiers


- Random search: randomsearch_gbdt

"""

import appconfig

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from sklearn.ensemble        import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics         import roc_auc_score, roc_curve, auc    
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score

import scipy.stats as sp

import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class expon_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, scale):
        self._distribution = sp.expon(scale)
        
    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = sp.loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
        


def randomsearch_gbdt(data, features, target, aux=None, weight=None): 
    """
    
    """

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    estimators = [
        ('scale', 'passthrough'),
        ('clfy', 'passthrough') 
    ]
    pipe = Pipeline(estimators)
    
    logger.info('Pipeline parameters:\n' + str(pipe.get_params()))

    p_grid =[
        {        
            "scale": [ PowerTransformer(method="yeo-johnson") , PowerTransformer(method="yeo-johnson") ],
            "clfy": [ RandomForestClassifier(random_state=0) ],
            "clfy__max_depth" : sp.randint(1, 20),
            "clfy__n_estimators" : loguniform_int(10,1000) ,
            "clfy__min_samples_split" : [5] , 
            "clfy__max_features" : ['sqrt','log2',None] #[0.4,1.],
        },
        {        
            "scale": [ PowerTransformer(method="yeo-johnson") , PowerTransformer(method="yeo-johnson") ],
            "clfy": [ GradientBoostingClassifier(random_state=0) ],
            "clfy__max_depth" : sp.randint(1, 20),
            "clfy__n_estimators" : expon_int(scale=100) ,
            "clfy__min_samples_split" : [5] , 
            "clfy__learning_rate" : sp.expon(scale=0.2),
            "clfy__max_features" : ['sqrt','log2',None] #[0.4,1.],
        }        
    ]
    
    clf = RandomizedSearchCV(estimator=pipe, param_distributions=p_grid, n_jobs = -1, n_iter=20, cv=inner_cv, scoring='roc_auc')
     
    clf.fit(data[features],data[target])
    logger.info("Tuned best params: {}".format(clf.best_params_))

    bdtname, bdt = clf.best_estimator_.steps[1]

    roc_auc = clf.best_score_

    logger.info("ROC AUC: {:.3f}".format(roc_auc))
    
    return  clf.best_estimator_, roc_auc

