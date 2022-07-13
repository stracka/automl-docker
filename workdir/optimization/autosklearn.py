#!/usr/bin/env python
# coding: utf-8

"""

Optimization with various strategies: 
For illustration, currently limited to tree ensemble classifiers

- AutoSklearn
https://www.analyticsvidhya.com/blog/2021/10/beginners-guide-to-automl-with-an-easy-autogluon-example/
https://machinelearningmastery.com/auto-sklearn-for-automated-machine-learning-in-python/
https://neptune.ai/blog/a-quickstart-guide-to-auto-sklearn-automl-for-machine-learning-practitioners
https://insaid.medium.com/applying-automl-part-1-using-auto-sklearn-1cf06a789c15
python
from autosklearn.experimental.askl2 import AutoSklearn2Classifier


- SMAC3: https://github.com/automl/SMAC3 bayesian optimization with random forests
- Spearmint (active) and GPyOpt (not active) use Gaussian Processes, bayesian optimization: https://sheffieldml.github.io/GPyOpt/index.html , https://github.com/HIPS/Spearmint


- AutoKeras

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


# auto-sklearn 
if appconfig._has_autosklearn:
    from autosklearn.metrics import roc_auc as askl_roc_auc
    from autosklearn.experimental.askl2 import AutoSklearn2Classifier
    from autosklearn.classification import AutoSklearnClassifier
else:
    logger.warning("Autosklearn module not found.")
 
    
#https://stackoverflow.com/questions/42297159/inheriting-from-a-class-which-doesnt-exist-in-python 

if appconfig._has_autosklearn:
    parent = AutoSklearnClassifier
else: 
    parent = object

class AutoSKPredictor(parent):
    def __init__(self):
        self._a = 1 

def autosklearn2_gbdt(data, features, target, aux=None, weight=None, n_trials=None, timeout=600): 
    """    
    https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_interpretable_models.html#sphx-glr-examples-40-advanced-example-interpretable-models-py
    """
    
    branch_names = features
    if aux is not None:
        branch_names = branch_names + [aux]
        
        
    '''
        # autosklearn2 has its own list
        # https://automl.github.io/auto-sklearn/master/_modules/autosklearn/experimental/askl2.html#AutoSklearn2Classifier
        include_dict={'classifier': ['random_forest', 'gradient_boosting'],
                  'feature_preprocessor': ['no_preprocessing',
                                           'polynomial',
                                           'select_percentile_classification']}
    '''

    if not appconfig._has_autosklearn:
        logger.info("autosklearn module not found.")
        return None, 0.5

    else:
        # use only 1 processor for the time being
        clf = AutoSklearn2Classifier(time_left_for_this_task=timeout,
                                     metric=askl_roc_auc,
                                     #per_run_time_limit=30, 
                                     memory_limit=3000,
                                     n_jobs=1) 

        y = data[target]
        X = data[features]
    
        clf.fit(X,y)
    
        #clf.export('tpot_exported_pipeline.py') 
    
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)    
        #roc_auc = cross_val_score(estimator=clf_best, X=X, y=y, n_jobs=-1, cv=inner_cv, scoring='roc_auc').mean()

        roc_auc = 0.5
        
        logger.info(clf.cv_results_)
        logger.info(clf.leaderboard)
        
        #predictions = automl.predict(X_test)
        #print("Accuracy score holdout: ", sklearn.metrics.accuracy_score(y_test, predictions))
        
        logger.info("ROC AUC: {:.3f}".format(roc_auc))
        
        clf.refit(X, y)
        return clf, roc_auc
    
