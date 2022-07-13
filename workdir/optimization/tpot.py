#!/usr/bin/env python
# coding: utf-8

"""

Optimization with various strategies: 
For illustration, currently limited to tree ensemble classifiers

- TPOT: genetic programming


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

# TPOT
from tpot import TPOTClassifier
    
import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)



def tpot_gbdt(data, features, target, aux=None, weight=None, n_trials=None, timeout=600): 
    """
    http://epistasislab.github.io/tpot/using/
    http://epistasislab.github.io/tpot/using/#built-in-tpot-configurations

    https://github.com/EpistasisLab/tpot/issues/520

    https://www.kaggle.com/code/akashram/automl-overview-added-pycaret-hyperopt/notebook

    generations: Number of iterations to the run pipeline optimization process. The default is 100.

    population_size: Number of individuals to retain in the genetic programming population every generation. The default is 100.

    offspring_size: Number of offspring to produce in each genetic programming generation. The default is 100.

    mutation_rate: Mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the GP algorithm how many pipelines to apply random changes to every generation. Default is 0.9

    crossover_rate: Crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the genetic programming algorithm how many pipelines to "breed" every generation.

    """

    branch_names = features
    if aux is not None:
        branch_names = branch_names + [aux]
    
    tpot_config = {
        'sklearn.ensemble.GradientBoostingClassifier': {
            "max_depth" : [2, 5, 20],
            "n_estimators" : [100] 
        },
        'sklearn.ensemble.RandomForestClassifier': {
            "max_depth" : [2, 5, 20],
            "n_estimators" : [100] 
        }
    }

    clf = TPOTClassifier(generations = 3, population_size = 11,
                         random_state = 13, verbosity = 2,
                         cv=3, scoring='roc_auc',
                         config_dict=tpot_config,
                         n_jobs = -1,
                         max_time_mins = np.ceil(timeout/60) )

    y = data[target]
    X = data[features]

    clf.fit(X,y)
    clf_best = clf.fitted_pipeline_    

    #clf.export('tpot_exported_pipeline.py')
    
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)    
    roc_auc = cross_val_score(estimator=clf_best, X=X, y=y, n_jobs=-1, cv=inner_cv, scoring='roc_auc').mean()

    logger.info("ROC AUC: {:.3f}".format(roc_auc))

    clf_best.fit(data[features],data[target])

    return clf_best, roc_auc

