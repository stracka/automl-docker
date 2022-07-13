#!/usr/bin/env python
# coding: utf-8

"""

Optimization with various strategies: 
For illustration, currently limited to tree ensemble classifiers

- Optuna: tree-structured Parzen estimator (TPE). Not limited to sklearn

"""

import appconfig

import optuna
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

if appconfig._has_hepml:
    from hep_ml.losses           import BinFlatnessLossFunction
    from hep_ml.gradientboosting import UGradientBoostingClassifier

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    # suppress warnings triggered by hep_ml 

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

#https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
#https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html?highlight=powertransform

    
def optuna_gbdt(data, features, target, aux=None, weight=None, n_trials=10, timeout=600): 
    """
    https://arxiv.org/pdf/1907.10902.pdf
    https://optuna.org/
    """

    branch_names = features
    if aux is not None:
        branch_names = branch_names + [aux]
    
    SEED = 4005    

    
    def objective(trial):

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

        classifier_name = trial.suggest_categorical("clfy", ["GradientBoostingClassifier", "RandomForestClassifier"]) #"FlatnessLossUGBC"])

        if classifier_name == "GradientBoostingClassifier":
            
            estimators = [
                ('scale', PowerTransformer(method="yeo-johnson")),
                ('clfy', GradientBoostingClassifier(random_state=0) ) 
            ]
            pipe = Pipeline(estimators)

            params = {
                "clfy__max_depth" : trial.suggest_int("gbdt_max_depth", 2, 32, log=True),
                "clfy__n_estimators" : trial.suggest_int("gbdt_n_estimators", 20, 500, log=True),
                "clfy__max_features" : trial.suggest_categorical("gbdt_max_features", ['sqrt','log2',None])
            }
            
            pipe.set_params(**params)

            return cross_val_score(estimator=pipe, X=data[features], y=data[target], n_jobs=-1, cv=inner_cv, scoring='roc_auc').mean()
            
        elif classifier_name == "RandomForestClassifier" :
 
            estimators = [ 
                ('scale', PowerTransformer(method="yeo-johnson")),
                ('clfy', RandomForestClassifier(random_state=0)) 
            ] 
            
            pipe = Pipeline(estimators)
            
            params = {
                "clfy__max_depth" : trial.suggest_int("gbdt_max_depth", 2, 32, log=True),
                "clfy__n_estimators" : trial.suggest_int("gbdt_n_estimators", 20, 500, log=True),
                "clfy__max_features" : trial.suggest_categorical("gbdt_max_features", ['sqrt','log2',None])
            }

            pipe.set_params(**params)
 
            return cross_val_score(estimator=pipe, X=data[features], y=data[target], n_jobs=-1, cv=inner_cv, scoring='roc_auc').mean()

        elif classifier_name == "FlatnessLossUGBC" :

            # this classifier cannot appear in a pipeline as it requires pandas input, not a numpy array.
            # would have to implement manually, or with a wrapper
            if not appconfig._has_hepml:
                logger.warning("hep_ml module not found. FlatnessLossUGBC not available.")
                return 0

            else:
                pipe = UGradientBoostingClassifier(
                    loss= BinFlatnessLossFunction([aux], n_bins=2, uniform_label=1) ,
                    random_state=0,
                    train_features=features,
                    max_depth = trial.suggest_int("gbdt_max_depth", 2, 32, log=True),
                    n_estimators = trial.suggest_int("gbdt_n_estimators", 20, 500, log=True),
                    max_features = trial.suggest_categorical("gbdt_max_features", ['sqrt','log2',None])
                )
                
                X = data[branch_names]
                y = data[target]

                clsig=np.where(np.array(pipe.classes_)==1)[0][0]

                roc_aucs = []
                for itrain,itest in inner_cv.split(np.zeros(len(X)),y):
                    pipe.fit(X.iloc[itrain], y.iloc[itrain])
                    #twoclass_output = pipe.predict_proba(X.iloc[itest])[:,int(clsig)]
                    #fpr, tpr, thresholds = roc_curve(y.iloc[itest], twoclass_output)
                    #roc_auc = auc(fpr, tpr)                
                    roc_auc = roc_auc_score( y.iloc[itest], pipe.predict_proba(X.iloc[itest])[:,int(clsig)] )
                    roc_aucs = roc_aucs + [roc_auc]
                    
                return np.mean(roc_aucs)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
    logger.info(study.best_trial)
    logger.info(study.best_value)
    logger.info(study.best_params)
    
    estimators = []
    params = dict()
    clf_best = None
    
    y = data[target]
    X = data[features]
    
    if study.best_params['clfy'] == "GradientBoostingClassifier":            
        estimators = [
            ('scale', PowerTransformer(method="yeo-johnson")),
            ('clfy', GradientBoostingClassifier(random_state=0) ) 
        ]
        params = {
            "clfy__max_depth" :  study.best_params['gbdt_max_depth'],
            "clfy__n_estimators" :  study.best_params['gbdt_n_estimators'],
            "clfy__max_features" :  study.best_params['gbdt_max_features']
        }
        
        clf_best = Pipeline(estimators)
        clf_best.set_params(**params)   
        
    elif study.best_params['clfy'] == "RandomForestClassifier" :
        estimators = [
            ('scale', PowerTransformer(method="yeo-johnson")),
            ('clfy', RandomForestClassifier(random_state=0) ) 
        ]
        params = {
            "clfy__max_depth" :  study.best_params['gbdt_max_depth'],
            "clfy__n_estimators" :  study.best_params['gbdt_n_estimators'],
            "clfy__max_features" :  study.best_params['gbdt_max_features']
        }
        
        clf_best = Pipeline(estimators)
        clf_best.set_params(**params)   


    elif study.best_params['clfy'] == "FlatnessLossUGBC" :

        X = data[branch_names]

        if appconfig._has_hepml:
            clf_best = UGradientBoostingClassifier(
                loss= BinFlatnessLossFunction([aux], n_bins=2, uniform_label=1),
                random_state=0,
                train_features=features,
                max_depth = study.best_params['gbdt_max_depth'],
                n_estimators = study.best_params['gbdt_n_estimators'],
                max_features = study.best_params['gbdt_max_features']
            )
        else:
            logger.warning("Something went wrong. hep_ml not available but best model?!")
            clf_best = None
        
    clf_best.fit(X,y)
        
    return clf_best, study.best_value


