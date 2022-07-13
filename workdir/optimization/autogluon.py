#!/usr/bin/env python
# coding: utf-8

"""

Optimization with various strategies: 
For illustration, currently limited to tree ensemble classifiers

- AutoGluon (Tabular)
https://explained.ai/rf-importance/

https://www.analyticsvidhya.com/blog/2021/10/beginners-guide-to-automl-with-an-easy-autogluon-example/

"""

import appconfig

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# suppress warnings triggered by hep_ml 

from hep_ml.losses           import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier

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

if appconfig._has_autogluon: 
    from autogluon.tabular import TabularDataset, TabularPredictor
    parent = TabularPredictor
else:
    parent = object


class AutoGluonPredictor(parent):
    """
        https://github.com/awslabs/autogluon/issues/1493
        Adding classes_ and predict_proba
        Wrapper for compatibility with scikit-learn pipelines and methods
    """


    if not appconfig._has_autogluon: 
        logger.warning("AutoGluon module not found.")
        def __init__(self):super().__init__()
                     
    else:
        def __init__(self,
                     label='',
                     problem_type=None,
                     eval_metric=None,
                     verbosity=0,
                     sample_weight=None,
                     weight_evaluation=False,
                     groups=None,
                     hyperparameters = None,
                     time_limit = None,
                     presets='best_quality'):
            super().__init__(label=label,
                             problem_type=problem_type,
                             eval_metric=eval_metric,
                             verbosity=verbosity,
                             sample_weight=sample_weight,
                             weight_evaluation=weight_evaluation,
                             groups=groups)
            
            self.tl = time_limit
            self.hp = hyperparameters
            self.pre = presets
                
            #presets = 
            #'best_quality', applies stacking/bagging to provide best quality, time consuming
            #'medium_quality', which produces less accurate models but facilitates faster prototyping
            #'good_quality', quickly deploy basic model
            #'optimize_for_deployment', quickly deploy basic model
        
        @property
        def classes_(self):
            return super().class_labels

        def fit(self, X, y=None, sample_weight=None, check_input=True):
            X_df = pd.DataFrame(X)
            y_df = pd.DataFrame(y)
            X_names = X_df.columns
            y_names = y_df.columns
            
            # This code has not been well considered, but is to resolve some errors. 
            if len(X_names) != len(set(X_names)):
                X_names = [f'X{i}' for i in range(len(X_names))]
            else:
                if len(list(set(X_names) & set(y_names))) >= 1 :
                    X_names = [f'X{i}' for i in range(len(X_names))]
                    y_names = ['y0']

            X_df.columns = X_names
            y_df.columns = y_names
            y_name = y_names[0]
            self.X_names = X_names
            self.y_names = y_names
            ###########
            
            self.__init__(label=y_name,
                          hyperparameters = self.hp,
                          time_limit = self.tl,
                          presets=self.pre)
            
            train_data = pd.concat([X_df, y_df], axis=1)
        
            super().fit(train_data,
                        presets=self.pre,
                        time_limit = self.tl,
                        hyperparameters = self.hp )
            
            
            return self


        def score(self, X, y, sample_weight=None):
        
            from sklearn.metrics         import accuracy_score
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        '''
        from  sklearn.metrics         import roc_auc_score #, roc_curve, auc, accuracy_score
        return roc_auc_score(y, self.predict_proba(X)[:,super().positive_class],  sample_weight=sample_weight)
        '''                      
            
        def predict(self, X, y=None, sample_weight=None, check_input=True):
            X_df = pd.DataFrame(X)
            X_df.columns = self.X_names
            
            return super().predict(X_df)
        
        def predict_proba(self, X, **predict_proba_params):
            """
            **predict_proba_params : Ignored. 
            
            Returns
            -------
            y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
            """
            X_df = pd.DataFrame(X)
            X_df.columns = self.X_names

            return super().predict_proba(X_df, model=None, as_pandas=False, as_multiclass=True)
        
    
        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self
            
        def get_params(self, deep=True):
            """
            Parameters
            ----------
            deep : Ignored. (for compatibility with sklearn)
            Returns
            ----------
            self : returns an dictionary of parameters.
            """
            
            params = {}
            return params        
        
        
def autogluon_gbdt(data, features, target, aux=None, weight=None, n_trials=None, timeout=600): 
    """
        https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-indepth.html
        https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-custom-model.html
        
        https://github.com/awslabs/autogluon/issues/1479
        https://github.com/awslabs/autogluon/issues/401
        
        https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor.fit
        
        Stable model options include:
                    'GBM' (LightGBM)
                    'CAT' (CatBoost)
                    'XGB' (XGBoost)
                    'RF' (random forest)
                    'XT' (extremely randomized trees)
                    'KNN' (k-nearest neighbors)
                    'LR' (linear regression)
                    'NN_MXNET' (neural network implemented in MXNet)
                    'NN_TORCH' (neural network implemented in Pytorch)
                    'FASTAI' (neural network with FastAI backend)

        The hyperparameters dict currently has the following possible keys:
        'NN', 'GBM', 'CAT', 'RF', 'XT', 'KNN', 'custom'
        corresponding to 7 different models that can be trained.
        
        To omit one of these models during task.fit(), you simply drop that key from the
        hyperparameters dict. For example, you can tell AutoGluon to only consider RF and GBM models (with their default hyperparameter settings) via:
        
        task.fit(..., hyperparameters={'RF':{}, 'GBM':{}})
        
    """

  
    if not appconfig._has_autogluon: 
        logger.warning("AutoGluon module not found.")
        return None, 0
    
    else:
        
        branch_names = features
        if aux is not None:
            branch_names = branch_names + [aux]

        y = data[target]
        X = data[features]
        Xy = data[features+[target]]

        params={'RF':{}, 'GBM':{}}
    
        tp = TabularPredictor(label = target,
                              problem_type = 'binary',
                              eval_metric = 'roc_auc')

        clf = AutoGluonPredictor(tp,
                                 time_limit = timeout,
                                 hyperparameters = params,
                                 presets = 'best_quality')

        '''
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        itrain,itest = next(inner_cv.split(np.zeros(len(X)),y))
        clf.fit(X.iloc[itrain], y.iloc[itrain])
        td_test = TabularDataset(Xy.iloc[itest])
        lb_df = clf.leaderboard(td_test, silent=True)
        fi_df = clf.feature_importance(data=td_test, silent=True)
        '''
        
        clf.fit(X, y)
        fs_dict = clf.fit_summary(show_plot=False, verbosity=1)    
        roc_auc = fs_dict['model_performance'][fs_dict['model_best']]
    
        '''
        lb_df = clf.leaderboard(td_train, silent=True)    
        roc_auc = lb_df.loc[lb_df['model']==clf.get_model_best(),'score_val'].values[0]
        logger.info(lb_df)
        
        '''
        
        td_train = TabularDataset(Xy)    
        fi_df = clf.feature_importance(data=td_train, silent=True)
        logger.info(roc_auc)
        logger.info(fi_df)
    
        #best model by validation score that can infer.
        clf.refit_full(model='best', set_best_to_refit_full=True)
        clf.delete_models(models_to_keep='best', dry_run=False)
        clf.save_space()
        
        return clf, roc_auc

