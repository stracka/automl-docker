#!/usr/bin/env python
# coding: utf-8

"""

Optimization with various strategies: 
For illustration, currently limited to tree ensemble classifiers


- Random search: randomsearch_gbdt


- Optuna: tree-structured Parzen estimator (TPE). Not limited to sklearn


- TPOT: genetic programming


- AutoGluon (Tabular)



- SMAC3: https://github.com/automl/SMAC3 bayesian optimization with random forests
- Spearmint (active) and GPyOpt (not active) use Gaussian Processes, bayesian optimization: https://sheffieldml.github.io/GPyOpt/index.html , https://github.com/HIPS/Spearmint


- AutoKeras


- AutoSklearn



https://www.analyticsvidhya.com/blog/2021/10/beginners-guide-to-automl-with-an-easy-autogluon-example/



https://explained.ai/rf-importance/



"""


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

# Optuna
import optuna

# TPOT
from tpot import TPOTClassifier

# AutoGluon
from autogluon.tabular import TabularDataset, TabularPredictor



import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

#https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
#https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html?highlight=powertransform



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

    
    '''
    from sklearn.metrics import roc_auc_score, roc_curve, auc 

    twoclass_output = clf.best_estimator_.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, twoclass_output)
    roc_auc = auc(fpr, tpr)

    #for var, imp in zip(featlist, bdt.feature_importances_):
    #    logger.info(f'{var}: {imp}')    

    for i in np.argsort(bdt.feature_importances_):
        logger.info('%20s  %6.4f'%(features[i],bdt.feature_importances_[i]))        
    '''



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
            # would have to implement manually

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

        clf_best = UGradientBoostingClassifier(
            loss= BinFlatnessLossFunction([aux], n_bins=2, uniform_label=1),
            random_state=0,
            train_features=features,
            max_depth = study.best_params['gbdt_max_depth'],
            n_estimators = study.best_params['gbdt_n_estimators'],
            max_features = study.best_params['gbdt_max_features']
        )


        
    clf_best.fit(X,y)
        
    return clf_best, study.best_value



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


class AutoGluonPredictor(TabularPredictor):
    """
    https://github.com/awslabs/autogluon/issues/1493
    Adding classes_ and predict_proba
    Wrapper for compatibility with scikit-learn pipelines and methods
    """

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


def report(clf):

    
    from sklearn.metrics import roc_auc_score, roc_curve, auc 

    plot_range = (twoclass_output.min(), twoclass_output.max())
    plt.hist(twoclass_output[y_test==0],range=plot_range,density=True,histtype='step',bins=50,label='BG')
    plt.hist(twoclass_output[y_test==1],range=plot_range,density=True,histtype='step',bins=50,label='Signal')
    plt.legend()
    plt.title('BDT output')
    plt.savefig('BDT_output.pdf',bbox_inches='tight')
    plt.clf()
        

    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.3f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('BDT_ROC.pdf',bbox_inches='tight')
    plt.clf()
    
    from joblib import dump, load
    name = 'BDT_etaPi_'+str(len(features))+'_vflat.joblib'
    dump(bdt, name)
