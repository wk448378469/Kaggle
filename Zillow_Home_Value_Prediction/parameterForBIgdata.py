# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:56:29 2017

@author: 凯风
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import ElasticNet,Lasso,Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, train_test_split,cross_val_score
from sklearn.svm import SVR
import gc
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error  


train = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself2.csv')

train = pd.merge(train, properties, how='left', on='parcelid')
properties['transactiondate'] = '2016-07-29'
train = train[ train.logerror > -0.4 ]
train = train[ train.logerror < 0.419 ]

y = train['logerror']
X = train.drop(['parcelid','logerror','transactiondate'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del train,properties,X,y;gc.collect()


def MAE(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

#####  xgboost  #######

dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test)

num_rounds = 4000
random_state = 2016
num_iter = 25
init_points = 5

# 初始化参数
params = {
        'eta': 0.1,
        'silent': 1,
        'eval_metric': 'mae',
        'verbose_eval': True,
        'seed': random_state
        }

def xgb_evaluate( min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha):
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)

    cv_result = xgb.cv(params, dtrain, num_boost_round=num_rounds, nfold=5,
             seed=random_state,
             callbacks=[xgb.callback.early_stop(50)])
    
    return -cv_result['test-mae-mean'].values[-1]

xgbBO = BayesianOptimization(xgb_evaluate, {
                                            'min_child_weight': (1, 15),
                                            'colsample_bytree': (0.1, 1),
                                            'max_depth': (3, 15),
                                            'subsample': (0.5, 1),
                                            'gamma': (0, 10),
                                            'alpha': (0, 10),
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)
xgbBO.res['max']['max_params']
best_params = xgbBO.res['max']['max_params']



def logregobj(preds, dtrain):  
    labels = dtrain.get_label()  
    con = 2  
    x = preds - labels  
    grad = con * x / (np.abs(x) + con)  
    hess = con ** 2 / (np.abs(x) + con) ** 2  
    return grad, hess  
  
def evalerror(preds, dtrain):  
    labels = dtrain.get_label()  
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))  


def runXGB(train,test,index,RANDOM_STATE):  
    train_index, test_index = index  
    y = train['loss']  
    X = train.drop(['loss', 'id'], 1)  
    X_test = test.drop(['id'], 1)  
    del train,test  
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]  
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]  
  
    xgtrain = xgb.DMatrix(X_train, label=y_train)  
    xgval = xgb.DMatrix(X_val, label=y_val)  
    xgtest = xgb.DMatrix(X_test)  
    X_val = xgb.DMatrix(X_val)  
  
    params = {  
        'min_child_weight': 10,  
        'eta': 0.1,  
        'colsample_bytree': 0.7,  
        'max_depth': 12,  
        'subsample': 0.7,  
        'alpha': 1,  
        'gamma': 1,  
        'silent': 1,  
        'verbose_eval': True,  
        'seed': RANDOM_STATE  
    }  
    rounds = 3000  
  
    watchlist = [(xgtrain, 'train'), (xgval, 'eval')]  
    model = xgb.train(params, xgtrain, rounds, watchlist, obj=logregobj, feval=evalerror, early_stopping_rounds=100)  
  
    cv_score = mean_absolute_error(model.predict(X_val) ,y_val)  
    predict = model.predict(xgtest)
    
    print ("iteration = %d"%(model.best_iteration))  
    return predict, cv_score  

nfolds = 10  
RANDOM_STATE = 113  
kf = KFold(X_train.shape[0], n_folds = nfolds, shuffle = True, random_state = RANDOM_STATE) 
predicts = np.zeros(y_test.shape)  
for i, index in enumerate(kf):  
    print('Fold %d' % (i + 1))  
    predict, cv_score = runXGB(dtrain, dtest, index, RANDOM_STATE)  
    print (cv_score)  
    predicts += predict  
predicts = predicts / nfolds















#####lightgbm#######
d_train = lgb.Dataset(X_train,label=y_train)

param_test = {'max_depth': range(5,15,2),  
              'num_leaves': range(10,40,5),  
              }
estimator = LGBMRegressor(  
        num_leaves = 50, # cv调节50是最优值  
        max_depth = 13,  
        learning_rate =0.1,   
        n_estimators = 1000,   
        objective = 'regression',   
        min_child_weight = 1,   
        subsample = 0.8,  
        colsample_bytree=0.8,  
        nthread = 7,  
    )
gsearch = GridSearchCV( estimator , param_grid = param_test, scoring='mae', cv=5 )
gsearch.fit(X_train, y_train)
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

#接下来就是repeat，记得看https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-tuning.md
# 上面的链接是关于参数说明的















######other model #####
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(clf)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
score = rmsle_cv(stacked_averaged_models)
print(" Averaged base models score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
