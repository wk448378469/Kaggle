# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:35:32 2017

@author: 凯风
"""
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold, train_test_split

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.svm import SVR
import tensorflow as tf
import xgboost as xgb

from statsmodels.regression.quantile_regression import QuantReg

from dataset import Dataset
from convert_target import *

SEED = 0

kf = KFold(ntrain,n_folds=5,shuffle=True,random_state=SEED) # 交叉验证用的,分成5份

param = {
        'et_params' : {
                        'n_jobs': 16,
                        'n_estimators': 100,
                        'max_features': 0.5,
                        'max_depth': 12,
                        'min_samples_leaf': 2,
                        },

        'rf_params': {
                        'n_jobs': 16,
                        'n_estimators': 100,
                        'max_features': 0.2,
                        'max_depth': 12,
                        'min_samples_leaf': 2,
                        },

        'xgb_params':{
                        'seed': 0,
                        'colsample_bytree': 0.7,
                        'silent': 1,
                        'subsample': 0.7,
                        'learning_rate': 0.075,
                        'objective': 'reg:linear',
                        'max_depth': 4,
                        'num_parallel_tree': 1,
                        'min_child_weight': 1,
                        'eval_metric': 'rmse',
                        'nrounds': 500
                        }}

def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_index,test_index) in enumerate(kf): # 把刚刚交叉的数据分别取出来
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.train(x_tr,y_tr)        # 用4份来训练
        oof_train[test_index] = clf.predict(x_te)       # 用一份来看预测结果
        oof_test_skf[i,:] = clf.predict(x_text)         # 把结果存到oof_test_skf中
    oof_test[:] = oof_test_skf.mean(axis = 0)
    return oof_train.reshape(-1,1) , oof_test.reshape(-1,1)

xg = XgbWrapper(seed = SEED,params=xgb_params)
et = SklearnWrapper(clf =ExtraTreesRegressor,seed = SEED,params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))

x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}±{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)