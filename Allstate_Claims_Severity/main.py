# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:35:32 2017

@author: 凯风
"""
import numpy as np
import pandas as pd
import convert_target
import model
from tqdm import tqdm

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


SEED = 0
ntrain = 188318
ntest = 125546
NFOLDS = 5
kf = KFold(ntrain,n_folds=5,shuffle=True,random_state=SEED) 
newTrain = {}


params = {
        'xgb1' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.norm_y,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                        'max_depth': 7,
                                        'eta': 0.1,
                                        'colsample_bytree': 0.5,
                                        'subsample': 0.95,
                                        'min_child_weight': 5,})
                    },

        'sk-etr' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.norm_y,
                    'model': model.SklearnWrapper(
                                        clf =ExtraTreesRegressor,
                                        seed = SEED,
                                        params={
                                        'max_features':['auto'],
                                        'min_samples_split': 3 })
                    }
}


def get_oof(clf,x_train,y_train,x_text):
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


with tqdm(total=len(params),desc='training',unit='cols') as pbar:

    for key in params:
        features = params[key]['features']
        convert_target = params[key]['convert_target']
        model = params[key]['model']
        
        # 获取训练数据
        train_data = []
        for feature in features:
            train_data.append(Dataset.load_part('train',feature))
        
        y_train = None
        x_train = None
        x_text = None
        # 训练模型
        oof_train, oof_test = get_oof(model)

        # 打印训练结果
        print("%s result: %f" % (key, mean_absolute_error(y_train, oof_train)))

        # 将数据添加到合并
        newTrain[key] = (oof_train,oof_test)
    
        pbar.update(1)

# 将newTrain的第一层数据进行合并
x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)


# 

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