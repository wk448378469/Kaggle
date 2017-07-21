# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:08:01 2017

@author: 凯风
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor   # 可以使用sklearn的CV
from sklearn import cross_validation, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import gc

rcParams['figure.figsize'] = 12, 4
train = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv")
properties = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')
for c in properties.columns:
    # 填补缺失值为-1，xgboost推荐的
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        # 如果是字符串，则LabelEndoder
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))
        
train_df = train.merge(properties, how='left', on='parcelid')
train_df=train_df[ train_df.logerror > -0.4 ]       # (train_df.logerror < -0.4).sum() 大约700个样本
train_df=train_df[ train_df.logerror < 0.419 ]

x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)

train = x_train
train['logerror'] = y_train

target = 'logerror'


del train_df; gc.collect()
del properties; gc.collect()
del x_train; gc.collect()
del y_train; gc.collect()

def modelFit(alg, dtrain, predictors, useTrainCV = True, cvFolds = 5, earlyStopRound = 50):
    if useTrainCV:
        # 获取模型参数
        xgb_param = alg.get_xgb_params()
        # 转换xgboost需要的数据
        xgtrain = xgb.DMatrix(dtrain[predictors].values,label=dtrain[target].values)
        # 交叉验证
        cvresult = xgb.cv(xgb_param ,xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cvFolds, metrics='mae', early_stopping_rounds=earlyStopRound)
        
        # 打印最合适的评估器数量
        print ('\nNumber of estimators: %s' % cvresult.shape[0])
        # 设置有多少个基评估器~这个也算是最重要的参数了
        alg.set_params(n_estimators = cvresult.shape[0])
    
    # 训练
    alg.fit(dtrain[predictors],dtrain[target],eval_metric='mae')
    
    # 预测
    dtrainPredictions = alg.predict(dtrain[predictors])

    # 打印结果
    print ('\nModel report')
    MAE = metrics.mean_absolute_error(dtrain[target].values,dtrainPredictions)
    print ('MAE: %f ' % MAE)
    R2SCORE = metrics.r2_score(dtrain[target].values,dtrainPredictions)
    print ('R2 score: %f ' % R2SCORE)
    
    fig,ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(alg,max_num_features=30,height=0.8,ax=ax)
    plt.show()

    return MAE,R2SCORE

# 获取每个特征的名称
predictors = [x for x in train.columns if x not in [target]]

# test，主要是确定基评估器的数量
xgb1 = XGBRegressor(
            learning_rate =0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'reg:linear',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
xgb1_mae,xgb1_r2 = modelFit(xgb1, train, predictors)      # n_estimators = 84

# part1 调整基于树的一些参数：
    # max_depth          默认是5，测试在3-10之间吧
    # min_child_weight   默认是1
    # gamma              默认是0
    # subsample          默认是0.8
    # colsample_bytree   默认是0.8
    # scale_pos_weight   默认是1

param_test1 = {
 'max_depth':list(range(3,10,2)),
 'min_child_weight':list(range(1,6,2)),
}
gsearch1 = GridSearchCV(estimator = XGBRegressor(
                                        learning_rate =0.1, n_estimators=84, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test1, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=4, iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_
gsearch1.best_params_      # max_depth = 5  min_child_weight = 5
gsearch1.best_score_


# 刚刚是较大步数的去找~，找到了就缩小范围下
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6,7,9]
}
gsearch2 = GridSearchCV(estimator = XGBRegressor(
                                        learning_rate =0.1, n_estimators=84, max_depth=5,
                                        min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test2, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=4, iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_
gsearch2.best_params_      #变成了 4 7
gsearch2.best_score_

# gamma~
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBRegressor(
                                        learning_rate =0.1, n_estimators=84, max_depth=4,
                                        min_child_weight=7, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test3, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=4, iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.best_params_        # 默认的0 
gsearch3.best_score_


# 调整了几个参数之后，需要重新确定一下基评估器的数量~
xgb2 = XGBRegressor(
            learning_rate =0.1,
            n_estimators=1000,
            max_depth=4,
            min_child_weight=7,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'reg:linear',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
xgb2_mae,xgb2_r2 = modelFit(xgb2, train, predictors)  # n_estimators变成了88


# 获得评估器的数量之后，确定下两个参数subsample、colsample_bytree
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBRegressor(
                                        learning_rate =0.1, n_estimators=88, max_depth=4,
                                        min_child_weight=7, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test4, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=4, iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.best_params_              # colsample_bytree=0.7    subsample=0.9
gsearch4.best_score_


# 提升一下精度，以0.05为单位~
param_test5 = {
 'subsample':[i/100.0 for i in range(65,80,5)],
 'colsample_bytree':[i/100.0 for i in range(85,100,5)]
}
gsearch5 = GridSearchCV(estimator = XGBRegressor(
                                        learning_rate =0.1, n_estimators=88, max_depth=4,
                                        min_child_weight=7, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                        objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test5, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=6, iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.best_params_                  # colsample_bytree=0.9    subsample=0.65
gsearch5.best_score_                                 # 这个有点问题嗒~                     

# 看看正则化项的两个参数reg_lambda  默认是1 、reg_alpha 默认是0 
param_test6 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
        'reg_lambda':[10,1,0.1,0.01,1e-4]
}
gsearch6 = GridSearchCV(estimator = XGBRegressor(
                                        learning_rate =0.1, n_estimators=88, max_depth=4,
                                        min_child_weight=7, gamma=0, subsample=0.65, colsample_bytree=0.9,
                                        objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test6, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=6, iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.best_params_          # reg_alpha=1     reg_lambda=0.0001
gsearch6.best_score_


# 继续缩小精度~~~~
param_test7 = {
        'reg_alpha':[0,0.01,0.1,1,1.5],
        'reg_lambda':[1e-5,1e-4,1e-3,1e-2]
}
gsearch7 = GridSearchCV(estimator = XGBRegressor(
                                        learning_rate =0.1, n_estimators=88, max_depth=4,
                                        min_child_weight=7, gamma=0, subsample=0.65, colsample_bytree=0.9,
                                        reg_alpha=1,reg_lambda=0.0001,
                                        objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test7, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=6, iid=False, cv=5)
gsearch7.fit(train[predictors],train[target])
gsearch7.best_params_                   # reg_alpha=1     reg_lambda=1e-05  我觉得可能默认的0是最合适的~
gsearch7.best_score_


# 降低学习速度，然后增加一些基评估器
xgb3 = XGBRegressor(
            learning_rate =0.01,
            n_estimators=5000,
            max_depth=4,
            min_child_weight=7,
            gamma=0,
            subsample=0.65,
            colsample_bytree=0.9,
            objective= 'reg:linear',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
xgb3_mae,xgb3_r2 = modelFit(xgb3, train, predictors)  # n_estimators变成了1330

####汇总一下：
    # n_estimators=1330
    # learning_rate =0.01
    # max_depth=4
    # min_child_weight=7
    # gamma=0
    # subsample=0.65
    # colsample_bytree=0.9
    # scale_pos_weight = 1