# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:13:18 2017

@author: 凯风
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb
import pandas as pd

# 读取数据 
trainX = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/trainX.csv',header=None) 
trainY = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/trainY.csv',header=None)
needPre = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/testX.csv',header=None)
needPreId = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/testY.csv',header=None)

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33)

# 交叉验证，获取最有参数~
param = {
         'max_depth':[4,5,6,7],
         'min_child_weight':list(range(1,8,2)),
         'gamma':[0.01,0.1,0.3,0.5]
         }
gscv = GridSearchCV(estimator=xgb.sklearn.XGBClassifier(), param_grid=param, scoring='log_loss')
gscv.fit(X_train, y_train.values.reshape(-1))
gscv.best_params_
log_loss(y_test,gscv.predict_proba(X_test)[:,1])

# 全数据训练
xgb = xgb.sklearn.XGBClassifier(max_depth=gscv.best_params_['max_depth'],
                                min_child_weight=gscv.best_params_['min_child_weight'],
                                gamma=gscv.best_params_['gamma'])
xgb.fit(trainX,trainY.values.reshape(-1))

# 预测并保存数据
needPreId['shot_made_flag'] = xgb.predict_proba(needPre)[:,1]
needPreId.columns = ['shot_id','shot_made_flag']
needPreId.to_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Prediction_Result/prediction_xgboost.csv',index=False)
