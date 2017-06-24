# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:23:32 2017

@author: 凯风
"""

import pandas as pd
import xgboost as xgb

target = pd.read_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/target.csv',header=None)
target_dict = pd.read_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/target_dict.csv',header=None)
train = pd.read_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/train.csv',header=None)
test = pd.read_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/test.csv',header=None)

dneedpre = xgb.DMatrix(test)
dtrain = xgb.DMatrix(train,target)

xgb_params = {
                # xgboost的
                'objective': 'multi:softprob',
                'eta':0.4,
                'silent':0,
                'nthread':4,
                'num_class':39,
                'eval_metric':'mlogloss',
                # 树结构相关的
                'max_depth': 8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alfa':0.05,
                'subsample':0.8,
                'colsample_bytree':1,
                # 其他
                'max_delta_step':1 
                }

# 训练
gbdt = xgb.train(xgb_params, dtrain, num_boost_round=10)

# 预测
ypred = gbdt.predict(dneedpre)

output = pd.DataFrame(ypred,columns=target_dict)
output.columns.names = ['Id']
output.index.names = ['Id']
output.index += 1
output.to_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/prediction_xgboost.csv')
# 自己在文件中补充上ID吧。。。。