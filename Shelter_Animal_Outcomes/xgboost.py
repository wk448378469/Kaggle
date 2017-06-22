# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:18:18 2017

@author: 凯风
"""

import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# 要预测的
need_pre = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/test.csv')
# 用来训练的
train = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train.csv')
target = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train_target.csv',header=None)

# 在读取数据的时候应该有一些参数设置，可以快速处理的，但是暂时还没掌握的辣么好~
need_pre.drop('Unnamed: 0',axis=1,inplace=True)
train.drop('Unnamed: 0',axis=1,inplace=True)
target.drop(0,axis=1,inplace=True)

# 切分数据集
x_train, x_test, y_train, y_test = train_test_split(train,target,test_size=0.4)

# 转换成xgb的数据形式
dneedpre = xgb.DMatrix(need_pre,missing=-9999)
dtrain = xgb.DMatrix(x_train,y_train,missing = -9999)
dtest = xgb.DMatrix(x_test,missing = -9999)

param1 = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}

param2 = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}

param3 = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}

param4 = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}

param5 = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}

num_round = 125

# 训练
bst1 = xgb.train(param1, dtrain, num_round)
bst2 = xgb.train(param2, dtrain, num_round)
bst3 = xgb.train(param3, dtrain, num_round)
bst4 = xgb.train(param4, dtrain, num_round)
bst5 = xgb.train(param5, dtrain, num_round)
# 预测
ypred = (bst1.predict(dtest) + bst2.predict(dtest) + bst3.predict(dtest) +  bst4.predict(dtest) +  bst5.predict(dtest))/5
# 查看结果
print(log_loss(y_test,ypred))   # 0.738，反正比random的好多了9 9 


# 用全数据训练
num_round = 200
dtrain = xgb.DMatrix(train,target,missing = -9999)
bst1 = xgb.train(param1, dtrain, num_round)
bst2 = xgb.train(param2, dtrain, num_round)
bst3 = xgb.train(param3, dtrain, num_round)
bst4 = xgb.train(param4, dtrain, num_round)
bst5 = xgb.train(param5, dtrain, num_round)
ypred = (bst1.predict(dneedpre) + bst2.predict(dneedpre) + bst3.predict(dneedpre) + bst4.predict(dneedpre) + bst5.predict(dneedpre))/5

'''
    data.ix[data.OutcomeType=='Return_to_owner','target'] = 0
    data.ix[data.OutcomeType=='Transfer','target'] = 1
    data.ix[data.OutcomeType=='Euthanasia','target'] = 2
    data.ix[data.OutcomeType=='Died','target'] = 3
    data.ix[data.OutcomeType=='Adoption','target'] = 4
'''

output = pd.DataFrame(ypred,columns=['Return_to_owner','Transfer','Euthanasia','Died','Adoption'])
output.columns.names = ['ID']
output.index.names = ['ID']
output.index += 1
output.to_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/model_pre/prediction_xgboost.csv')
# 自己在文件中补充上ID吧。。。。