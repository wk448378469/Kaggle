# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:58:39 2017

@author: 凯风
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
from datetime import datetime

# 读取数据
print( "\nReading data ...")
prop = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv')
aa = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')
prop['parcelid'] = aa['parcelid']
del aa ; gc.collect()
train = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv")

### lightGBM
print( "\nLightGBM working ..." )
for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:
        # 降低数据在内存的占比
        prop[c] = prop[c].astype(np.float32)

#删除掉没用的一个
prop.drop([prop.columns[0]],axis=1,inplace=True)

# 合并数据，train中的parcelid是连接键
df_train = train.merge(prop, how='left', on='parcelid')

# 删掉没用的特征
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = df_train['logerror'].values

# 特征名称保存
train_columns = x_train.columns

# 删除数据，不占内存
del df_train; gc.collect()

# 将数据转换成numpy的array
# 将数据转换成lightGBM的数据格式
x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)

# lightGBM的模型超参数
params = {
        'max_bin':10,
        'learning_rate':0.0021,
        'boosting_type':'gbdt',
        'objective':'regression',
        'metric':'l1',
        'sub_feature':0.5,
        'bagging_fraction':0.85,
        'bagging_freq':40,
        'num_leaves':512,
        'min_data':500,
        'min_hessian':0.05,
        'verbose':0,
        'num_threads':1
        }
# lightGBM train
num_boost_round = 430
clf = lgb.train(params, d_train, num_boost_round=num_boost_round)

# 清内存
del d_train; gc.collect()
del x_train; gc.collect()

sample = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/sample_submission.csv')
# 获取需要预测的样本编号
sample['parcelid'] = sample['ParcelId']
# 合并数据
df_test = sample.merge(prop, on='parcelid', how='left')

# 清内存
del sample, prop; gc.collect()
# 预测数据的生成
x_test = df_test[train_columns]
# 清内存
del df_test; gc.collect()

# 预测
lgb_pred = clf.predict(x_test)

# 清内存
del x_test; gc.collect()



### xgboost
print( "\nReading data again...")
properties = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv')
properties.drop([properties.columns[0]],axis=1,inplace=True)
aa = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')
properties['parcelid'] = aa['parcelid']
del aa ; gc.collect()

# 合并数据
train_df = train.merge(properties, how='left', on='parcelid')
# 删掉没用的特征
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)

# 丢弃掉训练数据的一些异常值
train_df=train_df[ train_df.logerror > -0.4 ]       # (train_df.logerror < -0.4).sum() 大约700个样本
train_df=train_df[ train_df.logerror < 0.419 ]

# 准备xgboost的数据
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
    
# xgboost的超参数
'''
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'base_score': y_mean,
    'silent': 1
}'''
xgb_params = {
    'eta': 0.01,
    'max_depth': 4,
    'min_child_weight':7,
    'subsample': 0.65,
    'colsample_bytree':0.9,
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 0
}

# xgboost 训练
num_boost_rounds = 242
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)

# xgboost 预测
xgb_pred = model.predict(dtest)

# 清内存
del train_df; gc.collect()
del x_train; gc.collect()
del x_test; gc.collect()
del properties; gc.collect()
del dtest; gc.collect()
del dtrain; gc.collect()


### LR
# 随机器
np.random.seed(17)
random.seed(17)

train = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv")
aa = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')
properties['parcelid'] = aa['parcelid']
del aa ; gc.collect()
submission = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/sample_submission.csv")

def get_features(df):
    # 根据日期生成新的特征
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

# 合并数据
train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')

properties = []

# exc:字符串特征，col:数值型特征
exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

# 调用函数生成新的特征
train = get_features(train[col])
test['transactiondate'] = '2016-07-29'   # train['transactiondate'].value_counts()[:1]  训练集该日期的数据最多
test = get_features(test[col])

# LR 训练预测~
reg = LinearRegression(n_jobs=-1)
reg.fit(train, y)
print(MAE(y, reg.predict(train)))   # LR在训练集上的表现
train = []
y = []

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']

# 不同结果的比重
XGB_WEIGHT = 0.6266
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0550

# 基于训练数据的基准平均值
BASELINE_PRED = 0.0115

# 结合数据
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*lgb_pred

for i in range(len(test_dates)): # 迭代全部需要预测的日期
    # 将测试集中的transactiondate转换成需要预测的日期
    test['transactiondate'] = test_dates[i]
    # 根据原有结合的pred0(lgb和xgb结合的) 和 LR结合(主要拟合日期的数据分布)
    pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0
    # 保存数据
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

# 保存数据
submission.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( "\nFinished ..." )