# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:58:39 2017

@author: 凯风
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
from datetime import datetime
from sklearn.linear_model import ElasticNet,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, train_test_split
import argparse

class StackingAverageModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_fold=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_fold = n_fold
    
    def fit(self,X,y,y_test):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_fold,shuffle=True)
        out_of_fold_predictions = np.zeros((X.shape[0],len(self.base_models)))
        for i ,clf in enumerate(self.base_models):
            for train_index,holdout_index in kfold.split(X,y):
                instance = clone(clf)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],y[train_index])
                y_pred = instance.predict(X[holdout_index])
                mae = np.sum([abs(y_test[i]-y_pred[i]) for i in range(len(y_test))]) / len(y_test)
                print ('{model} model result : {result} '.format(model = 'basemodel-%s' % i ,result = mae))
                out_of_fold_predictions[holdout_index,i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions,y)
        return self
    
    def predict(self,X):
        meta_features = np.column_stack([
                np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                for base_models in self.base_models_])
        predictData = self.meta_model_.predict(meta_features)
        return predictData


class EnsemblingStacked(object):
    def __init__(self,train,properties,submission):
        self.LRModel = LinearRegression(n_jobs=-1)
        self.LRModelColumns = []
        self.LRModel.fit(self.get_features(train.drop(['parcelid','logerror'],axis=1)),train['logerror'])
        y = train['logerror'].values
        X = train.drop(['parcelid','logerror','transactiondate'],axis=1).get_values()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        del X,y;gc.collect()
            # learning_rate =0.01
            # max_depth=4
            # min_child_weight=7
            # gamma=0
            # subsample=0.65
            # colsample_bytree=0.9
            # scale_pos_weight = 1
    def xgboostModel(self):
        y_mean = np.mean(self.y_train)
        xgb_params_1 = {
                        'eta': 0.01,
                        'max_depth': 4,
                        'subsample': 0.65,
                        'eval_metric': 'mae',
                        'min_child_weight': 7,
                        'gamma':0,
                        'seed':1314,
                        'colsample_bytree':0.9,
                        'base_score': y_mean,
                        }
        xgb_params_2 = {
                        'eta': 0.037,
                        'max_depth': 5,
                        'subsample': 0.80,
                        'eval_metric': 'mae',
                        'lambda':0.8,
                        'seed':123456,
                        'alpha':0.4,
                        'base_score': y_mean,
                        }
        dtrain = xgb.DMatrix(self.X_train, self.y_train)
        dtest = xgb.DMatrix(self.X_test)
        model1 = xgb.train(dict(xgb_params_1, silent=1), dtrain, num_boost_round=350)
        model2 = xgb.train(dict(xgb_params_2, silent=1), dtrain, num_boost_round=250)
        xgb_pred1 = model1.predict(dtest)
        xgb_pred2 = model2.predict(dtest)
        xgb_pred = (xgb_pred1 + xgb_pred2) / 2
        print ('xgboost model result : {result} '.format(result = self.MAE(xgb_pred)))
        del dtrain,dtest,xgb_params_1,xgb_params_2,xgb_pred ; gc.collect()
        
        # 预测
        dpred = xgb.DMatrix(properties.drop(['parcelid','transactiondate'],axis=1).get_values())
        returnData = (model1.predict(dpred) + model2.predict(dpred)) / 2
        return returnData
    
    def lgbModel(self):
        params = {'max_bin':10,
                  'learning_rate':0.002,
                  'boosting_type':'gbdt',
                  'objective':'regression',
                  'metric':'l1',
                  'sub_feature':0.5,
                  'bagging_fraction':0.7,
                  'bagging_freq':20,
                  'num_leaves':60,
                  'min_data':500,
                  'min_hessian':0.05,
                  'verbose':0,
                  'num_threads':1
                  }
        d_train = lgb.Dataset(self.X_train,label=self.y_train)
        clf = lgb.train(params, d_train, num_boost_round=500)
        lgb_pred = clf.predict(self.X_test)
        print ('lightGBM model result : {result} '.format(result = self.MAE(lgb_pred)))
        del d_train,lgb_pred;gc.collect()
        
        # 预测
        returnData = clf.predict(properties.drop(['parcelid','transactiondate'],axis=1))
        return returnData


    def stackModel(self):
        # basemodel
        lasso = make_pipeline(RobustScaler(), Lasso(alpha =1, random_state=77))
        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.9, l1_ratio=.5, random_state=3))
        rfc = RandomForestRegressor(n_estimators=50,random_state=14,n_jobs=-1)
        etr = ExtraTreesRegressor(n_estimators=40,n_jobs=-1,random_state=22)
        rigde = make_pipeline(RobustScaler(), Ridge(alpha =0.7,random_state=33))
        GBoost = GradientBoostingRegressor(n_estimators=150, loss='huber', random_state =5)
        
        # 调用实例
        models = StackingAverageModels(base_models = (ENet, GBoost, rigde , etr ,rfc), meta_model = lasso)
        models.fit(self.X_train, self.y_train , self.y_test)
        stack_pred = models.predict(self.X_test)
        print ('stack model result : {result} '.format(result = self.MAE(stack_pred)))
        
        # 预测
        returnData = models.predict(properties.drop(['parcelid','transactiondate'],axis=1).get_values())
        return returnData

    def MAE(self,ypred):
        return np.sum([abs(self.y_test[i]-ypred[i]) for i in range(len(self.y_test))]) / len(self.y_test)
    
    def get_features(self,df):
        df["transactiondate"] = pd.to_datetime(df["transactiondate"])
        df["transactiondate_year"] = df["transactiondate"].dt.year
        df["transactiondate_month"] = df["transactiondate"].dt.month
        df['transactiondate'] = df['transactiondate'].dt.quarter
        if len(self.LRModelColumns) == 0:
            self.LRModelColumns = df.columns
        else:
            df = df.reindex(columns = self.LRModelColumns)
        return df
    
    def conbine(self):
        # 参数准备
        lgbWeight = (1 - args.XGB_WEIGHT - args.BASELINE_WEIGHT - args.STACK_WEIGHT) / (1 - args.OLS_WEIGHT)
        xgbWeight = args.XGB_WEIGHT / (1 - args.OLS_WEIGHT)
        stackWeight = args.STACK_WEIGHT / (1 - args.OLS_WEIGHT) 
        baselineWeight =  args.BASELINE_WEIGHT / (1 - args.OLS_WEIGHT)
        
        # 除OLS外的预测结果的结合
        basePred = xgbWeight*self.xgboostModel() + baselineWeight*args.BASELINE_PRED + lgbWeight*self.lgbModel() + stackWeight * self.stackModel()
        
        # 准备提交的数据
        test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
        test_columns = ['201610','201611','201612','201710','201711','201712']
        properties.drop(['parcelid'],axis=1,inplace=True)
        
        for i in range(len(test_dates)): 
            properties['transactiondate'] = test_dates[i]
            pred = args.OLS_WEIGHT * self.LRModel.predict(self.get_features(properties)) + (1 - args.OLS_WEIGHT) * basePred
            submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
            print('predict...', i)
        
        # 保存数据
        submission.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)


if __name__ == '__main__':
    # 定义命令参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--datafile', default='D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv', help='the path to the feature data file')
    parser.add_argument('-a1', '--XGB_WEIGHT', default=0.5840, help='xgboost weight')
    parser.add_argument('-a2', '--BASELINE_WEIGHT', default=0.0056, help='baseline weight')
    parser.add_argument('-a3', '--BASELINE_PRED', default=0.0115, help='baseline predict')
    parser.add_argument('-a4', '--OLS_WEIGHT', default=0.050, help='ols weight')
    parser.add_argument('-a5', '--STACK_WEIGHT', default=0.2052, help='stacking model weight')
    args = parser.parse_args()
    
    #随机器
    np.random.seed(63)
    random.seed(36)

    # 确保输出的参数合法
    try:
        args.XGB_WEIGHT = float(args.XGB_WEIGHT)
        args.BASELINE_WEIGHT = float(args.BASELINE_WEIGHT)
        args.BASELINE_PRED = float(args.BASELINE_PRED)
        args.OLS_WEIGHT = float(args.OLS_WEIGHT)
        args.STACK_WEIGHT = float(args.STACK_WEIGHT)
    except:
        raise('Unrecognized args')
    
    # 读取数据
    train = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv", parse_dates=["transactiondate"])
    properties = pd.read_csv(args.datafile)
    submission = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/sample_submission.csv")
    
    # 处理下数据
    train = pd.merge(train, properties, how='left', on='parcelid')
    properties['transactiondate'] = '2016-07-29'
    train = train[ train.logerror > -0.4 ]
    train = train[ train.logerror < 0.419 ]
    
    
    ES = EnsemblingStacked(train,properties,submission)
    ES.conbine()
