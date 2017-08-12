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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold as Kf
import argparse
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error  


class StackingAverageModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_fold=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_fold = n_fold
    
    def fit(self,X,y):
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
    def __init__(self,train,properties,submission,random_state=2017):
        self.random_state = random_state
        self.LRModel = LinearRegression(n_jobs=-1)
        self.LRModelColumns = []
        self.LRModel.fit(self.get_features(train.drop(['parcelid','logerror'],axis=1)),train['logerror'])
        self.X = train.drop(['parcelid','logerror','transactiondate'],axis=1)
        self.y = train['logerror']

    def xgboostModel(self):
        def evalerror(preds, dtrain):  
            labels = dtrain.get_label()  
            return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))  
        
        def runXGB(trainX, trainY ,testX, index, RANDOM_STATE):  
            train_index, test_index = index  
            X_train, X_val = trainX.iloc[train_index], trainX.iloc[test_index]  
            y_train, y_val = trainY.iloc[train_index], trainY.iloc[test_index]  
          
            xgtrain = xgb.DMatrix(X_train, label=y_train)  
            xgval = xgb.DMatrix(X_val, label=y_val)  
            xgtest = xgb.DMatrix(testX.drop(['parcelid','transactiondate'],axis=1))  
            X_val = xgb.DMatrix(X_val)  
          
            params = {  
                        'eta': 0.1,  
                        'silent': 1,  
                        'verbose_eval': True,
                        'objective': 'reg:linear',
                        'seed':RANDOM_STATE,
                        'alpha': 1.4412900129552941,
                        'colsample_bytree': 0.54608931927832327,
                        'gamma': 0.13040896563538462,
                        'max_depth': 13,
                        'min_child_weight': 10.278039912781676,
                        'subsample': 0.99808084817020015
                      }  
            rounds = 3000  
          
            watchlist = [(xgtrain, 'train'), (xgval, 'eval')]  
            model = xgb.train(params, xgtrain, rounds, watchlist, feval=evalerror, early_stopping_rounds=50)  
          
            cv_score = mean_absolute_error(model.predict(X_val) ,y_val)  
            predict = model.predict(xgtest)
            
            print ("iteration = %d"%(model.best_iteration))
            del X_val,X_train,y_train,y_val,xgtrain,xgval,xgtest,model;gc.collect()
            return predict, cv_score  
        
        nfolds = 10  
        kf = Kf(self.X.shape[0], n_folds = nfolds, shuffle = True, random_state = self.random_state) 
        predicts = np.zeros(properties.shape[0])  
        for i, index in enumerate(kf):  
            print('Xgboost fold %d' % (i + 1))  
            predict, cv_score = runXGB(self.X, self.y, properties, index, self.random_state)  
            print (cv_score)  
            predicts += predict  
        predicts = predicts / nfolds
        return predicts
    
    def lgbModel(self):
        def runLGB(trainX, trainY ,testX, index, RANDOM_STATE):  
            train_index, test_index = index  
            X_train, X_val = trainX.iloc[train_index], trainX.iloc[test_index]  
            y_train, y_val = trainY.iloc[train_index], trainY.iloc[test_index]  
          
            xgtrain = lgb.Dataset(X_train,label=y_train)
            xgval = lgb.Dataset(X_val, label=y_val)  
          
            params = {  
                      'objective':'regression',
                      'metric':'l1',
                      'seed': RANDOM_STATE,
                      'feature_fraction': 0.30892969365095979,
                      'lambda_l1': 0.44204748407615901,
                      'lambda_l2': 1.4814177193018863,
                      'max_bin': 162,
                      'max_depth': 65,
                      'min_data_in_leaf': 105,
                      'min_sum_hessian_in_leaf': 0.00070310444034356442,
                      'num_leaves': 29,
                      }  
            rounds = 3000  
          
            model = lgb.train(params, xgtrain, rounds, valid_sets=xgval,  early_stopping_rounds=50)  
            cv_score = mean_absolute_error(model.predict(X_val) ,y_val)
            predict = model.predict(testX)
            
            print ("iteration = %d"%(model.best_iteration))
            del model,X_train,X_val,y_train,y_val,xgtrain,xgval ; gc.collect()
            return predict, cv_score  
        
        nfolds = 10
        kf = Kf(self.X.shape[0], n_folds = nfolds, shuffle = True, random_state = self.random_state) 
        predicts = np.zeros(properties.shape[0])
        testXData = properties.drop(['parcelid','transactiondate'],axis=1)

        for i, index in enumerate(kf):  
            print('lightGBM fold %d' % (i + 1))  
            predict, cv_score = runLGB(self.X, self.y, testXData, index, self.random_state)  
            print (cv_score)  
            predicts += predict  
        predicts = predicts / nfolds
        return predicts


    def stackModel(self):
        # basemodel
        lasso = make_pipeline(StandardScaler(), Lasso(alpha =0.0002, random_state=self.random_state))
        ridge = make_pipeline(RobustScaler(), Ridge(alpha=1,random_state=self.random_state))      
        ENet = make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=1, random_state=self.random_state))      
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =self.random_state)
        MLP = make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(30, ), 
                                   activation='logistic', solver='sgd', 
                                   alpha=0.62,max_iter=1000,random_state=self.random_state))
        bayeR = make_pipeline(RobustScaler(),BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-04, 
                                              alpha_2=1e-02, lambda_1=1, lambda_2=0))
        models = StackingAverageModels(base_models = (lasso,ridge,ENet,GBoost,MLP,bayeR), meta_model = lasso)
        models.fit(self.X.get_values(),self.y.values)
        # 预测
        returnData = models.predict(properties.drop(['parcelid','transactiondate'],axis=1).get_values())
        return returnData

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
    parser.add_argument('-f', '--datafile', default='D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself2.csv', help='the path to the feature data file')
    parser.add_argument('-a1', '--XGB_WEIGHT', default=0.6700, help='xgboost weight')
    parser.add_argument('-a2', '--BASELINE_WEIGHT', default=0.0056, help='baseline weight')
    parser.add_argument('-a3', '--BASELINE_PRED', default=0.0114572, help='baseline predict')
    parser.add_argument('-a4', '--OLS_WEIGHT', default=0.050, help='ols weight')
    parser.add_argument('-a5', '--STACK_WEIGHT', default=0.1052, help='stacking model weight')
    args = parser.parse_args()
    
    #随机器
    np.random.seed(1)
    random.seed(2)

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
