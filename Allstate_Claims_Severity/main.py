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
import scipy.sparse as sp

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.svm import SVR
import tensorflow as tf

from statsmodels.regression.quantile_regression import QuantReg

from dataset import Dataset


SEED = 0
ntrain = 188318         # 训练集样本数
ntest = 125546          # 需要预测的样本数
NFOLDS = 5              # 交叉验证的折数
kf = KFold(ntrain,n_folds=5,shuffle=True,random_state=SEED) 

# 用来保存不同模型预测的结果，作为stack的训练数据
newTrain = []
newTest = []

params = {
        'xgb1' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.norm,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                        'max_depth': 7,
                                        'eta': 0.1,
                                        'colsample_bytree': 0.5,
                                        'subsample': 0.95,
                                        'min_child_weight': 5,})
                    },

        'nn1'   :  {
                    'features':['numeric','categorical_dummy','cluster_rbf_100'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':True,
                    'model':model.TensorflowWrapper(
                            n_step=1000,
                            input_size=1044,
                            learn_rate=0.1,
                            activation_function=tf.nn.relu)
                    },

        'sk-etr' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':True,
                    'model': model.SklearnWrapper(
                                        clf =ExtraTreesRegressor,
                                        seed = SEED,
                                        params={
                                        'max_features':'auto',
                                        'min_samples_split': 3,
                                        'min_samples_leaf':3,
                                        'n_estimators':15,})
                    },
                    
        'sk-rfr' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.powed,
                    'needScale':True,
                    'model': model.SklearnWrapper(
                                        clf =RandomForestRegressor,
                                        seed = SEED,
                                        params={
                                        'max_features':'auto',
                                        'min_samples_split': 3,
                                        'min_samples_leaf':2,})
                    },
                    
        'sk-gbr' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.powed_ofs,
                    'needScale':True,
                    'model': model.SklearnWrapper(
                                        clf =GradientBoostingRegressor,
                                        seed = SEED,
                                        params={
                                        'loss':'lad',
                                        'learning_rate': 0.15,
                                        'min_samples_split':3,})
                    },
                
        'sk-abr' : {
                    'features': ['numeric','categorical_encoded'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':True,
                    'model': model.SklearnWrapper(
                                        clf =AdaBoostRegressor,
                                        seed = SEED,
                                        params={
                                        'n_estimators':300,
                                        'learning_rate': 1.0 })
                    },
                    
        'sk-ridge' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.log,
                    'needScale':True,
                    'model': model.SklearnWrapper(
                                        clf =Ridge,
                                        seed = SEED,
                                        params={
                                        'alpha':1.2,})
                    },
                    
        'sk-lasso' : {
                    'features': ['numeric','categorical_counts'],
                    'convert_target':convert_target.powed,
                    'needScale':True,
                    'model': model.SklearnWrapper(
                                        clf =Lasso,
                                        seed = SEED,
                                        params={
                                        'alpha':1.1,
                                        'max_iter': 1200,})
                    },
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
        
    oof_test[:] = oof_test_skf.mean(axis = 0)           # 所有结果的平均~
    
    return oof_train.reshape(-1,1) , oof_test.reshape(-1,1)


with tqdm(total=len(params),desc='training',unit='cols') as pbar:

    for key in params:
        features = params[key]['features']
        convertTarget = params[key]['convert_target']
        model = params[key]['model']
        needScale = params[key]['needScale']
        
        # 获取训练数据
        train_data = []
        test_data = []
        for feature in features:
            train_data.append(sp.csc_matrix(Dataset.load_part('train',feature)))
            test_data.append(sp.csc_matrix(Dataset.load_part('test',feature)))
            
        x_train = sp.hstack(train_data)[:10000]
        x_text = sp.hstack(train_data)[:10000]
        y_train = Dataset.load_part('train','loss').reshape((-1,1))[:10000]

        # 判断是否需要处理目标变量
        if convert_target != None:
            y_train = convert_target.forwardConversion(y_train)
            
        # 判断是否需要标准化数据
        if needScale == True:
            ss = StandardScaler()
            x_train = ss.fit_transform(x_train)
            x_text = ss.transform(x_text)
        
        # 训练模型(stacking 的第一层)
        oof_train, oof_test = get_oof(model,x_train,y_train,x_text)

        # 还要把目标变量换回来~~~
        if convert_target != None:
            y_train = convert_target.reverseConversion(y_train)

        # 打印训练结果
        print("%s result: %f" % (key, mean_absolute_error(y_train, oof_train)))

        # 将数据添加到合并
        newTrain.append(oof_train)
        newTest.append(oof_test)
        
        del oof_train,oof_test,x_train,x_text,y_train
        
        pbar.update(1)
        
newTrain = sp.hstack(newTrain)
newTest = sp.hstack(newTest)

