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
import xgboost as xgb
from dataset import Dataset

from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf

print ('Begin...')

SEED = 0
ntrain = 188318         # 训练集样本数
ntest = 125546          # 需要预测的样本数
NFOLDS = 5              # 交叉验证的折数
kf = KFold(ntrain,n_folds=5,shuffle=True,random_state=SEED) 
SUBMISSION_FILE = 'D:/mygit/Kaggle/Allstate_Claims_Severity/submission.csv'

# 用来保存不同模型预测的结果，作为stack的训练数据
newTrain = []
newTest = []

params = {

        'xgb1' : {
                    'features': ['numeric_boxcox','categorical_dummy'],
                    'convert_target':convert_target.norm,
                    'needScale':False,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 7,
                                                'eta': 0.1,
                                                'colsample_bytree': 0.5,
                                                'subsample': 0.95,
                                                'min_child_weight': 5,
                                                'n_iter': 400,
                                                'colsample_bytree': 0.5,})
                    },
                    
        'xgb2' : {
                    'features': ['numeric','categorical_counts','numeric_rank_norm'],
                    'convert_target':convert_target.norm,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 7,
                                                'eta': 0.03,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'min_child_weight': 2,
                                                'gamma': 0.2,
                                                'n_iter':2000,})
                    },
        
        
        'xgb3' : {
                    'features': ['numeric','categorical_dummy','cluster_rbf_75'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 12,
                                                'eta': 0.01,
                                                'colsample_bytree': 0.5,
                                                'subsample': 0.8,
                                                'gamma': 1,
                                                'alpha': 1,
                                                'n_iter':3000,})
                    },        

        'xgb4' : {
                    'features': ['numeric_combinations','categorical_dummy','cluster_rbf_25'],
                    'convert_target':convert_target.powed,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 7,
                                                'eta': 0.02,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'min_child_weight': 2,
                                                'n_iter':3000,})
                    },    

        'xgb5' : {
                    'features': ['numeric_edges','categorical_counts','cluster_rbf_50'],
                    'convert_target':convert_target.powed_ofs,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 7,
                                                'eta': 0.03,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'min_child_weight': 4,})
                    },    

        'xgb6' : {
                    'features': ['numeric','categorical_dummy','cluster_rbf_200'],
                    'convert_target':None,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 8,
                                                'eta': 0.04,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'alpha': 0.9,})
                    },    

        'xgb7' : {
                    'features': ['numeric_combinations','categorical_counts','cluster_rbf_100'],
                    'convert_target':None,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 8,
                                                'eta': 0.01,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'alpha': 0.9,
                                                'lambda': 2.1})
                    },    

        'xgb8' : {
                    'features': ['numeric_boxcox','numeric_scaled','numeric_unskew','categorical_counts','cluster_rbf_50'],
                    'convert_target':None,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 8,
                                                'eta': 0.01,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'alpha': 0.9,
                                                'lambda': 2.1})
                    },

        'xgb9' : {
                    'features': ['numeric','categorical_dummy','cluster_rbf_50','numeric_rank_norm','numeric_edges'],
                    'convert_target':None,
                    'needScale':True,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 10,
                                                'eta': 0.01,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'alpha': 0.9,
                                                'lambda': 2.1})
                    },

        'xgb10' : {
                    'features': ['svd'],
                    'convert_target':None,
                    'needScale':False,
                    'model': model.XgbWrapper(
                                        seed = SEED,
                                        params={
                                                'max_depth': 10,
                                                'eta': 0.01,
                                                'colsample_bytree': 0.4,
                                                'subsample': 0.95,
                                                'alpha': 0.9,
                                                'lambda': 2.1})
                    },

        'nn1'   :  {
                    'features':['numeric_scaled','categorical_dummy'],
                    'convert_target':None,
                    'needScale':True,
                    'model':model.TensorflowWrapper(
                            n_step=1000,
                            input_size=944,
                            learn_rate=0.7,
                            activation_function=tf.nn.relu)
                    },

        'nn2'   :  {
                    'features':['numeric_scaled','categorical_counts','cluster_rbf_200'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':True,
                    'model':model.TensorflowWrapper(
                            n_step=2500,
                            input_size=330,
                            learn_rate=0.4,
                            activation_function=tf.nn.tanh)
                    },
        
        'nn3'   :  {
                    'features':['numeric_scaled','categorical_dummy'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':True,
                    'model':model.TensorflowWrapper(
                            n_step=2000,
                            input_size=944,
                            learn_rate=0.4,
                            activation_function=tf.nn.sigmoid)
                    },
        
        'nn4'   :  {
                    'features':['numeric_scaled','cluster_rbf_75','numeric_unskew', 'cluster_rbf_200'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':True,
                    'model':model.TensorflowWrapper(
                            n_step=2000,
                            input_size=301,
                            learn_rate=0.5,
                            activation_function=tf.nn.elu)
                    },
        
        'nn5'   :  {
                    'features':['svd','cluster_rbf_25'],
                    'convert_target':convert_target.log_ofs,
                    'needScale':False,
                    'model':model.TensorflowWrapper(
                            n_step=2000,
                            input_size=525,
                            learn_rate=0.6,
                            activation_function=tf.nn.relu)
                    },

        'nn6'   :  {
                    'features':['cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75', 'cluster_rbf_100','cluster_rbf_200'],
                    'convert_target':None,
                    'needScale':True,
                    'model':model.TensorflowWrapper(
                            n_step=1500,
                            input_size=450,
                            learn_rate=0.4,
                            activation_function=tf.nn.relu6)
                    },
}


def get_oof(clf,x_train,y_train,x_test):
    
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_index,test_index) in enumerate(kf): # 把刚刚交叉的数据分别取出来
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.train(x_tr,y_tr)        # 用4份来训练
        oof_train[test_index] = clf.predict(x_te)       # 用一份来看预测结果
        oof_test_skf[i,:] = clf.predict(x_test)         # 把结果存到oof_test_skf中
        
    oof_test[:] = oof_test_skf.mean(axis = 0)           # 所有结果的平均~
    
    return oof_train.reshape(-1,1) , oof_test.reshape(-1,1)


with tqdm(total=len(params),desc='training',unit='cols') as pbar:

    for key in params:
        features = params[key]['features']
        if params[key]['convert_target'] != None:
            convertTarget = params[key]['convert_target']()
        model = params[key]['model']
        needScale = params[key]['needScale']
        
        print ('    Obtain %s feature data...' % key)
        # 获取训练数据
        train_data = []
        test_data = []
        for feature in features:
            train_data.append(sp.csc_matrix(Dataset.load_part('train',feature)))
            test_data.append(sp.csc_matrix(Dataset.load_part('test',feature)))
            
        x_train = sp.hstack(train_data)
        x_test = sp.hstack(test_data)
        y_train = Dataset.load_part('train','loss').reshape((-1,1))

        print ('    Convert variables...')
        # 判断是否需要处理目标变量
        if params[key]['convert_target'] != None:
            y_train = convertTarget.forwardConversion(y = y_train)
            
        # 判断是否需要标准化数据
        if needScale == True:
            ss = StandardScaler(with_mean=False)
            x_train = ss.fit_transform(x_train)
            x_test = ss.transform(x_test)
        
        print ('    %s Model...' % key)
        # 训练模型(stacking 的第一层)
        oof_train, oof_test = get_oof(model,x_train,y_train,x_test)

        # 还要换回来~~~
        if convertTarget != None:
            oof_test = convertTarget.reverseConversion(y = oof_test)
            oof_train = convertTarget.reverseConversion(y = oof_train)

        print ('    Merge data...')
        # 将数据添加到合并
        newTrain.append(oof_train)
        newTest.append(oof_test)
        
        del oof_train,oof_test,x_train,x_test,y_train
        
        pbar.update(1)

# 合并所有的数据    
newTrain = sp.csr_matrix(np.hstack(newTrain))
newTest = sp.csr_matrix(np.hstack(newTest))

# 保存一下中间数据，方便日后翻旧账
Dataset.save_part_feature('new',list(params.keys()))
Dataset(new=newTrain).save('train')
Dataset(new=newTest).save('test')

# 转成xgboost的数据形式
y_train = Dataset.load_part('train','loss').reshape((-1,1))
newTrain = xgb.DMatrix(newTrain, label=np.log(y_train))
newTest = xgb.DMatrix(newTest)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

res = xgb.cv(xgb_params, newTrain, num_boost_round=2000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True, feval=xg_eval_mae, maximize=False)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, newTrain, best_nrounds)

submission = pd.DataFrame()
submission['id'] = pd.read_csv('D:/mygit/Kaggle/Allstate_Claims_Severity/test.csv')['id']
submission['loss'] = np.exp(gbdt.predict(newTest))
submission.to_csv(SUBMISSION_FILE,index=None)

