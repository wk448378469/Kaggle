# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:58:41 2017

@author: 凯风
"""

import pandas as pd
from sklearn.metrics import log_loss
import scipy.sparse as sp
import xgboost as xgb


def loadData():
    print ('loading original data...')
    for name in ['training','test']:
        data1 = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/%s_variants' % name)
        data2 = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/%s_text' % name , sep='\|\|', engine='python', header=None, skiprows=1, names=["ID","Text"])
        if name == 'training':
            train = pd.merge(data1,data2,how='right',on='ID')
        if name == 'test':
            test = pd.merge(data1,data2,how='right',on='ID')
    return train,test

def preprocessOne(data):
    data.loc[:,'Text_count'] = data['Text'].apply(lambda x:len(x.split()))
    return data

if __name__ == '__preprocess_basic__':
    # 读取数据
    train ,test = loadData()
    target = train['Class']
    
    # 丢掉没用的数据
    train.drop(['ID','Class'] ,axis=1 ,inplace=True)
    test.drop(['ID'] ,axis=1 ,inplace=True)    
    
    # 获取文本数量的新特征
    train = preprocessOne(train)
    test = preprocessOne(test)