# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 18:37:44 2017

@author: 凯风
"""

import pandas as pd
from tqdm import tqdm
from datetime import datetime

result = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/sub20170804_202352.csv')
train = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv')

test_columns = ['201610','201611','201612','201710','201711','201712']
train.drop(['transactiondate'],axis=1,inplace=True)

repeatData = train.parcelid.value_counts().reindex()
repeatData = repeatData[repeatData.values > 1]

for i in range(len(repeatData)):
    delId = repeatData.index.values[i]
    trainIndex = train.ix[train['parcelid'] == delId].index.values
    train.drop(trainIndex,inplace = True)

print (result.shape)

result = pd.merge(result, train, how='left', left_on='ParcelId', right_on='parcelid')

print (result.shape)

result.drop(['parcelid'],axis=1,inplace=True)
del train

with tqdm(total=result.shape[0], desc=' Transforming ' ,unit='cols') as pbar:
    for index in range(result.shape[0]):
        if pd.notnull(result.ix[index].logerror):
            for column in test_columns:
                result.ix[index,column] = result.ix[index].logerror
        pbar.update(1)   

result.drop(['logerror'],axis=1,inplace=True)

print (result.shape)

result.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)






