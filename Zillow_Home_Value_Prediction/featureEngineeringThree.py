# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:19:02 2017

@author: 凯风
"""

import pandas as pd
import gc
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew,boxcox
from sklearn.cluster import MiniBatchKMeans


### 准备数据
print('loading data')
allData = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')
features = allData.columns

for c, dtype in zip(allData.columns, allData.dtypes):	
    if dtype == np.float64:
        allData[c] = allData[c].astype(np.float32)


### 填补缺失数据
allData[features[1]].fillna(0,inplace=True)
LE1 = LabelEncoder()
allData[features[1]] = LE1.fit_transform(allData[features[1]])
del LE1;gc.collect()

allData[features[2]].fillna(0,inplace=True)
LE2 = LabelEncoder()
allData[features[2]] = LE2.fit_transform(allData[features[2]])
del LE2;gc.collect()

allData[features[3]].fillna(allData[features[3]].median(),inplace=True)

allData[features[4]].fillna(allData[features[4]].mode()[0],inplace=True)

allData[features[5]].fillna(allData[features[5]].mode()[0],inplace=True)

allData[features[6]].fillna(0,inplace=True)

allData[features[7]].fillna(0,inplace=True)
LE3 = LabelEncoder()
allData[features[7]] = LE3.fit_transform(allData[features[7]])
del LE3;gc.collect()

allData[features[8]].fillna(allData[features[8]].mode()[0],inplace=True)

allData.drop([features[9]],axis=1,inplace=True)

allData[features[10]].fillna(allData[features[10]].median(),inplace=True)

allData[features[11]].fillna(allData[features[11]].mean(),inplace=True)

allData[features[12]].fillna(allData[features[12]].median(),inplace=True)

allData[features[13]].fillna(0,inplace=True)

allData[features[14]].fillna(allData[features[14]].median(),inplace=True)

allData[features[15]].fillna(allData[features[15]].median(),inplace=True)

allData[features[16]].fillna(allData[features[16]].median(),inplace=True)

allData[features[17]].fillna(allData[features[17]].mode()[0],inplace=True)
LE4 = LabelEncoder()
allData[features[17]] = LE4.fit_transform(allData[features[17]])
del LE4;gc.collect()

allData[features[18]].fillna(0,inplace=True)

allData[features[19]].fillna(0,inplace=True)

allData[features[20]].fillna(allData[features[20]].mode()[0],inplace=True)

allData[features[21]].fillna(allData[features[21]].mean(),inplace=True)

allData.drop([features[22]],axis=1,inplace=True)

allData[features[23]].fillna(allData[features[23]].mode()[0],inplace=True)
LE5 = LabelEncoder()
allData[features[23]] = LE5.fit_transform(allData[features[23]])
del LE5;gc.collect()

allData[features[24]].fillna(allData[features[24]].median(),inplace=True)

allData[features[25]].fillna(allData[features[25]].median(),inplace=True)

allData[features[26]].fillna(allData[features[26]].median(),inplace=True)

# 27——31都是和泳池相关的特征，丢失率都很高，但是丢失率却不同一....
allData[features[27]].fillna(0,inplace=True)
allData[features[28]].fillna(allData[features[28]].median(axis=0),inplace=True)
allData[features[29]].fillna(0,inplace=True)
allData[features[30]].fillna(0,inplace=True)
allData[features[31]].fillna(0,inplace=True)

allData[features[32]].fillna(allData[features[32]].mode()[0],inplace=True)
LE6 = LabelEncoder()
allData[features[32]] = LE6.fit_transform(allData[features[32]])
del LE6;gc.collect()

allData[features[33]].fillna(allData[features[33]].mode()[0],inplace=True)
LE7 = LabelEncoder()
allData[features[33]] = LE7.fit_transform(allData[features[33]])
del LE7;gc.collect()

allData.drop([features[34]],axis=1,inplace=True)

allData[features[35]].fillna(allData[features[35]].median(axis=0),inplace=True)
allData[features[35]] = np.log(allData[features[35]]).astype('float32')

allData[features[36]].fillna(0,inplace=True)

allData[features[37]].fillna(allData[features[37]].mode()[0],inplace=True)
LE8 = LabelEncoder()
allData[features[37]] = LE8.fit_transform(allData[features[37]])
del LE8;gc.collect()

allData.drop([features[38]],axis=1,inplace=True)

allData[features[39]].fillna(allData[features[39]].mode()[0],inplace=True)
LE9 = LabelEncoder()
allData[features[39]] = LE9.fit_transform(allData[features[39]])
del LE9;gc.collect()

allData[features[40]].fillna(allData[features[40]].mode()[0],inplace=True)

allData.drop([features[41]],axis=1,inplace=True)

allData[features[42]].fillna(allData[features[42]].mode()[0],inplace=True)

allData[features[43]].fillna(allData[features[43]].mode()[0],inplace=True)
LE10 = LabelEncoder()
allData[features[43]] = LE10.fit_transform(allData[features[43]])
del LE10;gc.collect()

allData[features[44]].fillna(allData[features[44]].mode()[0],inplace=True)

allData[features[45]].fillna(allData[features[45]].median(),inplace=True)

allData[features[46]].fillna(allData[features[46]].mean(),inplace=True)

allData[features[47]].fillna(allData[features[47]].mode()[0],inplace=True)
LE11 = LabelEncoder()
allData[features[47]] = LE11.fit_transform(allData[features[47]])
del LE11;gc.collect()

allData[features[48]].fillna(allData[features[48]].mode()[0],inplace=True)
LE12 = LabelEncoder()
allData[features[48]] = LE12.fit_transform(allData[features[48]])
del LE12;gc.collect()

allData[features[49]] = allData[features[49]].apply(lambda x:1 if x is True else 0)

allData[features[50]].fillna(allData[features[50]].median(),inplace=True)

allData[features[51]].fillna(allData[features[51]].median(),inplace=True)

allData[features[52]].fillna(allData[features[52]].mode()[0],inplace=True)

allData[features[53]].fillna(allData[features[53]].median(),inplace=True)

allData[features[54]].fillna(allData[features[54]].mean(),inplace=True)

allData[features[55]] = allData[features[55]].apply(lambda x:1 if x is 'Y' else 0)

allData[features[56]].fillna(allData[features[56]].mode()[0],inplace=True)

allData[features[57]].fillna(allData[features[57]].mean(),inplace=True)
allData[features[57]] = np.log(allData[features[57]]).astype('float32')
allData[features[57]].fillna(allData[features[57]].mean(),inplace=True)


### 造特征
allData['totalcnt'] = allData[features[3]] + allData[features[10]] + allData[features[11]] + allData[features[12]] + allData[features[13]] + allData[features[14]] + allData[features[15]] + allData[features[16]] + allData[features[21]] + allData[features[45]]

numeric_feats = ['totalcnt',features[3],features[10],features[11],features[12],features[13],features[14],features[15],features[16],features[21],features[26],features[28],features[35],features[45],features[46],features[47],features[50],features[51],features[53],features[54],features[57]]
for col in range(len(numeric_feats)):
    sk = skew(allData[numeric_feats[col]])
    if sk > 0.25:
        value , lam = boxcox(allData[numeric_feats[col]] + 1)
        allData[numeric_feats[col]] = value
        del value,lam;gc.collect()

kmeans = MiniBatchKMeans(20,random_state=17*20+11,n_init=5)
kmeans.fit(allData)
cluster_rbf = np.log(kmeans.transform(allData))

for col in range(cluster_rbf.shape[1]):
    featureName = 'cluster_rbf_' + str(col)
    allData[featureName] = cluster_rbf[:,col].astype('float32')

allData.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv',index=False)

