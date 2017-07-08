# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 09:27:17 2017

@author: 凯风
"""


import pandas as pd
import numpy as np
import scipy.sparse as sp
import xgboost as xgb

features = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/svdData.csv')
target = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/target.csv',header=None)
target[0] = target[0] - 1
ntrain = target.shape[0]

clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
clf.fit(features[:ntrain],target[0])
xgb_test = clf.predict_proba(features[ntrain:])

classes = "class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
subm = pd.DataFrame()
subm['ID'] = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/test_variants')['ID']

for index in classes:
    subm[index] = xgb_test[:,classes.index(index)]
    
subm.to_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/submission_xgb.csv', index=False)	