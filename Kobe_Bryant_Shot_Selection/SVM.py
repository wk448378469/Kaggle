# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:35:00 2017

@author: 凯风
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import pandas as pd

trainX = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/trainX.csv',header=None) 
trainY = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/trainY.csv',header=None)
needPre = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/testX.csv',header=None)
needPreId = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Processed_data/testY.csv',header=None)

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)

param = {
        'C':[0.001,0.01,0.1,1,10],
        'gamma':[0.1,0.01]
        }

svc = SVC()
cv = KFold(n_splits=4 , shuffle=True , random_state=1)
clf = GridSearchCV(estimator=svc , param_grid=param , cv=cv)
clf.fit(X_train.values,y_train.values.reshape(-1,))

svc = SVC(C=1,gamma=0.01)
svc.fit(X_train,y_train.values.reshape(-1,))
y_pre = svc.predict(X_test)
log_loss(y_test,y_pre)  # 不咋地啊这个分数~

pre = svc.predict(needPre)
needPreId['shot_made_flag'] = pre
needPreId.columns = ['shot_id','shot_made_flag']
needPreId.to_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/Prediction_Result/prediction_svm.csv',index=False)