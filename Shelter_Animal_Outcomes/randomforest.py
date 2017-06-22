# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:17:16 2017

@author: 凯风
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pandas as pd

# 读取数据
test = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/test.csv')
train = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train.csv')
target = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train_target.csv',header=None)

train.drop('Unnamed: 0',axis=1,inplace=True)
test.drop('Unnamed: 0',axis=1,inplace=True)
target.drop(0,axis=1,inplace=True)

def best_params(X,y):
    rfc = RandomForestClassifier()
    param_grid = {
            'n_estimators': [10, 40, 100, 200, 400],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_impurity_split': [1e-9, 1e-7, 1e-4, 1e-3],
            'min_samples_split': [1, 5, 8],
            'min_samples_leaf': [1, 2, 5]
            }
    cv_rfc = GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5)
    cv_rfc.fit(X,y)
    return cv_rfc.best_params_

best_param = best_params(train,target)

RFC = RandomForestClassifier(n_estimators=10,max_features='auto',min_impurity_split=1e-7,min_samples_split=2,min_samples_leaf=1)
RFC.fit(train,target)
prediction = RFC.predict_proba(test)

'''
    data.ix[data.OutcomeType=='Return_to_owner','target'] = 0
    data.ix[data.OutcomeType=='Transfer','target'] = 1
    data.ix[data.OutcomeType=='Euthanasia','target'] = 2
    data.ix[data.OutcomeType=='Died','target'] = 3
    data.ix[data.OutcomeType=='Adoption','target'] = 4
'''

output = pd.DataFrame(prediction,columns=['Return_to_owner','Transfer','Euthanasia','Died','Adoption'])
output.columns.names = ['ID']
output.index.names = ['ID']
output.index += 1
output.to_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/model_pre/prediction_randomforest.csv')
# 自己在文件中补充上ID吧。。。。