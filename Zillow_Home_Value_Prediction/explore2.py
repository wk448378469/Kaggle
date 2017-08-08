# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:09:11 2017

@author: 凯风
"""

import pandas as pd
import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv')

train = pd.merge(train, properties, how='left', on='parcelid')
del properties ; gc.collect()

y = train['logerror']
x = train.drop(['logerror','parcelid'],axis=1)
del train ; gc.collect()

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    return df

x = get_features(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
del x,y ;gc.collect()

dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test)
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'silent': 1
}

model_XGB = xgb.train(xgb_params,dtrain,num_boost_round=250)

fig,ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model_XGB,max_num_features=76,height=0.8,ax=ax)
plt.show()

y_pred = model_XGB.predict(dtest)

test1 = np.sum([abs(y_test.values[i]-y_pred[i]) for i in range(len(y_test))]) / len(y_test)

# 造特征吧继续

myself = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv')
RFImputer = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyRFImputer.csv')

myselfColumnsType = myself.dtypes.reindex()
RFImputerColumnsType = RFImputer.dtypes.reindex()

RFImputer['airconditioningtypeid'] = RFImputer.airconditioningtypeid.astype('int64')
RFImputer['architecturalstyletypeid'] = RFImputer.architecturalstyletypeid.astype('int64')
RFImputer['basementsqft'] = RFImputer.basementsqft.astype('float64')
RFImputer['bathroomcnt'] = RFImputer.bathroomcnt.astype('float64')
RFImputer['bedroomcnt'] = RFImputer.bedroomcnt.astype('float64')
RFImputer['buildingclasstypeid'] = RFImputer.buildingclasstypeid.astype('float64')
RFImputer['buildingqualitytypeid'] = RFImputer.buildingqualitytypeid.astype('int64')
RFImputer['calculatedbathnbr'] = RFImputer.calculatedbathnbr.astype('float64')
RFImputer['decktypeid'] = RFImputer.decktypeid.astype('int64')
RFImputer['finishedfloor1squarefeet'] = RFImputer.finishedfloor1squarefeet.astype('float64')
RFImputer['calculatedfinishedsquarefeet'] = RFImputer.calculatedfinishedsquarefeet.astype('float64')
RFImputer['finishedsquarefeet12'] = RFImputer.finishedsquarefeet12.astype('float64')
RFImputer['finishedsquarefeet13'] = RFImputer.finishedsquarefeet13.astype('float64')
RFImputer['finishedsquarefeet15'] = RFImputer.finishedsquarefeet15.astype('float64')
RFImputer['finishedsquarefeet50'] = RFImputer.finishedsquarefeet50.astype('float64')
RFImputer['finishedsquarefeet6'] = RFImputer.finishedsquarefeet6.astype('float64')
RFImputer['fips'] = RFImputer.fips.astype('int64')
RFImputer['fireplacecnt'] = RFImputer.fireplacecnt.astype('float64')
RFImputer['fullbathcnt'] = RFImputer.fullbathcnt.astype('float64')
RFImputer['garagecarcnt'] = RFImputer.garagecarcnt.astype('float64')
RFImputer['garagetotalsqft'] = RFImputer.garagetotalsqft.astype('float64')
RFImputer['heatingorsystemtypeid'] = RFImputer.heatingorsystemtypeid.astype('float64')
RFImputer['latitude'] = RFImputer.latitude.astype('float64')
RFImputer['longitude'] = RFImputer.longitude.astype('float64')
RFImputer['lotsizesquarefeet'] = RFImputer.lotsizesquarefeet.astype('float64')
RFImputer['poolcnt'] = RFImputer.poolcnt.astype('float64')
RFImputer['poolsizesum'] = RFImputer.poolsizesum.astype('float64')
RFImputer['pooltypeid10'] = RFImputer.pooltypeid10.astype('float64')
RFImputer['pooltypeid2'] = RFImputer.pooltypeid2.astype('float64')
RFImputer['pooltypeid7'] = RFImputer.pooltypeid7.astype('float64')
RFImputer['propertylandusetypeid'] = RFImputer.propertylandusetypeid.astype('int64')
RFImputer['rawcensustractandblock'] = RFImputer.rawcensustractandblock.astype('float64')
RFImputer['regionidcity'] = RFImputer.regionidcity.astype('float64')
RFImputer['regionidcounty'] = RFImputer.regionidcounty.astype('float64')
RFImputer['regionidneighborhood'] = RFImputer.regionidneighborhood.astype('float64')
RFImputer['regionidzip'] = RFImputer.regionidzip.astype('int64')
RFImputer['roomcnt'] = RFImputer.roomcnt.astype('float64')
RFImputer['storytypeid'] = RFImputer.storytypeid.astype('int64')
RFImputer['threequarterbathnbr'] = RFImputer.threequarterbathnbr.astype('float64')
RFImputer['typeconstructiontypeid'] = RFImputer.typeconstructiontypeid.astype('int64')
RFImputer['unitcnt'] = RFImputer.unitcnt.astype('float64')
RFImputer['yardbuildingsqft17'] = RFImputer.yardbuildingsqft17.astype('float64')
RFImputer['yardbuildingsqft26'] = RFImputer.yardbuildingsqft26.astype('float64')
RFImputer['yearbuilt'] = RFImputer.yearbuilt.astype('int64')
RFImputer['numberofstories'] = RFImputer.numberofstories.astype('int64')
RFImputer['fireplaceflag'] = RFImputer.fireplaceflag.astype('int64')
RFImputer['structuretaxvaluedollarcnt'] = RFImputer.structuretaxvaluedollarcnt.astype('float64')
RFImputer['taxvaluedollarcnt'] = RFImputer.taxvaluedollarcnt.astype('float64')
RFImputer['assessmentyear'] = RFImputer.assessmentyear.astype('int64')
RFImputer['landtaxvaluedollarcnt'] = RFImputer.landtaxvaluedollarcnt.astype('int64')
RFImputer['taxamount'] = RFImputer.taxamount.astype('float64')
RFImputer['taxdelinquencyflag'] = RFImputer.taxdelinquencyflag.astype('int64')
RFImputer['taxdelinquencyyear'] = RFImputer.taxdelinquencyyear.astype('float64')
RFImputer['censustractandblock'] = RFImputer.censustractandblock.astype('float64')


RFImputer.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyRFImputer.csv',index=False)





