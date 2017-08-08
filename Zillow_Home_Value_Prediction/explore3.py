# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:09:11 2017

@author: 凯风
"""

import pandas as pd

RFImputer = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyRFImputer.csv')
properties = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv')

RFImputer['parcelid'] = properties['parcelid']

train = pd.read_csv("D:/mygit/Kaggle/Zillow_Home_Value_Prediction/train_2016_v2.csv", parse_dates=["transactiondate"])

train = pd.merge(train, RFImputer, how='left', on='parcelid')

RFImputer.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyRFImputer.csv',index=False)





