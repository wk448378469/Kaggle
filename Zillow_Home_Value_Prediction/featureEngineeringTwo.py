# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:48:42 2017

@author: 凯风
"""

import pandas as pd
import rfImputer
import numpy as np

allData = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')
        
cols = ['hashottuborspa','propertycountylandusecode',
        'propertyzoningdesc','fireplaceflag','taxdelinquencyflag']

allData.drop([cols[0]],axis=1,inplace=True)
allData.drop([cols[1]],axis=1,inplace=True)
allData.drop([cols[2]],axis=1,inplace=True)
def processFirePlaceFlag(x):
    if x is True:
        return 1
    return x
allData[cols[3]] = allData[cols[3]].apply(processFirePlaceFlag)
def processTaxelinQuencyFlag(x):
    if x == 'Y':
        return 1
    return x
allData[cols[4]] = allData[cols[4]].apply(processTaxelinQuencyFlag)

dtype_df = allData.dtypes.reset_index()
dtype_df.columns = ['Count','ColumnType']
print (dtype_df)

for c, dtype in zip(allData.columns, allData.dtypes):	
    if dtype == np.float64:
        allData[c] = allData[c].astype(np.float32)

imp_df = rfImputer.rfImputer(allData)
imp_df.impute('random_forest')

imp_df.imputed_df().to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyRFImputer.csv')

