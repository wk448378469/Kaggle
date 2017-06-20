
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:09:33 2017

@author: 凯风
"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb


def processing_part1(data,types='train'):
    
    # 处理动物名字这个略显鸡肋的feature
	# 明天这个地方调整一下，在名字上在做些文章
    data['hasname'] = data['Name'].fillna(0)
    data.loc[data['hasname'] != 0 ,'hasname'] = 1
    data.drop('Name',axis=1,inplace=True)
    
    if types == 'train':
        data.drop('AnimalID',axis=1,inplace=True)
    else:
        data.drop('ID',axis=1,inplace=True)
    
    # 处理AgeuponOutcome这个特征
    def pro_age(age):
        try:
            age_str = age.split()
        except:
            return None
        if 'year' in age_str[1]:
            return float(age_str[0]) * 365
        elif 'month' in age_str[1]:
            return float(age_str[0]) * 30
        elif 'week' in age_str[1]:
            return float(age_str[0]) * 7
        elif 'day' in age_str[1]:
            return float(age_str[0])
    data['AgeuponOutcome'] = data['AgeuponOutcome'].map(pro_age)
    data.loc[data['AgeuponOutcome'].isnull(),'AgeuponOutcome'] = data['AgeuponOutcome'].median()
    
    # 处理目标变量
    if types == 'train':    
        data.ix[data.OutcomeType=='Return_to_owner','target'] = 0
        data.ix[data.OutcomeType=='Transfer','target'] = 1
        data.ix[data.OutcomeType=='Euthanasia','target'] = 2
        data.ix[data.OutcomeType=='Died','target'] = 3
        data.ix[data.OutcomeType=='Adoption','target'] = 4
        data.drop(['OutcomeType','OutcomeSubtype'],axis=1,inplace=True)
    
    # 处理动物类型
    data.ix[data.AnimalType == 'Cat','is_cat'] = 1
    data.ix[data.AnimalType == 'Dog','is_cat'] = 0
    data.ix[data.AnimalType == 'Dog','is_dog'] = 1
    data.ix[data.AnimalType == 'Cat','is_dog'] = 0
    data.drop('AnimalType',axis=1,inplace=True)
    
    # 处理性别问题
	# 这里也记得调整下，不要随便删除特征！！！
    gender = {'Neutered Male':1, 'Spayed Female':2, 'Intact Male':3, 'Intact Female':4, 'Unknown':5, np.nan:0}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)
    
    return data
    
def processing_part2(data):
    pass

def processing_part3(data):
    pass

def processing_part4(data):
    pass

    
if __name__ == '__main__':

    train_file = 'C:/Users/dell/Desktop/train.csv'
    test_file = 'C:/Users/dell/Desktop/test.csv'
    # 读取数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # 清洗数据
    train_data = processing_part1(train_data,types='train')
    test_data = processing_part1(test_data,types='test')


    # 处理Breed
    train_data = processing_part2(train_data)
    test_data = processing_part2(test_data)

    # 处理color
    train_data = processing_part3(train_data)
    test_data = processing_part3(test_data)
    
    # 处理dateTime
    train_data = processing_part4(train_data)
    test_data = processing_part4(test_data)
    
    