
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

def processing_part0(train,test):
    train['hasname'] = train['Name'].fillna(0)
    train.loc[train['hasname'] != 0 , 'hasname'] = 1
    test['hasname'] = test['Name'].fillna(0)
    test.loc[test['hasname'] != 0 , 'hasname'] = 1
    
    train['Name'].fillna('Unnamed',inplace=True)
    test['Name'].fillna('Unnamed',inplace=True)
    
    unique_name_train,name_counts_train = np.unique(train_data['Name'],return_counts=True)
    unique_name_test,name_counts_test = np.unique(test_data['Name'],return_counts=True)
    
    name_dict_train = dict(zip(unique_name_train,1e0*name_counts_train/np.sum(name_counts_train)))
    name_dict_test = dict(zip(unique_name_test,1e0*name_counts_test/np.sum(name_counts_test)))
    
    #合并两个名字的列表
    name_dict = dict(name_dict_train,**name_dict_test)
    
    def occur(x,dic):
        # 返回名字的使用情况
        return dic[x]
    
    train['nameoccurrance'] = train['Name'].apply(lambda x : occur(x,name_dict))
    test['nameoccurrance'] = test['Name'].apply(lambda x: occur(x,name_dict))
    
    train.drop('Name',axis=1,inplace=True)
    test.drop('Name',axis=1,inplace=True)
    
    return train,test


def processing_part1(data,types='train'):
    
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
    def pro_fertility(sex):
        if 'Intact' in sex:
            return 1
        elif 'Neutered' in sex or 'Spayed' in sex:
            return 2
        else:
            return 0

    def pro_sex(sex):
        if 'Male' in sex:
            return 1
        elif 'Female' in sex:
            return 2
        else:
            return 0
    # 处理缺失值
    data['SexuponOutcome'].fillna('Unknown')
    train_data.loc[train_data['SexuponOutcome'] != train_data['SexuponOutcome'],'SexuponOutcome'] = 'Unknown'
    # 生成两个新的feature
    data['fertility'] = data['SexuponOutcome'].map(pro_fertility)
    data['sex'] = data['SexuponOutcome'].map(pro_sex)
    
    gender = {'Neutered Male':1, 'Spayed Female':2, 'Intact Male':3, 'Intact Female':4, 'Unknown':5}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)    
    return data
    
def processing_part2(data):
    pass

def processing_part3(data):
    pass

def processing_part4(data):
    pass

    
if __name__ == '__main__':

    train_file = 'D:/mygit/Kaggle/Shelter_Animal_Outcomes/train.csv'
    test_file = 'D:/mygit/Kaggle/Shelter_Animal_Outcomes/test.csv'
    # 读取数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # 处理name
    train_data,test_data = processing_part0(train_data,test_data)
    
    # 主要处理Age,sex这两个特征
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
    
    