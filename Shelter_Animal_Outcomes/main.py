
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:09:33 2017

@author: 凯风
"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    # 处理NaN浮点型的一个数据
    train_data.loc[train_data['SexuponOutcome'] != train_data['SexuponOutcome'],'SexuponOutcome'] = 'Unknown'
    # 生成两个新的feature
    data['fertility'] = data['SexuponOutcome'].map(pro_fertility)
    data['sex'] = data['SexuponOutcome'].map(pro_sex)
    
    gender = {'Neutered Male':1, 'Spayed Female':2, 'Intact Male':3, 'Intact Female':4, 'Unknown':5}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)    
    return data
    
def processing_part2(data):
    # 新增一个是否是混合的种类
    data['isMix'] = data['Breed'].str.contains('mix',case=False).astype(int)
    
    # 是否是交叉
    def breed_cross(breedstring):
        if "/" in breedstring:
            return 1
        return 0
    data['cross'] = data['Breed'].apply(lambda x:breed_cross(x))
    
    # 毛发的类型
    def breed_hairtype(breedstring):
        if "Shorthair" in breedstring:
            return 1
        if "Medium Hair" in breedstring:
            return 2
        if "Longhair" in breedstring:
            return 3
        if "hairless" in breedstring.lower():
            return 4
        if "Wirehair" in breedstring:
            return 5
        return 0
    data['hairtype'] = data['Breed'].apply(lambda x:breed_hairtype(x))
    
    # 是否有攻击性
    def breed_anyof(breedstring, anyof):
        for w in anyof:
            if w.lower() in breedstring.lower():
                return 1
        return 0
    data['Isaggressive'] = data['Breed'].apply(lambda x:breed_anyof(x,["Rottweiler", "Pit Bull", "Siberian Husky"]))
    
    data.drop('Breed',axis=1,inplace=True)
    return data

def processing_part3(data):
    
    def multi_color(colorstring):
        if "/" in colorstring:
            return 1
        return 0
    data['multi_color'] = data['Color'].apply(lambda x: multi_color(x))
    
    # 对色表
    colors = ['Apricot', 'Black', 'Black Brindle', 'Black Smoke', 'Black Tiger', 'Blue', 'Blue Cream', \
                         'Blue Merle', 'Blue Smoke', 'Blue Tick', 'Blue Tiger', 'Brown', 'Brown Brindle', 'Brown Merle', \
                         'Brown Tabby', 'Brown Tiger', 'Buff', 'Chocolate', 'Cream', 'Fawn', 'Gold', 'Gray', 'Liver', \
                         'Liver Tick', 'Orange', 'Pink', 'Red', 'Red Merle', 'Red Tick', 'Ruddy', 'Sable', 'Silver', \
                         'Tan', 'Tricolor', 'White', 'Yellow', 'Yellow Brindle','Gray Tiger',\
                         'Agouti', 'Apricot', 'Black', 'Black Smoke', 'Black Tabby', 'Black Tiger', 'Blue', \
                         'Blue Cream', 'Blue Point', 'Blue Smoke', 'Blue Tabby', 'Brown', 'Brown Tabby', 'Brown Tiger', \
                         'Buff', 'Calico', 'Calico Point', 'Chocolate', 'Chocolate Point', 'Cream', 'Cream Tabby', \
                         'Flame Point', 'Gray', 'Gray Tabby', 'Lilac Point', 'Lynx Point', 'Orange', 'Orange Tabby', \
                         'Orange Tiger', 'Pink', 'Seal Point', 'Silver', 'Silver Lynx Point', 'Silver Tabby', 'Tan', \
                         'Torbie', 'Tortie', 'Tortie Point', 'Tricolor', 'White', 'Yellow', 'Red', \
                         'Black Brindle','Fawn']

    color_groups = ['Light','Dark','Dark/Medium','Dark','Dark/Medium','Medium','Medium/Light', \
                 'Light/Medium', 'Medium', 'Medium', 'Medium', 'Medium','Medium', 'Light/Medium', 
                 'Medium', 'Medium','Light','Dark','Light','Light/Dark','Medium','Medium','Medium',
                 'Light/Medium','Medium','Medium','Medium','Light/Medium','Light/Medium','Medium','Medium/Dark','Medium',
                 'Medium','Light/Medium/Dark','Light','Light','Light/Medium','Medium/Dark',\
                 'Medium','Light','Dark','Dark', 'Dark', 'Dark/Medium', 'Medium',\
                 'Medium/Light','Light/Medium','Medium','Medium/Light','Medium','Medium/Light','Medium',\
                 'Light','Light/Medium/Dark','Light/Medium/Dark','Dark','Medium/Dark','Light','Light/Medium',\
                 'Light','Medium','Medium/Light', 'Light/Medium','Light/Medium','Medium','Medium',\
                  'Medium','Medium','Light/Dark','Medium','Light/Medium','Medium/Dark', 'Medium',\
                 'Medium/Dark','Dark/Medium','Light/Dark','Light/Medium/Dark','Light','Light', 'Medium',\
                 'Dark/Medium','Light']

    cdict = dict(zip(colors, color_groups))
    def parse_color(colorstring):
        split = colorstring.split('/')
        string = ""
        #print split
        for j in split:
            #print '   ',j
            string += cdict[j]+'/'
        collect = []
        for j in string[:-1].split('/'):
            if j not in collect:
                collect.append(j)
        if len(collect) == 3:
            color_groups = 'Tricolor'
        else:
            color_groups = "/".join(collect)
        return color_groups
    
    def islight(colorstring):
        if 'Light' in colorstring:
            return 1
        return 0
    def ismedium(colorstring):
        if 'Medium' in colorstring:
            return 1
        return 0
    def isdark(colorstring):
        if 'Dark' in colorstring:
            return 1
        return 0
    def istricolor(colorstring):
        if 'Tricolor' in colorstring:
            return 1
        return 0
    data['colorgroups'] = data['Color'].apply(lambda x: parse_color(x))
    data['islight'] = data['colorgroups'].apply(lambda x:islight(x))
    data['ismedium'] = data['colorgroups'].apply(lambda x:ismedium(x))
    data['isdark'] = data['colorgroups'].apply(lambda x:isdark(x))
    data['istricolor'] = data['colorgroups'].apply(lambda x:istricolor(x))
    
    data.drop(['Color','colorgroups'],axis=1,inplace=True)
    return data

def processing_part4(data):
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['year'] = data['DateTime'].dt.year
    data['month'] = data['DateTime'].dt.month
    data['day'] = data['DateTime'].dt.day
    data['wday'] = data['DateTime'].dt.dayofweek
    data['hour'] = data['DateTime'].dt.hour + data['DateTime'].dt.minute/60e0
    data['qtr'] = data['DateTime'].dt.quarter
    data.drop('DateTime',axis=1,inplace=True)
    return data

def standard(train,test):
    SS = StandardScaler()
    SS.fit(train)
    train = SS.transform(train)
    test = SS.transform(test)
    return train,test

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
    
    # 标准化
    target = train_data['target']
    train_data.drop('target',axis=1,inplace=True)
    train_data,test_data = standard(train_data,test_data)
    
    # 保存数据
    target.to_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train_target.csv')
    pd.DataFrame(train_data).to_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train.csv')
    pd.DataFrame(test_data).to_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/test.csv')