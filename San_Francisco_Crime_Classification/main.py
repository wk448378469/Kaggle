# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:35:31 2017

@author: 凯风
"""




def process_date(data):
    # 字符串转化为事件对象
    data['Dates'] = pd.to_datetime(data['Dates'])
    
    # 创建些新的特征，年、月、日、小时、时刻
    data['year'] = data['Dates'].dt.year
    data['month'] = data['Dates'].dt.month
    data['day'] = data['Dates'].dt.day
    data['wday'] = data['Dates'].dt.dayofweek
    data['hour'] = data['Dates'].dt.hour + data['Dates'].dt.minute/60e0
    data['qtr'] = data['Dates'].dt.quarter
    
    # one-hot-code
    dummy_dayofweek = pd.get_dummies(data['wday'],prefix='wday')
    data = data.join(dummy_dayofweek)
    
    # 新增特征，春夏秋冬   
    def season(x):
        spring = 0
        summer = 0
        fall = 0
        winter = 0
        if x in [2,3,4]:
            spring = 1
        if x in [5,6,7]:
            summer = 1
        if x in [8,9,10]:
            fall = 1
        if x in [11,12,1]:
            winter = 1
        return spring,summer,fall,winter
    data['spring'],data['summer'],data['fall'],data['winter'] = zip(*data['month'].apply(season))
    
    # 新增特征，上午、下午
    def time(x):
        Emorning = 0
        morning = 0
        afternoon = 0
        night = 0
        if x < 12 or x >= 6:
            morning = 1
        if x < 6 or x >= 0:
            Emorning = 1
        if x > 18 or x <= 24:
            night = 1
        if x >= 12 or x <= 18:
            afternoon = 1
        return morning,afternoon,night,Emorning
    data['morning'],data['afternoon'],data['night'],data['Emorning'] = zip(*data['hour'].apply(time))
    
    # 删除旧的特征
    data.drop(['Dates','DayOfWeek','wday'],axis=1,inplace=True)
    return data

def process_address(train,test):
    # 说实话我觉得这个特征处理的很吊~~~
    # 发明者的连接：https://www.kaggle.com/papadopc
    from copy import deepcopy
    # 案发地址的集合
    addresses=sorted(train["Address"].unique())
    # 案件类型的集合
    categories=sorted(train["Category"].unique())
    # 每类案件的数量
    C_counts=train.groupby(["Category"]).size()
    # 每个案发地点，每一类案件的发生数量
    A_C_counts=train.groupby(["Address","Category"]).size()
    # 每个案发地点的案件数量
    A_counts=train.groupby(["Address"]).size()
    # 测试集的案发地址的集合
    new_addresses=sorted(test["Address"].unique())
    # 测试集的每个案发地点的案件数量
    new_A_counts=test.groupby("Address").size()
    # 只在测试集中出现的新的案发地址的集合
    only_new=set(new_addresses+addresses)-set(addresses)
    # 同时在测试集和训练集的案发低脂的集合
    in_both=set(new_addresses).intersection(addresses)
    # 存放每个案件地址，发生案件的几率
    logodds={}
    # 存放每个案件地址，发生每一类案件的概率
    logoddsPA={}
    # 最小记录数量
    MIN_CAT_COUNTS=2
    # 默认案件地址发生案件的几率
    default_logodds=np.log(C_counts/len(train))-np.log(1.0-C_counts/float(len(train)))
    # 迭代每一个地址，对logodds和logoddsPA进行填充~
    for addr in addresses:
        PA=A_counts[addr]/float(len(train)) 
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        for cat in A_C_counts[addr].keys():
            if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
                PA = A_C_counts[addr][cat]/float(A_counts[addr])
                logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0-PA)
        logodds[addr]=pd.Series(logodds[addr])
        logodds[addr].index=range(len(categories))
    for addr in only_new:
        PA=new_A_counts[addr]/float(len(test)+len(train))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        logodds[addr].index=range(len(categories))
    for addr in in_both:
        PA=(A_counts[addr]+new_A_counts[addr])/float(len(test)+len(train))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
        
    # 新增特征，案发地址，发生案件的总概率
    train["logoddsPA"]=train["Address"].apply(lambda x: logoddsPA[x])
    test["logoddsPA"]=test["Address"].apply(lambda x: logoddsPA[x])

    # 新增特征，案发地址，发生每一类案件的概率
    address_features=train["Address"].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+str(x) for x in range(len(address_features.columns))]
    train = train.join(address_features.ix[:,:])
    address_features=test["Address"].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+str(x) for x in range(len(address_features.columns))]
    test = test.join(address_features.ix[:,:])

    train.drop('Address',axis = 1, inplace=True)
    test.drop('Address',axis = 1, inplace=True)
    return train,test

def process_SS(train,test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train)
    
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train,test


if __name__ =='__main__':
    # 数据集没有缺失值9 9 
    import pandas as pd
    import numpy as np
    train = pd.read_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/train.csv')
    test = pd.read_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/test.csv')
    test.drop('Id',axis=1,inplace=True)
    train.drop(['Descript','Resolution'],axis=1,inplace=True)
    
    # 日期处理
    train = process_date(train)
    test = process_date(test)
    
    # Phdistrict处理
    dummy_pdd_train = pd.get_dummies(train['PdDistrict'],prefix='PdD')
    train = train.join(dummy_pdd_train)
    dummy_pdd_test = pd.get_dummies(test['PdDistrict'],prefix='PdD')
    test = test.join(dummy_pdd_test)
    train.drop('PdDistrict',axis=1,inplace=True)
    test.drop('PdDistrict',axis=1,inplace=True)
    
    # 犯罪地点处理
    # 判断是否发生在交叉路口
    train['interaction'] = train['Address'].apply(lambda x : 1 if '/' in x else 0)
    test['interaction'] = test['Address'].apply(lambda x:1 if '/' in x else 0)
    # 生成新的关于犯罪地点的特征
    train,test = process_address(train,test)
        
    # 目标变量处理
    target = train['Category']
    num = len(target.unique())      # 犯罪类型的数量
    name = list(target.unique())    # 犯罪类型的字符串          
    for i in range(0,num):
        target[target.values == name[i]] = i    
    target_dict = pd.DataFrame(name)
    train.drop('Category',axis=1,inplace=True)
    
    # 标准化feature
    train,test = process_SS(train,test)
    
    # 保存文件
    target.to_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/train_target.csv',header=False,index=False)
    target_dict.to_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/target_dict.csv',header=False,index=False)
    pd.DataFrame(train).to_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/train.csv',header=False,index=False)
    pd.DataFrame(train).to_csv('D:/mygit/Kaggle/San_Francisco_Crime_Classification/Processed_data/test.csv',header=False,index=False)