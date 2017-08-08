# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:49:57 2017

@author: 凯风
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
from scipy.stats import boxcox

data = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv')
originalData = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')

'''[
    'parcelid',                                 ID
    'airconditioningtypeid',                    冷却系统类型，若有
    'architecturalstyletypeid',                 房屋的建筑类型
    'basementsqft',                             生活区低于或部分低于地面的面积
    'bathroomcnt',                              浴室数量
    'bedroomcnt',                               卧室数量
    'buildingclasstypeid',                      建筑框架式
    'buildingqualitytypeid',                    建筑状况从高到低的总体评估
    'calculatedbathnbr',                        和卧室数量很像
    'finishedfloor1squarefeet',                 进入后第一层的面积
    'calculatedfinishedsquarefeet',             总居住面积                           ###
    'finishedsquarefeet12',                     完成生活区
    'finishedsquarefeet13',                     周围生活区
    'finishedsquarefeet15',                     总面积
    'finishedsquarefeet50',                     第一层的居住面积
    'finishedsquarefeet6',                      基地未完成和完成区
    'fips',                                     标准代码
    'fireplacecnt',                             壁炉的数量
    'fullbathcnt',                              完整的浴室数量（包含更多内容）
    'garagecarcnt',                             车库数量
    'garagetotalsqft',                          车库面积
    'heatingorsystemtypeid',                    供暖系统类型
    'latitude',                                 经度
    'longitude',                                维度
    'lotsizesquarefeet',                        也是一个面积的特征                        
    'poolcnt',                                  泳池数量            
    'poolsizesum',                              泳池的面积
    'pooltypeid10',                             水疗或热水缸
    'pooltypeid2',                              带有热水缸的泳池
    'pooltypeid7',                              没有热水缸的泳池
    'propertycountylandusecode',                县级土地使用代码
    'propertylandusetypeid',                    财产用地类型
    'rawcensustractandblock',                   人口普查ID
    'regionidcity',                             物业所在县
    'regionidcounty',                           物业所在市
    'regionidzip',                              所属的邮编编码
    'roomcnt',                                  住宅房间总数量
    'threequarterbathnbr',                      室内3/4间浴室
    'typeconstructiontypeid',                   所用建筑材料
    'unitcnt',                                  结构内置的单位数
    'yardbuildingsqft17',                       院子相关的
    'yardbuildingsqft26',                       仓库建在院子里
    'yearbuilt',                                建成年
    'numberofstories',                          stories或levels的数量
    'fireplaceflag',                            现在是否有壁炉
    'structuretaxvaluedollarcnt',               建筑结构在包裹上的评估价值            ###
    'taxvaluedollarcnt',                        包裹的总税款评估价值                  ###
    'assessmentyear',                           财产税评估年度
    'landtaxvaluedollarcnt',                    包裹土地面积的评估价值
    'taxamount',                                该评估年度的财产税总额                 ###
    'taxdelinquencyflag',                       截至2015年，该包裹的财产税已到期
    'taxdelinquencyyear',                       未缴财产税到期的年份
    'censustractandblock',                      人口普查和块ID组合 - 还包含通过扩展
'''

# 是否溢价，根据浴室的数量和卧室的数量来判定
data['isPremium'] = np.where((data['bedroomcnt'] >=2) & (data['bathroomcnt']>=2) , 1 , 0)

# 两个年限的之差,评估年-建成年
originalData['yearbuilt'].fillna(originalData['yearbuilt'].mode()[0],inplace=True)
data['assessmentYearsPass'] = data['assessmentyear'] - originalData['yearbuilt']

# 是否溢价2，根据壁炉fireplacecnt、泳池poolcnt
data['isPremium2'] = np.where((data['fireplacecnt'] >=1) & (data['poolcnt']>=1) , 1 , 0)

# finishedfloor1squarefeet 和 finishedsquarefeet50之差
data['floor1OtherUsed'] = data['finishedfloor1squarefeet'] - data['finishedsquarefeet50']

# 每平米价值评估
data['perSquareMeterValue'] = data['taxvaluedollarcnt'] / data['finishedsquarefeet15']

# regionidcity,label化
LE1 = LabelEncoder()
data['regionidcity'] = LE1.fit_transform(data['regionidcity'])
del LE1 ;gc.collect()

# 每个城市的平均房屋的评估价值
grouped = data['structuretaxvaluedollarcnt'].groupby(data['regionidcounty'])
everyCountyValueMean = grouped.mean()
data['countydollardifference'] = data['regionidcounty'].apply(lambda x: everyCountyValueMean[x])
data['countydollardifference'] = data['structuretaxvaluedollarcnt'] - data['countydollardifference']
del everyCountyValueMean,grouped ; gc.collect()

# 每个县的凭据房屋的价值评估
grouped = data['structuretaxvaluedollarcnt'].groupby(data['regionidcity'])
everyCityValueMean = grouped.mean()
data['citydollardifference'] = data['regionidcity'].apply(lambda x: everyCityValueMean[x])
data['citydollardifference'] = data['structuretaxvaluedollarcnt'] - data['citydollardifference']
del everyCityValueMean,grouped ; gc.collect()

# rawcensustractandblock
data['rawcensustractandblock'] = originalData['rawcensustractandblock'].fillna(originalData['rawcensustractandblock'].mean())
LE2 = LabelEncoder()
data['rawcensustractandblock'] = LE2.fit_transform(data['rawcensustractandblock'])
del LE2 ; gc.collect()

# 根据经纬度造特征，计算每个城市的中心，然后求每个房子距离中心的距离
grouped1 = data['latitude'].groupby(data['regionidcounty'])
everyCountyLatitudeCenter = grouped1.mean()                     # 三个城市的中心经纬度
grouped2 = data['longitude'].groupby(data['regionidcounty'])
everyCountyLongitudeCenter = grouped2.mean()
data['contryLatitudeCenter'] = data['regionidcounty'].apply(lambda x:everyCountyLatitudeCenter[x] )
data['contryLongitudeCenter'] = data['regionidcounty'].apply(lambda x:everyCountyLongitudeCenter[x] )
del grouped1,grouped2,everyCountyLatitudeCenter,everyCountyLongitudeCenter ; gc.collect()
data['distanceFromCountryCenter'] =np.sqrt(np.square((data['latitude'] - data['contryLatitudeCenter'])) + np.square((data['longitude'] - data['contryLongitudeCenter'])))
data.drop(['contryLatitudeCenter','contryLongitudeCenter'],axis=1,inplace=True)

# 和上面类似，是每个city的
grouped1 = data['latitude'].groupby(data['regionidcity'])
everyCityLatitudeCenter = grouped1.mean()                     # 三个城市的中心经纬度
grouped2 = data['longitude'].groupby(data['regionidcity'])
everyCityLongitudeCenter = grouped2.mean()
data['cityLatitudeCenter'] = data['regionidcity'].apply(lambda x:everyCityLatitudeCenter[x] )
data['cityLongitudeCenter'] = data['regionidcity'].apply(lambda x:everyCityLongitudeCenter[x] )
del grouped1,grouped2,everyCityLatitudeCenter,everyCityLongitudeCenter ; gc.collect()
data['distanceFromCityCenter'] =np.sqrt(np.square((data['latitude'] - data['cityLatitudeCenter'])) + np.square((data['longitude'] - data['cityLongitudeCenter'])))
data.drop(['cityLatitudeCenter','cityLongitudeCenter'],axis=1,inplace=True)

# propertyzoningdesc  之前删掉的
originalData['propertyzoningdesc'].fillna(originalData['propertyzoningdesc'].mode()[0],axis=0,inplace=True)
descType = originalData.propertyzoningdesc.value_counts()
data['descRatio'] = originalData['propertyzoningdesc'].apply(lambda x: descType[x] / originalData.shape[0])
del descType ; gc.collect()

# 车库的平均面积
data['catAverageArea'] = data['garagetotalsqft'] / data['garagecarcnt']
data['catAverageArea'].fillna(0,inplace=True)
data['catAverageArea'] = data['catAverageArea'].apply(lambda x: 0 if x == np.inf else x)

# 非居住区面积 finishedsquarefeet15 - calculatedfinishedsquarefeet
originalData['finishedsquarefeet15'].fillna(originalData['finishedsquarefeet15'].mean(),inplace=True)
originalData['calculatedfinishedsquarefeet'].fillna(originalData['calculatedfinishedsquarefeet'].mean(),inplace=True)
data['nonLiveArea'] = originalData['finishedsquarefeet15'] - originalData['calculatedfinishedsquarefeet']
data['nonLiveArea'] = data['nonLiveArea'].apply(lambda x : 0 if x <= 0 else x)
data['nonLiveArea'] = np.log(data['nonLiveArea'] + 1)

# structuretaxvaluedollarcnt - landtaxvaluedollarcnt
data['newValueCnt'] = data['structuretaxvaluedollarcnt'] - data['landtaxvaluedollarcnt']

# 价值相加
data['allValue'] = data['structuretaxvaluedollarcnt'] + data['taxvaluedollarcnt'] + data['landtaxvaluedollarcnt']

# 根据房屋的建筑年代，造特征，是否为2000年后的
originalData['yearbuilt'].fillna(originalData['yearbuilt'].mode()[0],inplace=True)
data['is2000After'] = originalData['yearbuilt'].apply(lambda x : 1 if x >= 2000 else 0)

# 财产的生命
data['valueLife'] = 2018 - originalData['yearbuilt']

# 一个除法特征
data['livingAreaError'] = data['calculatedfinishedsquarefeet']/data['finishedsquarefeet12']

# 生活区的比重
data['livingAreaProp'] = data['calculatedfinishedsquarefeet']/data['lotsizesquarefeet']
data['livingAreaProp2'] = data['finishedsquarefeet12']/data['finishedsquarefeet15']

# 非生活区的新特征
data['extraSpace'] = data['lotsizesquarefeet'] - data['calculatedfinishedsquarefeet'] 
data['extraSpace2'] = data['finishedsquarefeet15'] - data['finishedsquarefeet12'] 

# room的大小
data['avRoomSize'] = data['calculatedfinishedsquarefeet']/data['roomcnt'] 
data['avRoomSize'] = data['avRoomSize'].apply(lambda x: 0 if x == np.inf else x)

# 建筑结构价值和土地面积的比率
data['valueProp'] = data['structuretaxvaluedollarcnt']/data['landtaxvaluedollarcnt']

# 是否溢价三
data['isPremium3'] = ((data['garagecarcnt']>0) & (data['pooltypeid10']>0) & (data['airconditioningtypeid']!=5))*1 

# 经纬度转换一下
data['newLocationX'] = np.cos(data['latitude']) * np.cos(data['longitude'])
data['newLocationY'] = np.cos(data['latitude']) * np.sin(data['longitude'])
data['newLocationZ'] = np.sin(data['latitude'])

# 财产税的比例
data['valueRatio'] = data['taxvaluedollarcnt']/data['taxamount']

# 两个相似的特征描述的特征的乘法
data['taxScore'] = data['taxvaluedollarcnt']*data['taxamount']

# zipcode的出现频率
zipCount = data['regionidzip'].value_counts().to_dict()
data['zipCount'] = data['regionidzip'].map(zipCount)
del zipCount;gc.collect()

# city出现频率 ， county的就算了，毕竟只有三个~
cityCount = data['regionidcity'].value_counts().to_dict()
data['cityCount'] = data['regionidcity'].map(cityCount)
del cityCount;gc.collect()

# 就算是多项式的特征吧,structuretaxvaluedollarcnt在基于模型(xgboost)的中是最重要的~
data["structuretaxvaluedollarcnt^2"] = data["structuretaxvaluedollarcnt"] ** 2
data["structuretaxvaluedollarcnt^3"] = data["structuretaxvaluedollarcnt"] ** 3

# structuretaxvaluedollarcnt偏离平均值
grouped = data['structuretaxvaluedollarcnt'].groupby(data['regionidcity']).aggregate('mean').to_dict()
data['avgStructuretaxvaluedollarcnt'] = data['regionidcity'].map(grouped)
data['devStructuretaxvaluedollarcnt'] = abs((data['structuretaxvaluedollarcnt'] - data['avgStructuretaxvaluedollarcnt']))/data['avgStructuretaxvaluedollarcnt']
del grouped ; gc.collect()

del originalData; gc.collect()

data.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself2.csv',index=False)











