# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:38:56 2017

@author: 凯风
"""

import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#准备数据
print('loading data')
allData = pd.read_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/properties_2016.csv')
'''
dataDescription = {
        'airconditioningtypeid':' Type of cooling system present in the home (if any)',
        'architecturalstyletypeid':' Architectural style of the home (i.e. ranch, colonial, split-level, etc…)',
        'basementsqft':'Finished living area below or partially below ground level',
        'bathroomcnt':'Number of bathrooms in home including fractional bathrooms',
        'bedroomcnt':' Number of bedrooms in home ',
        'buildingqualitytypeid':' Overall assessment of condition of the building from best (lowest) to worst (highest)',
        'buildingclasstypeid':'The building framing type (steel frame, wood frame, concrete/brick) ',
        'calculatedbathnbr':'Number of bathrooms in home including fractional bathroom',
        'decktypeid':'Type of deck (if any) present on parcel',
        'threequarterbathnbr'	:'Number of 3/4 bathrooms in house (shower + sink + toilet)',
        'finishedfloor1squarefeet':'Size of the finished living area on the first (entry) floor of the home',
        'calculatedfinishedsquarefeet':'Calculated total finished living area of the home ',
        'finishedsquarefeet6'	:'Base unfinished and finished area',
        'finishedsquarefeet12':'Finished living area',
        'finishedsquarefeet13':'Perimeter  living area',
        'finishedsquarefeet15':'Total area',
        'finishedsquarefeet50':'Size of the finished living area on the first (entry) floor of the home',
        'fips':'Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details',
        'fireplacecnt':'Number of fireplaces in a home (if any)',
        'fireplaceflag':'Is a fireplace present in this home ',
        'fullbathcnt':'Number of full bathrooms (sink, shower + bathtub, and toilet) present in home',
        'garagecarcnt':'Total number of garages on the lot including an attached garage',
        'garagetotalsqft':'Total number of square feet of all garages on lot including an attached garage',
        'hashottuborspa':'Does the home have a hot tub or spa',
        'heatingorsystemtypeid':'Type of home heating system',
        'latitude':'Latitude of the middle of the parcel multiplied by 10e6',
        'longitude'	:'Longitude of the middle of the parcel multiplied by 10e6',
        'lotsizesquarefeet':'Area of the lot in square feet',
        'numberofstories':'Number of stories or levels the home has',
        'parcelid':'Unique identifier for parcels (lots) ',
        'poolcnt':'Number of pools on the lot (if any)',
        'poolsizesum':'Total square footage of all pools on property',
        'pooltypeid10'	:'Spa or Hot Tub',
        'pooltypeid2':'Pool with Spa/Hot Tub',
        'pooltypeid7':'Pool without hot tub',
        'propertycountylandusecode'	:'County land use code i.e. it\'s zoning at the county level',
        'propertylandusetypeid':'Type of land use the property is zoned for',
        'propertyzoningdesc':'Description of the allowed land uses (zoning) for that property',
        'rawcensustractandblock'	:'Census tract and block ID combined - also contains blockgroup assignment by extension',
        'censustractandblock'	:'Census tract and block ID combined - also contains blockgroup assignment by extension',
        'regionidcounty':'County in which the property is located',
        'regionidcity':'City in which the property is located (if any)',
        'regionidzip':'Zip code in which the property is located',
        'regionidneighborhood':'Neighborhood in which the property is located',
        'roomcnt':'Total number of rooms in the principal residence',
        'storytypeid':'Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.).  See tab for details.',
        'typeconstructiontypeid':'What type of construction material was used to construct the home',
        'unitcnt':'Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)',
        'yardbuildingsqft17':'Patio in  yard',
        'yardbuildingsqft26':'Storage shed/building in yard',
        'yearbuilt'	:'The Year the principal residence was built ',
        'taxvaluedollarcnt':'The total tax assessed value of the parcel',
        'structuretaxvaluedollarcnt':'The assessed value of the built structure on the parcel',
        'landtaxvaluedollarcnt':'The assessed value of the land area of the parcel',
        'taxamount'	:'The total property tax assessed for that assessment year',
        'assessmentyear':'The year of the property tax assessment ',
        'taxdelinquencyflag':'Property taxes for this parcel are past due as of 2015',
        'taxdelinquencyyear':'Year for which the unpaid propert taxes were due '
        }
'''
# 开始一个一个研究咋办吧,从缺失样本较多的开始吧
# 计算每一个特征的数据丢失率
features = allData.columns
sampleNum = allData.shape[0]
featuresMissAccount = {}
for feature in features:
    missNum = allData[feature].isnull().sum(axis=0)
    missAccount = missNum / sampleNum
    featuresMissAccount[feature] = missAccount
# 24个特征的数据缺失率大于80%....一共才57个~~~~阿西吧~

def featureBasic(feature):
    # 特征的基本信息展示
    featureData = allData[feature]
    # 打印数据类型
    print ('\nfeature type is ' , featureData.dtype)
    
    print ('loss of data ratio is %.4f%%' % (featuresMissAccount[feature]*100))
    
    # 打印特征描述
    print ('\nfeature description :' ,dataDescription[feature])
    
    # 获取特征的数值分布
    valueCount = featureData.value_counts()
    print ('\nthis feature value count is :' ,len(valueCount))
    
    if len(valueCount) <= 100:
        print ('\nmaybe is a nominal category feature')
        sns.countplot(x=feature, data=allData)
        plt.ylabel('count',fontsize=12)
        plt.xlabel(feature,fontsize=12)
        plt.xticks(rotation='vertical')
        plt.show()
    else:
        # 打印最大值最小值
        print ('\nmaybe is a nominal numerical feature')
        maxvalue = featureData[~featureData.isnull()].max()
        minvalue = featureData[~featureData.isnull()].min()
        print ('\nthe feature max value is %s , min value is %s'%(maxvalue,minvalue))
        
        # 1、99百分位数
        ulimit = np.percentile(featureData.values,99)
        llimit = np.percentile(featureData.values,1)
        # 替换一下
        featureData.ix[featureData>ulimit] = ulimit
        featureData.ix[featureData<llimit] = llimit
        plt.hist(featureData[~featureData.isnull()])
        plt.show()
    del featureData ; gc.collect()
    
def corrcoefFeature(feature):
    # 展示特征和其他特征的相关性
    if allData[feature].dtype not in ['float64','int64']:
        print ('this feature can not do this')
        return None
    
    x_cols = [col for col in allData.columns if col not in [feature,'parcelid'] if allData[col].dtype=='float64']
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(allData[col].values,allData[feature].values)[0,1])
    
    # 转换成数据框
    corr_df = pd.DataFrame({'col_labels':labels,'corr_values':values})
    # 排序
    corr_df = corr_df.sort_values(by='corr_values')
    
    if corr_df.shape[0] == len(corr_df['corr_values'].isnull()):
        print ('this feature is so unique')
        return None
    else:
        print (corr_df)
        # 绘制一下看看
        ind = np.arange(len(labels))
        width = 0.9
        fig,ax = plt.subplots(figsize=(12,40))
        rects = ax.barh(ind,np.array(corr_df.corr_values.values))
        ax.set_yticks(ind)
        ax.set_yticklabels(corr_df.col_labels.values,rotation='horizontal')
        ax.set_xlabel('correlation coefficient')
        ax.set_title('variables')
        plt.show()


##### 开始吧 #####
# features[0]是parcelid，就是每个样本的id

### airconditioningtypeid
print ('Num.1 feature...')
# featureBasic(features[1])  # AirConditioningTypeID,家里冷却系统，1-13，代表不同的内容
# corrcoefFeature(features[1])  # 看看其余特征，与这个AirConditioningTypeID 的相关性
'''
    1——Central                  中央空调 
    2——Chilled Water            冷却水             数据中未出现
    3——Evaporative Cooler       蒸发冷却器
    4——Geo Thermal              地热               数据中未出现
    5——None                     没有
    6——Other                    其他               数据中未出现
    7——Packaged AC Unit         鬼知道是什么       数据中未出现
    8——Partial                  部分？             数据中未出现
    9——Refrigeration            制冷
    10——Ventilation             通风               数据中未出现
    11——Wall Unit               墙？
    12——Window Unit             窗？
    13——Yes                     有事啥？？？？
'''
# 缺失数据72.8154,特征应该标称型的，所以咋处理呢~~~
# 因为缺失的数据较多，所以增加一个吧，dummy_na = True主要是
dummy_airconditioningtypeid = pd.get_dummies(allData[features[1]],dummy_na=True,prefix='airconditioningtypeid')
allData = allData.join(dummy_airconditioningtypeid)
allData.drop([features[1]],axis=1,inplace=True)
del dummy_airconditioningtypeid ; gc.collect()


### architecturalstyletypeid
print ('Num.2 feature...')
# featureBasic(features[2])   #特征是分类型的，翻译出来是建筑风格
# corrcoefFeature(features[2])
# 1\数据丢失率99.79%
# 2\根据learnExploreData.ipynb这个文件中，xgboost和randomforest基于模型的特征重要性，均为排进前三十
# 3\生活经验告诉我们，建筑风格对房屋的价格影响也不是很重要的
# 基于三点删掉
allData.drop([features[2]],axis=1,inplace=True)


### basementsqft
print ('Num.3 feature...')
# featureBasic(features[3])           # 数值型特征，特征大致代表的意思是生活区域低于地面的面积?
# corrcoefFeature(features[3])        # 数据丢失率99.9455%
# 基于模型的选择，xgboost和随机森林都不是很重要，删掉
allData.drop([features[3]],axis=1,inplace=True)


### bathroomcnt
print ('Num.4 feature...')
# featureBasic(features[4])           # 数值型特征，浴室的数量，是一个灰常重要的特征了算是
# corrcoefFeature(features[4])        # 数据丢失率0.3840%
allData['bathroomcnt_isnull'] = allData[features[4]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[4]].fillna(allData[features[4]].median(axis=0),inplace=True)


### bedroomcnt
print ('Num.5 feature...')
# featureBasic(features[5])            # 卧室的数量，丢失的数量也不多
# corrcoefFeature(features[5])        # 但是没有卧室的如何理解呢。。。
allData['bedroomcnt_isnull'] = allData[features[5]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[5]].fillna(allData[features[5]].median(axis=0),inplace=True)


### buildingclasstypeid
print ('Num.6 feature...')
# featureBasic(features[6])           # 建筑原材料是什么的特征，分类型的
# corrcoefFeature(features[6])        # 丢失率99.5769% , 基于模型也是不太重要，删掉吧
allData.drop([features[6]],axis=1,inplace=True)


### buildingqualitytypeid 
print ('Num.7 feature...')
# featureBasic(features[7])           # 建筑物状况评估，越低代表越好
# corrcoefFeature(features[7])        # 数据丢失率35% , 基于模型还是挺重要的特征
# 看数据分布大部分都是在4和7，所以咋填充数据呢~~~12个类别，dummy吧
dummy_airconditioningtypeid = pd.get_dummies(allData[features[7]],dummy_na=True,prefix='buildingqualitytypeid')
allData = allData.join(dummy_airconditioningtypeid)
allData.drop([features[7]],axis=1,inplace=True)
del dummy_airconditioningtypeid ; gc.collect()


### calculatedbathnbr
print ('Num.8 feature...')
# featureBasic(features[8])           # 又是一个浴室相关的特征，不过包含fractional
# corrcoefFeature(features[8])          # 数据丢失率不高，4%，算是数值型吧，34个
# 其实这个特征的相关性应该和bathroomcnt高些 的啊！！！corrcoefFeature应该有点问题
allData['calculatedbathnbr_isnull'] = allData[features[8]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[8]].fillna(allData[features[8]].median(axis=0),inplace=True)


### decktypeid
print ('Num.9 feature...')
# featureBasic(features[9])       # 只有一个数值，是66，丢失率99%以上,删了吧
allData.drop([features[9]],axis=1,inplace=True)


### finishedfloor1squarefeet
print ('Num.10 feature...')
# featureBasic(features[10])      # 一层楼的居住面积？ 丢失率挺高的93%
allData['finishedfloor1squarefeet_isnull'] = allData[features[10]].apply(lambda x:1 if pd.isnull(x) else 0)
# 剩下的咋办呢~
ulimit_99 = np.percentile(allData.ix[~allData[features[10]].isnull(),features[10]],99)
# 数据变化区间在3-30000，但是百分之99的数据都在3710之内
llimit_1 = np.percentile(allData.ix[~allData[features[10]].isnull(),features[10]],1)
allData[features[10]].fillna((llimit_1+ulimit_99)/2,inplace=True)
del llimit_1,ulimit_99 ; gc.collect()


### calculatedfinishedsquarefeet
print ('Num.11 feature...')
# featureBasic(features[11])      # 总面积,这个丢失的概率不高，才1.86%
# corrcoefFeature(features[11])     #  max:952576.0 , min:1.0          
# 百分之98的数据在624——5302之间
ulimit_99 = np.percentile(allData.ix[~allData[features[11]].isnull(),features[11]],99)
llimit_1 = np.percentile(allData.ix[~allData[features[11]].isnull(),features[11]],1)
allData[features[11]].fillna(allData[features[11]].mean(),inplace=True)
allData['calculatedfinishedsquarefeet_big'] = allData[features[11]].apply(lambda x:1 if x < llimit_1 else 0)
allData['calculatedfinishedsquarefeet_medium'] = allData[features[11]].apply(lambda x:1 if x > llimit_1 and x < ulimit_99 else 0)
allData['calculatedfinishedsquarefeet_small'] = allData[features[11]].apply(lambda x:1 if x > ulimit_99 else 0)
del llimit_1,ulimit_99 ; gc.collect()


### finishedsquarefeet12
print ('Num.12 feature...')
# featureBasic(features[12])      # 生活区域的面积
# corrcoefFeature(features[12])     # 丢失率 9.24%    最大290345，最小1
allData[features[12]].fillna(allData[features[12]].mean(),inplace=True)


### finishedsquarefeet13
print ('Num.13 feature...')
# featureBasic(features[13])          # 周边的生活区域面积？
# corrcoefFeature(features[13])       # 丢失率99.7430%，删掉吧
allData.drop([features[13]],axis=1,inplace=True)

       
### finishedsquarefeet15
print ('Num.14 feature...')
# featureBasic(features[14])          # 丢失率93.61%
# corrcoefFeature(features[14])       # 总区域面积
ulimit_99 = np.percentile(allData.ix[~allData[features[14]].isnull(),features[14]],99)
feature14Data = allData[features[14]].dropna()
feature14Data = feature14Data[feature14Data <= ulimit_99]
plt.hist(feature14Data)
plt.show()
allData['finishedsquarefeet15_isnull'] = allData[features[14]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[14]].fillna(feature14Data.mean(),inplace=True)
del feature14Data,ulimit_99 ; gc.collect()


### finishedsquarefeet50
print ('Num.15 feature...')
# featureBasic(features[15])         # 又是一个和面积相关的数据
# corrcoefFeature(features[15])       # 丢失率93.21%
allData['finishedsquarefeet50_isnull'] = allData[features[15]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[15]].fillna(allData[features[15]].mean(),inplace=True)


### finishedsquarefeet6
print ('Num.16 feature...')
# featureBasic(features[16])          # 删掉吧丢失率99了
# corrcoefFeature(features[16])
allData.drop([features[16]],axis=1,inplace=True)


### fips
print ('Num.17 feature...')
# featureBasic(features[17])           # 邮政编码？还是什么码反正
# corrcoefFeature(features[17])        # 丢失的不多
dummy_fips = pd.get_dummies(allData[features[17]],dummy_na=True,prefix='fips')
allData = allData.join(dummy_fips)
allData.drop([features[17]],axis=1,inplace=True)
del dummy_fips ; gc.collect()


### fireplacecnt
print ('Num.18 feature...')
# featureBasic(features[18])         # 家里壁炉的数量
# corrcoefFeature(features[18])     # 丢失率89.53%
dummy_fireplacecnt = pd.get_dummies(allData[features[18]],dummy_na=True,prefix='fireplacecnt')
allData = allData.join(dummy_fireplacecnt)
allData.drop([features[18]],axis=1,inplace=True)
del dummy_fireplacecnt ; gc.collect()


### fullbathcnt
print ('Num.19 feature...')
# featureBasic(features[19])          # 完整的浴室数量，包含些其他的应该
# corrcoefFeature(features[19])     # 丢失率不高4.32% 
allData[features[19]] = allData[features[19]].apply(lambda x:np.float64(np.random.randint(1,4)) if pd.isnull(x) else x)


### garagecarcnt
print ('Num.20 feature...')
# featureBasic(features[20])          # 车库的数量
# corrcoefFeature(features[20])      # 丢失率70.5%
allData['garagecarcnt_isnull'] = allData[features[20]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[20]] = allData[features[20]].apply(lambda x:np.float64(round(abs(np.random.normal(loc=1.5)))) if pd.isnull(x) else x)


### garagetotalsqft  
print ('Num.21 feature...')
# featureBasic(features[21])            # 车库的面积
# corrcoefFeature(features[21])        # 和数量应该的丢失率一直的
# 所以这可以利用这两个数据做一个线性模型，来相互预测这个缺失的数据
allData['garagetotalsqft_isnull'] = allData[features[21]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[21]].fillna(allData[features[21]].mean(),inplace=True)


### hashottuborspa
print ('Num.22 feature...')
# featureBasic(features[22])        #删了~,就一个值，丢失率还超高
allData.drop([features[22]],axis=1,inplace=True)


### heatingorsystemtypeid
print ('Num.23 feature...')
# featureBasic(features[23])      # 供暖系统
# corrcoefFeature(features[23])    # 应该有24个变量，但是只出现了14类
dummy_heatingorsystemtypeid = pd.get_dummies(allData[features[23]],dummy_na=True,prefix='heatingorsystemtypeid')
allData = allData.join(dummy_heatingorsystemtypeid)
allData.drop([features[23]],axis=1,inplace=True)
del dummy_heatingorsystemtypeid ; gc.collect()


### latitude longitude      经纬度，一起来处理吧
print ('Num.24 and 25 feature...')
# featureBasic(features[24])        # 数据丢失也应该是一起丢的~
# featureBasic(features[25])
allData[features[24]].fillna(allData[features[24]].median(),inplace=True)
allData[features[25]].fillna(allData[features[25]].median(),inplace=True)


### lotsizesquarefeet 
print ('Num.26 feature...')
# featureBasic(features[26])          # 地皮面积？
allData['lotsizesquarefeet_isnull'] = allData[features[26]].apply(lambda x:1 if pd.isnull(x) else 0)
allData[features[26]].fillna(allData[features[26]].mean(),inplace=True)


### poolcnt
print ('Num.27 feature...')
# featureBasic(features[27])       # 是否有泳池
allData[features[27]].fillna(0,inplace=True)


### poolsizesum
print ('Num.28 feature...')
# featureBasic(features[28])        # 接下来的几个都是和游泳池有关的，比较尴尬的是丢失率挺高
allData[features[28]].fillna(0,inplace=True)


### pooltypeid10
print ('Num.29 feature...')
# featureBasic(features[29])          # 是否有什么yongchi?
allData[features[29]].fillna(0,inplace=True)


### pooltypeid2
print ('Num.30 feature...')
# featureBasic(features[30])               # 都填0吧就
allData[features[30]].fillna(0,inplace=True)


### pooltypeid7
print ('Num.31 feature...')
# featureBasic(features[31])                   # 都填0吧就
allData[features[31]].fillna(0,inplace=True)


### propertycountylandusecode                 # 下属的县的编码
print ('Num.32 feature...')
allData[features[32]].value_counts()          # 丢失率貌似还不高0.41%
# 本身是个分类特征，240个，有点多啊~~~~
dummy_propertycountylandusecode = pd.get_dummies(allData[features[32]],dummy_na=True,prefix='propertycountylandusecode')
allData = allData.join(dummy_propertycountylandusecode)
allData.drop([features[32]],axis=1,inplace=True)
del dummy_propertycountylandusecode ; gc.collect()


### propertylandusetypeid
print ('Num.33 feature...')
# featureBasic(features[33])       # 土地类型
dummy_propertylandusetypeid = pd.get_dummies(allData[features[33]],dummy_na=True,prefix='propertylandusetypeid')
allData = allData.join(dummy_propertylandusetypeid)
allData.drop([features[33]],axis=1,inplace=True)
del dummy_propertylandusetypeid ; gc.collect()


### propertyzoningdesc
print ('Num.34 feature...')
# 土地用途的说明....
allData.drop([features[34]],axis=1,inplace=True)


### rawcensustractandblock
print ('Num.35 feature...')
# featureBasic(features[35])        # 老实说，我真的想删掉它！！
allData[features[35]].fillna(allData[features[35]].mean(),inplace=True)


### regionidcity
print ('Num.36 feature...')
# featureBasic(features[36])          # 物业所在的城市，若有
allData[features[36]].fillna(0,inplace=True)


### regionidcounty
print ('Num.37 feature...')
# featureBasic(features[37])             # 物业所在的县
dummy_regionidcounty = pd.get_dummies(allData[features[37]],dummy_na=True,prefix='regionidcounty')
allData = allData.join(dummy_regionidcounty)
allData.drop([features[37]],axis=1,inplace=True)
del dummy_regionidcounty ; gc.collect()
            

### regionidneighborhood
print ('Num.38 feature...')
# featureBasic(features[38])               # 物业所在的街道
# 这就很尴尬这个特征....真的不知道咋处理~~~~~~~~~~~~
# 丢失数据61.26%  但是是个标称型的明显的。。。。
# 如果做的好的话，可以用经纬度来预测这个变量的
allData[features[38]].fillna(allData[features[38]].median(),inplace=True)


### regionidzip
print ('Num.39 feature...')
# featureBasic(features[39])                  # 邮编，缺失了0.4683
allData[features[39]].fillna(allData[features[39]].median(),inplace=True)


### roomcnt
print ('Num.40 feature...')
# featureBasic(features[40])      # 主要住所的房间数，丢失率倒是不高
# corrcoefFeature(features[40])
allData[features[40]].fillna(allData[features[40]].mean(),inplace=True)


### storytypeid
print ('Num.41 feature...')
# featureBasic(features[41])     # 多层房屋的地板类型（即地下室和主楼层，楼层，阁楼等）。 详见标签。
# 是一个分类特征，但是只有一个值是7，所以其他的就填0把
allData[features[41]] = allData[features[41]].apply(lambda x:1 if x == 7.0 else 0)


### threequarterbathnbr
print ('Num.42 feature...')
# featureBasic(features[42])     # 鬼知道是什么意思的特征~
# Number of 3/4 bathrooms in house (shower + sink + toilet)
dummy_threequarterbathnbr = pd.get_dummies(allData[features[42]],dummy_na=True,prefix='threequarterbathnbr')
allData = allData.join(dummy_threequarterbathnbr)
allData.drop([features[42]],axis=1,inplace=True)
del dummy_threequarterbathnbr ; gc.collect()


### typeconstructiontypeid
print ('Num.43 feature...')
# featureBasic(features[43])      # 用什么建筑材料建筑的,丢失率挺高的，另外和之前的一个特征应该类似的
# 基于模型去选择的话真的可以删掉的，因为丢失率在99%以上了
allData.drop([features[43]],axis=1,inplace=True)


### unitcnt
print ('Num.44 feature...')
# featureBasic(features[44])
allData[features[44]].fillna(allData[features[44]].median(),inplace=True)


### yardbuildingsqft17
print ('Num.45 feature...')
# featureBasic(features[45])        # 庭院？院子？,删了吧，缺失的太多
allData.drop([features[45]],axis=1,inplace=True)


### yardbuildingsqft26
print ('Num.46 feature...')
# featureBasic(features[46])
allData.drop([features[46]],axis=1,inplace=True)


### yearbuilt
print ('Num.47 feature...')
# featureBasic(features[47])
allData[features[47]].fillna(allData[features[47]].median(),inplace=True)


### numberofstories
print ('Num.48 feature...')
# featureBasic(features[48])
allData[features[48]].fillna(allData[features[48]].mean(),inplace=True)


### fireplaceflag
print ('Num.49 feature...')
# featureBasic(features[49])     # 是否发生过火灾？
allData[features[49]] = allData[features[49]].apply(lambda x:1 if x is True else 0)


### structuretaxvaluedollarcnt
print ('Num.50 feature...')
# featureBasic(features[50])        # 建筑的评估价值，算是重要的
allData[features[50]].fillna(allData[features[50]].mean(),inplace=True)


### taxvaluedollarcnt
print ('Num.51 feature...')
# featureBasic(features[51])       # 又是一个评估价值
allData[features[51]].fillna(allData[features[51]].mean(),inplace=True)


### assessmentyear
print ('Num.52 feature...')
# featureBasic(features[52])    #财产税评估年度
allData[features[52]].fillna(2015.0,inplace=True)


### landtaxvaluedollarcnt
print ('Num.53 feature...')
# featureBasic(features[53])   #包裹土地面积的评估价值
allData[features[53]].fillna(allData[features[53]].mean(),inplace=True)


### taxamount
print ('Num.54 feature...')
# featureBasic(features[54])
allData[features[54]].fillna(allData[features[54]].mean(),inplace=True)


### taxdelinquencyflag
print ('Num.55 feature...')
# featureBasic(features[55])
allData[features[55]] = allData[features[55]].apply(lambda x:1 if x is 'Y' else 0)


### taxdelinquencyyear
print ('Num.56 feature...')
# featureBasic(features[56])
dummy_taxdelinquencyyear = pd.get_dummies(allData[features[56]],dummy_na=True,prefix='taxdelinquencyyear')
allData = allData.join(dummy_taxdelinquencyyear)
allData.drop([features[56]],axis=1,inplace=True)
del dummy_taxdelinquencyyear ; gc.collect()


### censustractandblock
print ('Num.57 feature...')
# featureBasic(features[57])
allData[features[57]].fillna(allData[features[57]].mean(),inplace=True)

print (allData.shape)

for c, dtype in zip(allData.columns, allData.dtypes):	
    if dtype == np.float64:
        allData[c] = allData[c].astype(np.float32)

allData.to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyMyself.csv')

print ('Done!')

