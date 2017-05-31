# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:03:09 2017

@author: kaifeng
"""

import pandas as pd
import numpy as np
from scipy.stats import skew     # 求峰度的
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error   
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt

df_train = pd.read_csv('C:/Users/carne/Desktop/train.csv')
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])   # 图看上去左偏
print (df_train['SalePrice'].skew())  # 偏移
print (df_train['SalePrice'].kurt())  # 峰度


# 看看saleprice和其他feature的关系
# 1、数值型
data = pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis = 1)
data.plot.scatter(x = 'GrLivArea',y = 'SalePrice', ylim = (0,800000)) # 貌似有点线性关系

data = pd.concat([df_train['SalePrice'],df_train['TotalBsmtSF']],axis = 1)
data.plot.scatter(x = 'TotalBsmtSF',y = 'SalePrice', ylim = (0,800000))  # 貌似也有点线性关系

# 2、分类型
data = pd.concat([df_train['SalePrice'],df_train['OverallQual']],axis = 1)
f,ax = plt.subplots(figsize = (8,6))
fig = sns.boxplot(x = 'OverallQual' , y = 'SalePrice' , data = data) # 箱型图哦
fig.axis(ymin = 0 , ymax = 800000)

data = pd.concat([df_train['SalePrice'],df_train['YearBuilt']],axis = 1)
f,ax = plt.subplots(figsize = (16,8))
fig = sns.boxplot(x = 'YearBuilt' , y = 'SalePrice' , data = data) # 箱型图哦
fig.axis(ymin = 0 , ymax = 800000)
plt.xticks(rotation = 90)  # 貌似新的更高些

# GrLivArea,TotalBsmtSF,OverallQual,YearBuilt似乎和价格有些线性关系
# 我们可以依次根据每个变量都来一遍....
# 当然以上的分析都是主观性的，还是来点客观的吧~

# 相关性矩阵
corrmat =df_train.corr() # 哇哦，这个吊啊
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)

# 有四个变量，两两之间的相关性特别的高分是：
# 1、TotalBsmtSF 和 1stFlrSF
# 2、GarageCars 和 GarageArea

# 把相关性矩阵打印出来看看
cols = corrmat.nlargest(10,'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='0.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()

# 对于上面的两两相关性，选择TotalBsmtSF\GarageCars，因为相关性更高
# TotRmsAbvGrd和GrLivArea也是变量中相关性比较高的一对

# 看看散点图
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(df_train[cols],size=2.5)
plt.show()


# 处理缺失数据吧
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent],axis = 1,keys=['total','percent'])
missing_data
'''
    1、当数据缺失超过一定限度的时候应该丢弃它，限度15%?
    2、Garage'sth'、Bsmt'sth'似乎丢失的都是一样的，所以保留一个主要数据，其余全部删除就好
    3、关于Electrical，缺失的不多，所以删除掉缺失的样本，这样是不是不太好...
'''

df_train = df_train.drop((missing_data[missing_data['total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

# 单变量分析(先标准化)
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print (low_range)
print (high_range) # high_range的值有点过高了，是不是可以定义为异常值呢？

# 双变量分析
data = pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis = 1)
data.plot.scatter(x = 'GrLivArea',y = 'SalePrice', ylim = (0,800000)) 
# 右边的两个数据点中，面积足够大，但是价格却很低，所以是边远地区的？
# 删除掉异常值吧

df_train.sort_values(by ='GrLivArea',ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

data = pd.concat([df_train['SalePrice'],df_train['TotalBsmtSF']],axis = 1)
data.plot.scatter(x = 'TotalBsmtSF',y = 'SalePrice', ylim = (0,800000)) 


#正态性

# saleprice的直方图和正太概率图
sns.distplot(df_train['SalePrice'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot = plt)

# 售价的并不是线性的，可以用对数变换解决
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot = plt) # 哇哦~

sns.distplot(df_train['GrLivArea'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot = plt)

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot = plt) # 哇哦

sns.distplot(df_train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
# 有很多0，不能进行转换，所以咋办呢，创造一个二元变量，忽略0？

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']),index = df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

df_train.loc[df_train['HasBsmt'] == 1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

# 同方差性
# 图像化是最好的做法
plt.scatter(df_train['GrLivArea'],df_train['SalePrice']) #与之前相比更有趣了~




#开始搞事
TARGET = 'SalePrice'
NFOLDS = 5        # 几个基模型
SEED = 0
NROWS = None
SUBMISSION_FILE = 'C:/Users/carne/Desktop/submission.csv'

train = pd.read_csv("C:/Users/carne/Desktop/train.csv")
test = pd.read_csv("C:/Users/carne/Desktop/test.csv")

ntrain = train.shape[0]
ntest = test.shape[0]

y_train = np.log(train[TARGET]+1)
train.drop([TARGET],axis = 1,inplace = True)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index   #找出feature中是数值的

skewed_feats = train[numeric_feats].apply(lambda x :skew(x.dropna())) #计算数值型的偏度
skewed_feats = skewed_feats[skewed_feats > 0.75]       #丢掉偏度太小的
skewed_feats = skewed_feats.index           

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])      # 把所有数值型的log

all_data = pd.get_dummies(all_data)  # one hot code

all_data = all_data.fillna(all_data.mean())     # 补全缺失数据用均值

x_train = np.array(all_data[:train.shape[0]])
x_text = np.array(all_data[train.shape[0]:])

kf = KFold(ntrain,n_folds=NFOLDS,shuffle=True,random_state=SEED) # 交叉验证用的,分成5份

class SklearnWrapper(object):
    # 把模型方法包装起来
    def __init__(self,clf,seed=0,params=None):
        params['random_state'] = seed
        self.clf = clf(**params)     # 把参数这个字段，用关键字参数的形式传给模型
    
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    
    def predict(self,x):
        return self.clf.predict(x)

class XgbWrapper(object):
    def __init__(self,seed=0,params = None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds',250)
    
    def train(self,x_train,y_train):
        dtrain = xgb.DMatrix(x_train,label=y_train)
        self.gbdt = xgb.train(self.param,dtrain,self.nrounds)
    
    def predict(self,x):
        return self.gbdt.predict(xgb.DMatrix(x))
    
def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_index,test_index) in enumerate(kf): # 把刚刚交叉的数据分别取出来
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.train(x_tr,y_tr)        # 用4份来训练
        oof_train[test_index] = clf.predict(x_te)       # 用一份来看预测结果
        oof_test_skf[i,:] = clf.predict(x_text)         # 把结果存到oof_test_skf中
    oof_test[:] = oof_test_skf.mean(axis = 0)
    return oof_train.reshape(-1,1) , oof_test.reshape(-1,1)

et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
        }

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}



rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.005
}

# 创建模型对象
xg = XgbWrapper(seed = SEED,params=xgb_params)
et = SklearnWrapper(clf =ExtraTreesRegressor,seed = SEED,params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

# 每个模型进行交叉验证
xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))

x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}±{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)
saleprice = np.exp(submission['SalePrice'])-1






