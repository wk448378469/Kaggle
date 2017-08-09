# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:48:42 2017

@author: 凯风
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import copy
import pandas as pd
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

class rfImputer(object):
    def __init__(self, data):
        self.data = data
        self.missing = self.find_missing()              # 捕获缺失数据
        self.prop_missing = self.prop_missing()         # 每一列数据的缺失比重
        self.columns = data.columns                     # 特征信息
        self.col_types = self.detect_dtype()           # 判断列的数值属性
        self.imputed_values = {}                        # 初始化，用来保存填充数据的

    def detect_dtype(self):
        # 判断数据类型的
        dtypes = {}
        
        # 迭代每一列
        for x in self.columns:
            if self.data[x].dtype == 'float32':
                if len(x.unique()) < 20:
                    dtype[x] = 'classification'
                else:
                    dtype[x] = 'regression'
            elif self.data[x].dtype == 'object':
                dtype[x] = 'classification'
            elif self.data[x].dtype == 'int64':
                if len(self.data[x].unique()) < 20:
                    dtype[x] = 'classification'
                else:
                    dtype[x] = 'regression'
            else:
                msg = 'Unrecognized data type: %s' %x.dtype
                raise ValueError(msg)
        
        return dtypes


    def find_missing(self):
        # 返回每一列的缺失数据
        missing = {}
        for var in self.columns:
            col = self.data[var]
            missing[var] = col.index[col.isnull()]

        return missing

    def prop_missing(self):
        # 返回缺失数据的列的缺失比重
        out = {}
        n = self.data.shape[0]
        for var in self.columns:
            out[var] = float(len(self.missing[var])) / float(n)
        return out
        
    
    def mean_mode_impute(self, var):
        
        if self.col_types[var] == 'regression':
            # 回归用均值
            statistic = self.data[var].mean()
        elif self.col_types[var] == 'classification':
            # 分类用众数
            statistic = self.data[var].mode()[0]
        else:
            raise ValueError('Unknown data type')
        
        # 根据每列的缺失情况返回，填充值*缺失数的数组
        out = np.repeat(statistic, repeats = len(self.missing[var]))
        return out

                    
    def rf_impute(self, impute_var, data_imputed):

        y = data_imputed[impute_var]                            # 将传入的列定义为目标变量
        include = [x for x in self.columns if x != impute_var]  # 将除去传入的列外的特征作为输入特征
        X = data_imputed[include]

        if self.col_types[impute_var] == 'classification':
            # 训练模型
            rf = RandomForestClassifier(n_estimators = 20, oob_score = True, n_jobs=2,)
            rf.fit(y = y, X = X)
            # oob_decision_function_ : array of shape = [n_samples, n_classes]
            # oob - out of bag 外包，随机森林是从数据中抽样进行训练的
            # 获取y的全部的预测分类结果，选最大的~
            oob_predictions = np.argmax(rf.oob_decision_function_, axis = 1)
            # 获取y中，对应原数据集中的缺失数据的值
            oob_imputation = oob_predictions[self.missing[impute_var]]
            
        else:
            rf = RandomForestRegressor(n_estimators = 20, oob_score = True, n_jobs=2,)
            rf.fit(y = y, X = X)
            oob_imputation = rf.oob_prediction_[self.missing[impute_var]]
        
        # 更新填充数据
        self.imputed_values[impute_var] = oob_imputation


    def get_divergence(self, imputed_old):
        # 初始化参数
        div_cat = 0
        norm_cat = 0
        div_cont = 0
        norm_cont = 0
        
        for var in self.imputed_values:             # 迭代填充数据
            if self.col_types[var] == 'regression':
                # △ = ∑(xold - xnew)^2 / ∑(xnew)^2
                div = imputed_old[var] - self.imputed_values[var]     # 两次填充值之差  
                div_cont += div.dot(div)                            # 乘积求和      
                norm_cont += self.imputed_values[var].dot(self.imputed_values[var])  # 乘积求和
            elif self.col_types[var] == 'classification':
                # △ = ∑(I(xold != xnew)) / ∑(xnew)^2
                div = [1 if old != new
                       else 0
                       for old, new in zip(imputed_old[var], self.imputed_values[var])]
                div_cat += sum(div)
                norm_cat += len(div)

        if norm_cat == 0:
            cat_out = 0
        else:
            cat_out = div_cat / norm_cat
        if norm_cont == 0:
            cont_out = 0
        else:
            cont_out = div_cont / norm_cont
        
        return cat_out, cont_out
    

    def impute(self):
        
        print ("Starting Random Forest Imputation")

        for var in self.data.columns:
            # 先利用正常的方法进行数据填充
            self.imputed_values[var] = self.mean_mode_impute(var)
            
        # 初始化参数
        div_cat = float('inf')
        div_cont = float('inf')
        stop = False
        i = 0
        #　搞事~
        while not stop:
            i += 1
            print ("Iteration %d:" %i)

            # 保存上一次迭代的结果
            imputations_old = copy.copy(self.imputed_values)       # 填充的数据
            div_cat_old = div_cat                                  # 标称型特征散度
            div_cont_old = div_cont                                # 数值型特征散度

            # 对原数据集进行缺失值填充，并复制，利用imputed_values
            data_imputed = self.imputed_df()
            
            for var in self.columns:
                # 利用随机森林对每一列进行缺失值预测
                self.rf_impute(var, data_imputed)
            
            # 获取散度
            # 一种评价标准，具体可查看：http://bioinformatics.oxfordjournals.org/content/28/1/112.short
            div_cat, div_cont = self.get_divergence(imputations_old)
            print ("Categorical divergence: %f" %div_cat)
            print ("Continuous divergence: %f" %div_cont)

            # 检查是否满足停止条件~
            if div_cat >= div_cat_old and div_cont >= div_cont_old:
                stop = True


    def imputed_df(self):

        if len(self.imputed_values) == 0:
            raise ValueError('No imputed values available. Call impute() first')
        
        out_df = self.data.copy()
        
        for var in self.columns:
            # 对每一列进行填充缺失数据
            out_df[var].iloc[self.missing[var]] = self.imputed_values[var]

        return out_df


imp_df = rfImputer(allData)
imp_df.impute('random_forest')

imp_df.imputed_df().to_csv('D:/mygit/Kaggle/Zillow_Home_Value_Prediction/newFeaturesbyRFImputer.csv')