# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:41:59 2017

@author: 凯风
"""

import numpy as np
from scipy.stats import skew,boxcox

from tqdm import tqdm
from dataset import Dataset

print ('loading...')
train_num = Dataset.load_part('train','numeric')            # 读取训练集数值型特征
test_num = Dataset.load_part('test','numeric')             # 读取测试集数值型特征

# 初始化数据
train_num_encode = np.zeros(train_num.shape,dtype=np.float32)
test_num_encode = np.zeros(test_num.shape,dtype=np.float32)

'''
    boxcox 说明:
        Box-Cox变换是统计建模中常用的一种数据变换，用于连续的响应变量不满足正态分布的情况。
        Box-Cox变换，变换之后，可以一定程度上减小不可观测的误差和预测变量的相关性。
        
        说人话就是把不符合正态分布的数据集，转化成大致符合正态分布。
        
    boxcox 公式：
              - (X^λ - 1)/λ      if λ != 0
        X^λ = |
              - ln(X)            if λ = 0
              
              PS. λ是待定变换参数，一般为λ=0,1/2,-1 ,scipy.stats默认λ是None
'''


with tqdm(total=train_num_encode.shape[1],desc='Transforming',unit='cols') as pbar:
    # 迭代每一个数值型特征
    for col in range(train_num_encode.shape[1]):
        # 获取特征下所有数据(包括测试集和训练集)
        values = np.hstack((train_num[:,col],test_num[:,col]))
        
        # 获取该特征下所有数据的不对称度程度
        sk = skew(values)
        
        if sk > 0.25:
            # box-cox处理
            values_enc,lam = boxcox(values+1) 
            train_num_encode[:,col] = values_enc[:train_num.shape[0]]
            test_num_encode[:,col] = values_enc[train_num.shape[0]:]
        else:
            # 不处理
            train_num_encode[:,col] = train_num[:,col]
            test_num_encode[:,col] = test_num[:,col]
            
        pbar.update(1)
        

print('saving...')

# 保存特征
Dataset.save_part_feature('numeric_boxcox',Dataset.get_part_feature('numeric'))
# 保存数据
Dataset(numeric_boxcox=train_num_encode).save('train')
Dataset(numeric_boxcox=test_num_encode).save('test')

print('Done')
        