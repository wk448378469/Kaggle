# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:36:54 2017

@author: 凯风
"""

import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset import Dataset

print('loading...')

# 读取训练集和测试集的标称型数据
train_cat = Dataset.load_part('train','categorical')
test_cat = Dataset.load_part('test','categorical')

# 初始化数据，shape和上面的一样
train_cat_counts = np.zeros(train_cat.shape,dtype=np.float32)
test_cat_counts = np.zeros(test_cat.shape,dtype=np.float32)

# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
with tqdm(total=train_cat.shape[1],desc='Counting',unit='cols') as pbar:
    # 迭代，全部的标称型特征
    for col in range(train_cat.shape[1]):
        # 获取特征下全部的数据分布
        train_series = pd.Series(train_cat[:,col])
        test_series = pd.Series(test_cat[:,col])
        
        # 合并，获取特征下每个分类变量的出现的次数
        counts = pd.concat((train_series,test_series)).value_counts()
        
        # 新特征，根据每个样本的每个特征的变量，新增特征变量出现的次数
        train_cat_counts[:,col] = train_series.map(counts).values
        test_cat_counts[:,col] = test_series.map(counts).values
        
        pbar.update(1)

print('saving...')

# 保存特征集
Dataset.save_part_feature('categorical_counts',Dataset.get_part_feature('categorical'))
Dataset(categorical_counts = train_cat_counts).save('train')
Dataset(categorical_counts = test_cat_counts).save('test')




