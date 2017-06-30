# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:35:21 2017

@author: 凯风
"""

import numpy as np

from scipy.stats.mstats import rankdata
from scipy.special import erfinv
from sklearn.preprocessing import scale,minmax_scale
from tqdm import tqdm
from dataset import Dataset

print('loading...')

# 读取数值型数据
train_num = Dataset.load_part('train','numeric')
test_num = Dataset.load_part('test','numeric')

# 初始化
train_num_encode = np.zeros(train_num.shape,dtype=np.float32)
test_num_encode = np.zeros(test_num.shape,dtype=np.float32)

with tqdm(total=train_num.shape[1], desc=' Transforming ' ,unit='cols') as pbar:
    # 遍历数值型特征
    for col in range(train_num.shape[1]):
        # 每个特征下数据合并
        values = np.hstack((train_num[:,col],test_num[:,col]))
        # 每个特征下数据进行排名
        values = rankdata(values).astype(np.float64)
        # 标准化到(-1，1)范围内
        values = minmax_scale(values,feature_range=(-0.999,0.999))
        # 高斯分布,erfinv-反误差函数
        values = scale(erfinv(values))
        # import matplotlib.pyplot as plt
        # plt.hist(values)
        
        # 保存到初始化的数据中
        train_num_encode[:,col] = values[:train_num.shape[0]]
        test_num_encode[:,col] = values[train_num.shape[0]:]
        
        pbar.update(1)
    
print ('saving...')
# 保存特征及数据
Dataset.save_part_feature('numeric_rank_norm',Dataset.get_part_feature('numeric'))
Dataset(numeric_rank_norm=train_num_encode).save('train')
Dataset(numeric_rank_norm=test_num_encode).save('test')

print('Done...')
