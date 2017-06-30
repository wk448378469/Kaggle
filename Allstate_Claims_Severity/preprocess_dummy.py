# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:01:03 2017

@author: 凯风
"""

import scipy.sparse as sp
import numpy as np

from tqdm import tqdm
from dataset import Dataset

# 最小频次
min_freq = 10

print('loading...')

# 读取训练集和测试集的标称型数据
train_cat = Dataset.load_part('train','categorical')
test_cat = Dataset.load_part('test','categorical')

# 初始化数据
train_cat_encode = []
test_cat_encode = []

# 获取全部标称型数据的name
cats = Dataset.get_part_feature('categorical')
feature = []

with tqdm(total=len(cats),desc='Encoding',unit='cols') as pbar:
    # 迭代每一个分类特征
    for col, cat in enumerate(cats):
        # 获取特征下，不同值及其频次
        value_counts = dict(zip(*np.unique(train_cat[:,col],return_counts=True)))
        
        train_rares = np.zeros(train_cat.shape[0],dtype=np.uint8)
        test_rares = np.zeros(test_cat.shape[0],dtype=np.uint8)
        
        # 迭代每一个特征下的每一个值
        for val in value_counts:
            # 判断该值是否大于最小频次
            if value_counts[val] >= min_freq:
                feature.append('%s_%s' % (cat,val)) # 添加特征
                # 添加特征数据
                train_cat_encode.append(sp.csr_matrix((train_cat[:, col] == val).astype(np.uint8).reshape((train_cat.shape[0], 1))))
                test_cat_encode.append(sp.csr_matrix((test_cat[:, col] == val).astype(np.uint8).reshape((test_cat.shape[0], 1))))
            else:
                # 小于最小频次，则转化为不进行dummy编码
                # 采用的是类似于pd.factorize的方法
                train_rares += (train_cat[:,col]==val).astype(np.uint8)
                test_rares += (test_cat[:,col]==val).astype(np.uint8)
            
        if train_rares.sum() > 0 and test_rares.sum() > 0:
            feature.append('%s_rare' % cat)
            train_cat_encode.append(sp.csr_matrix(train_rares.reshape((train_cat.shape[0],1))))
            test_cat_encode.append(sp.csr_matrix(test_rares.reshape((test_cat.shape[0],1))))
        
        pbar.update(1)

print('new feature numbers:',len(feature))        

print ('saving...')

# 所有特征进行在水平方向合并
train_cat_encode = sp.hstack(train_cat_encode,format='csr')
test_cat_encode = sp.hstack(test_cat_encode,format='csr')

# 保存特征及数据
Dataset.save_part_feature('categorical_dummy',feature)
Dataset(categorical_dummy = train_cat_encode).save('train')
Dataset(categorical_dummy = test_cat_encode).save('test')




