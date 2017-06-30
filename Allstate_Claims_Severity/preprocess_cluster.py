# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:00:43 2017

@author: 凯风
"""

import numpy as np

from dataset import Dataset,vstack,hstack
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

np.random.seed(1314)
gamma = 1.0

print ('loading...')

train_num = Dataset.load_part('train','numeric')            # 训练集数值型特征数据
train_cat = Dataset.load_part('train','categorical_dummy')        # 训练集标称型特征数据

test_num = Dataset.load_part('test','numeric')              # 测试集数值型特征数据
test_cat = Dataset.load_part('test','categorical_dummy')          #测试集标称型特征数据

print ('Standard scale ...')
# 标准化数据
ss = StandardScaler()
train_num_ss = ss.fit_transform(train_num).astype(np.float64)
test_num_ss = ss.transform(test_num).astype(np.float64)

print ('combining data ...')
all_data = hstack((vstack((train_num_ss,test_num_ss)).astype(np.float32),vstack((train_cat,test_cat))))

for n_clusters in [25,50,75,100,200]:
    # 定义特征名称
    part_name = 'cluster_rbf_%d' % n_clusters
    
    # 创建模型实例
    kmeans = MiniBatchKMeans(n_clusters,random_state=17*n_clusters+11,n_init=5)
    # 训练
    kmeans.fit(all_data)
    
    print ('')
    # 将数据转换为群集距离空间
    cluster_rbf = np.exp(-gamma * kmeans.transform(all_data))
    
    print ('save cluster_rbf_%d ...' % n_clusters)
    # 保存特征集
    Dataset.save_part_feature(part_name,['cluster_rbf_%d_%d' % (n_clusters,i) for i in range(n_clusters)])
    # 保存数据
    Dataset(**{part_name:cluster_rbf[:train_num_ss.shape[0]]}).save('train')
    Dataset(**{part_name:cluster_rbf[:test_num_ss.shape[0]]}).save('test')

print ('Done')

