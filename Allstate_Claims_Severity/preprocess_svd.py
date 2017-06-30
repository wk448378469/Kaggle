# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 19:04:55 2017

@author: 凯风
"""

import numpy as np

from dataset import Dataset,vstack,hstack
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

n_component = 500

print ('loading...')

# 读取数据集
train_num = Dataset.load_part('train','numeric')
train_cat = Dataset.load_part('train','categorical_dummy')
test_num = Dataset.load_part('test','numeric')
test_cat = Dataset.load_part('test','categorical_dummy')

train_cnt = train_num.shape[0]

# 标准化
ss = StandardScaler()
train_num_ss = ss.fit_transform(train_num)
test_num_ss = ss.transform(test_num)

# 合并数据
all_data = hstack((vstack((train_num_ss, test_num_ss)).astype(np.float32), vstack((train_cat, test_cat))))

# 删掉
del train_num,train_cat,test_num,test_cat

# svd初始化对象
svd = TruncatedSVD(n_components=n_component)
res = svd.fit_transform(all_data)
# print (np.sum(svd.explained_variance_ratio_))
# 500个特征能解释99.86%的方差

print ('saving..')
Dataset.save_part_feature('svd', ['svd%d' % i for i in range(n_component)])
Dataset(svd=res[:train_cnt]).save('train')
Dataset(svd=res[train_cnt:]).save('test')
print ('Done.................')

