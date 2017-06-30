# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:23:09 2017

@author: 凯风
"""

import numpy as np

from dataset import Dataset
from sklearn.preprocessing import StandardScaler

print ('loading...')

# 读取数据
train_num = Dataset.load_part('train', 'numeric')
test_num = Dataset.load_part('test', 'numeric')

print ('scaling...')

# 标准化数据
ss = StandardScaler()
train_num = ss.fit_transform(train_num)
test_num = ss.transform(test_num)

# 合并数据
# all_scaled = np.vstack((train_num, test_num))

print ('saving...')

# 保存数据和特征
Dataset.save_part_feature('numeric_scaled', Dataset.get_part_feature('numeric'))
Dataset(numeric_scaled=train_num).save('train')
Dataset(numeric_scaled=test_num).save('test')

print ('done~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')