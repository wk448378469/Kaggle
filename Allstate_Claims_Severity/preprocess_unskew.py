# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:39:29 2017

@author: 凯风
"""

import numpy as np
import pandas as pd

from dataset import Dataset
from sklearn.preprocessing import minmax_scale

print ('loading...')

# 读取数值型数据
train_num = Dataset.load_part('train', 'numeric')
test_num = Dataset.load_part('test', 'numeric')

# 合并~垂直方向
numeric = pd.DataFrame(np.vstack((train_num, test_num)), columns=Dataset.get_part_feature('numeric'))

df = pd.DataFrame(index=numeric.index)

# newnewnew preprocess      我要疯了估计是
df['cont1'] = np.sqrt(minmax_scale(numeric['cont1']))
df['cont4'] = np.sqrt(minmax_scale(numeric['cont4']))
df['cont5'] = np.sqrt(minmax_scale(numeric['cont5']))
df['cont8'] = np.sqrt(minmax_scale(numeric['cont8']))
df['cont10'] = np.sqrt(minmax_scale(numeric['cont10']))
df['cont11'] = np.sqrt(minmax_scale(numeric['cont11']))
df['cont12'] = np.sqrt(minmax_scale(numeric['cont12']))
df['cont6'] = np.log(minmax_scale(numeric['cont6']) + 0000.1)
df['cont7'] = np.log(minmax_scale(numeric['cont7']) + 0000.1)
df['cont9'] = np.log(minmax_scale(numeric['cont9']) + 0000.1)
df['cont13'] = np.log(minmax_scale(numeric['cont13']) + 0000.1)
df['cont14'] = (np.maximum(numeric['cont14'] - 0.179722, 0) / 0.665122) ** 0.25

print ('saving...')

# 保存数据及特征
Dataset.save_part_feature('numeric_unskew', list(df.columns))
Dataset(numeric_unskew=df.values[:train_num.shape[0]]).save('train')
Dataset(numeric_unskew=df.values[train_num.shape[0]:]).save('test')

print ('!@#@#@#$@$!@#!@!#!@#!##$%^&^&*^&* over')