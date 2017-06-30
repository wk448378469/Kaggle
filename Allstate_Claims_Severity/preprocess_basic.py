# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:58:41 2017

@author: 凯风
"""

import pandas as pd
import numpy as np
from dataset import Dataset

for name in ['train','test']:
    print ('loading original %s data...' % name)
    data = pd.read_csv('D:/mygit/Kaggle/Allstate_Claims_Severity/%s.csv.zip' % name)
    
    if name == 'train':
        cat_columns = [c for c in data.columns if c.startswith('cat')]   # 标称型特征
        num_columns = [c for c in data.columns if c.startswith('cont')]  # 数值型特征
        
        # 调用Dataset的方法保存特征名称
        Dataset.save_part_feature('categorical',cat_columns)
        Dataset.save_part_feature('numeric',num_columns)
    
    # 调用Dataset方法保存特征数据 
    Dataset(categorical=data[cat_columns].values).save(name)
    Dataset(numeric=data[num_columns].values.astype(np.float32)).save(name)
    
    # 调用Dataset方法保存目标变量
    if 'loss' in data.columns:
        Dataset(loss=data['loss']).save(name)

print('done.')