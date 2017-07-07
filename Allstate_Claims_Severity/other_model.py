# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:58:41 2017

@author: 凯风
"""

import pandas as pd
import numpy as np
from dataset import Dataset

stackingTrainData = pd.DataFrame(Dataset.load_part('train','new').tolist().toarray(),columns=range(14))
stackingTestData = pd.DataFrame(Dataset.load_part('test','new').tolist().toarray())
stackingTestData.drop(['11','12'], axis = 1, inplace = True)


trainTarget = Dataset.load_part('train','loss').reshape((-1,1))

# 模型一
# 对stackTestData的每一行求平均值
submissionDataAverage = stackingTestData.sum(axis=1)

# 模型二
# 计算stackingTrainData的每一列，与trainTarget比较，给stackTrainData的列名的成绩排序
# 根据排序赋予不同的权重，加权求和