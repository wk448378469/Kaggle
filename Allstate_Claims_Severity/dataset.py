# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:01:01 2017

@author: 凯风
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle 
import os

n_fold = 5
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache')

def hstack(x):
    if any(sp.issparse(p) for p in x):      # 判断是否是稀疏矩阵
        return sp.hstack(x,format='csr')    # 水平方向堆栈稀疏矩阵
    else:
        return np.hstack(x)                 # 水平方向堆栈矩阵
    
def vstack(x):
    if any(sp.issparse(p) for p in x):      # 判断是否是稀疏矩阵
        return sp.vstack(x,format='csr')    # 垂直方向堆栈稀疏矩阵
    else:
        return np.vstack(x)                 # 垂直方向堆栈矩阵
    
def save_pickle(filename,data):
    with open(filename,'wb') as f:          # 序列化对象的文件保存
        pickle.dump(data,f)
        
def load_pickle(filename):                  # 读取文件中的序列化对象
    with open(filename,'rb') as f:
        return pickle.load(f)
        
def save_csr(filename,array):
    # 多个数组保存为压缩的`.npz``格式的单个文件
    np.savez_compressed(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)
    
def load_csr(filename):
    loader = np.load(filename)
    # 读取文件里的数据
    return sp.csr_matrix((loader['data'],loader['indices'],loader['indptr']),shape=loader['shape'])

class Dataset(object):

    part_types = {
            # 特征总名称 : 类型
                # d1——只有一列
                # d2——包含多列
                # s2——多个，多列特征
            'id': 'd1',                             
            'loss': 'd1',
            'numeric': 'd2',
            'numeric_lin': 'd2',
            'numeric_scaled': 'd2',
            'numeric_boxcox': 'd2',
            'numeric_rank_norm': 'd2',
            'numeric_combinations': 'd2',
            'numeric_edges': 's2',
            'numeric_unskew': 'd2',
            'categorical': 'd2',
            'categorical_counts': 'd2',
            'categorical_encoded': 'd2',
            'categorical_dummy': 's2',
            'svd': 'd2',
            'manual': 'd2',
            'cluster_rbf_25': 'd2',
            'cluster_rbf_50': 'd2',
            'cluster_rbf_75': 'd2',
            'cluster_rbf_100': 'd2',
            'cluster_rbf_200': 'd2',
                }
    
    parts = part_types.keys()
    
    # classmethod是用来指定一个类的方法为类方法
    # 类方法既可以直接类调用Dataset.f()，也可以进行实例调用C().f()
    # 类方法的第一个参数cls，而实例方法的第一个参数是self
    
    @classmethod
    def save_part_feature(cls,part_name,features):
        # 保存特征名称，以part_types的key为集进行
        save_pickle('%s%s-features.pickle' % (cache_dir,part_name) , features)
        
    @classmethod
    def get_part_feature(cls,part_name):
        # 获取特征名称，以part_types的key为集进行
        return load_pickle('%s%s-features.pickle' % (cache_dir,part_name))
    
    @classmethod
    def load(cls,name,parts):
        # 读取part_types的key的特征的数据
        return cls(**{part_name:cls.load_part(name,part_name) for part_name in parts})
    
    @classmethod
    def load_part(cls,name,part_name):
        if cls.part_types[part_name][0] == 's':
            # 读取压缩数据
            return load_csr('%s%s-%s.npz' % (cache_dir,part_name,name))
        else:
            # 读取数据
            return np.load('%s%s-%s.npy' % (cache_dir,part_name,name))
        
    @classmethod
    def concat(cls,datasets):
        
        datasets = list(datasets)
        
        if len(datasets) == 0:
            return ValueError('empty')
        if len(datasets) == 1:
            return datasets[0]

        new_parts = {}
        for part_name in datasets[0].parts:
            new_parts[part_name] = pd.concat(part_name,[ds[part_name] for ds in datasets])
            
        return cls(**new_parts)
    
        
    def __init__(self,**parts):
        # 关键字参数parts
        # 是一个字典包含feature集和对应的数据
        # feature集是part_types的key
        self.parts = parts
    
    def __getitem__(self,key):
        # 魔法方法，允许对象使用像字典的操作
        return self.parts[key]
    
    
    def save(self,name):
        # 保存数据集( test or train or target or model predict)
        for part_name in self.parts:
            self.save_part(part_name,name)
            
    def save_part(self,part_name,name):
        # 依次保存数据集中的特征集
        if self.part_types[part_name][0] == 's':
            save_csr('%s%s-%s.npz' % (cache_dir,part_name,name),self.parts[part_name])
        else:
            np.save('%s%s-%s.npy' % (cache_dir,part_name,name),self.parts[part_name])
    
    def sliced(self,index):
        # 添加新的特征集
        new_parts= {}
        for part_name in self.parts:
            new_parts[part_name] = self.parts[part_name][index]
        
        return Dataset(**new_parts)
    
    
    
    
    
    