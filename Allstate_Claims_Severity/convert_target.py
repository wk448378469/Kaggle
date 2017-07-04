# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:07:32 2017

@author: 凯风
"""

from scipy.stats import boxcox
import numpy as np

class norm(object):

    def forwardConversion(self, y):
        return boxcox(np.log1p(y), lmbda=0.7)
    
    def reverseConversion(self, y):
        return np.expm1((y * 0.7 + 1)**(1/0.7))


class log(object):    

    def forwardConversion(self,y):
        return np.log(y)
    
    def reverseConversion(self,y):
        return np.exp(y)


class log_ofs(object):
    
    def forwardConversion(self, y, ofs=200):
        return np.log(y + ofs)
    
    def reverseConversion(self, y, ofs=200):
        return np.clip(np.exp(y) - ofs, 1.0, np.inf)

class powed(object):
    
    def forwardConversion(self, y, p=0.5):
        return y ** p
    
    def reverseConversion(self, y, p=0.5):
        return y ** (1 / p)

class powed_ofs(object):

    def forwardConversion(self, y, p=0.5, ofs=200):
        return (y + ofs) ** p
    
    def reverseConversion(self, y, p=0.5, ofs=200):
        return np.clip(y ** (1 / p) - ofs, 1.0, np.inf)