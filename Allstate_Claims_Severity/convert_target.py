# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:07:32 2017

@author: 凯风
"""

from scipy.stats import boxcox
import numpy as np


def norm_y(y):
    return boxcox(np.log1p(y), lmbda=0.7)

def norm_y_inv(y_bc):
    return np.expm1((y_bc * 0.7 + 1)**(1/0.7))
    
def log_y(y):
    return np.log(y)

def log_y_inv(y):
    return np.exp(y)

def log_ofs_y(y,ofs):
    return np.log(y + ofs)

def log_ofs_y_inv(yl,ofs):
    return np.clip(np.exp(yl) - ofs, 1.0, np.inf)

def pow_y(y,p):
    return y ** p

def pow_y_inv(y,p):
    return y ** (1 / p)


def pow_ofs_y(y, p, ofs):
    return (y + ofs) ** p

def pow_ofs_y_inv(y, p, ofs):
    return np.clip(y ** (1 / p) - ofs, 1.0, np.inf)
