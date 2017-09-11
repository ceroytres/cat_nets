# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:30:49 2017

@author: dmare
"""

import numpy as np


def sample_discrete_distribution(dist,shape = None):
    
    
    if shape is None:
        shape = (1,)
    
    if isinstance(shape,int):
        shape = tuple([shape])
    if isinstance(dist,list):
        dist = np.array(dist)
    dist/= dist.sum()
    K = dist.shape[0]
    
    z = np.random.gumbel(loc = 0.0, scale = 1.0, size = shape+(K,))
    dist = np.expand_dims(dist,0)
    
    y = np.log(dist)+z
    y = np.argmax(y,axis = -1)
    
    
    return y
