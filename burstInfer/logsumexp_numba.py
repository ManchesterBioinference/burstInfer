# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:47:03 2020

@author: Jon
"""
from numba import jit
import numpy as np

@jit(nopython=True)
def logsumexp_numba(X):
    r = 0.0
    for x in X:
        r += np.exp(x)  
    return np.log(r)
