# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:51:55 2019

@author: Jon
"""
import numpy as np

def log_sum_exp(arr, signs):
    arr_max = np.max(arr[:,:])
    term2_array = np.multiply(signs, np.exp(arr-arr_max))
    term2 = np.sum(term2_array)
    logsum = np.array([arr_max + np.log(np.abs(term2)), np.sign(term2)])
    return logsum
