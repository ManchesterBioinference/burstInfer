# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:46:08 2020

@author: Jon
"""
from numba import jit
import numpy as np

@jit(nopython=True)
def get_adjusted(state, K, W, ms2_coeff):
    
    #ms2_coeff_flipped = np.flip(ms2_coeff_flipped, 1)
    ms2_coeff_flipped = ms2_coeff
    
    one_accumulator = 0
    zero_accumulator = 0
    for count in np.arange(0,W):
        ##print(count)
        ##print(state&1)
        if state & 1 == 1:
            ##print('one')
            one_accumulator = one_accumulator + ms2_coeff_flipped[0,count]
        else:
            ##print('zero')
            zero_accumulator = zero_accumulator + ms2_coeff_flipped[0,count]    
        state = state >> 1
        ##print(state)
        
    return_list = []
    return_list.append(one_accumulator)
    return_list.append(zero_accumulator)
    
    return return_list
