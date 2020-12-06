# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:45:09 2020

@author: Jon
"""
from numba import jit
import numpy as np
from burstInfer.get_adjusted import get_adjusted

@jit(nopython=True)
def compute_dynamic_F(state, length, W, K, ms2_coeff_flipped, count_reduction_manual):
    
    ms2_coeff = ms2_coeff_flipped # HACK
    
    ##print(datetime.datetime.now().time())
    trace_length = length
    
    state_flipped = K**W - state - 1
    
    adjust = get_adjusted(state_flipped, K, W, ms2_coeff)
    adjust_ones = adjust[0]
    adjust_zeros = adjust[1]
    
    F1_log = np.log(adjust_ones)
    F0_log = np.log(adjust_zeros)
    
    log_f0_terms = np.zeros((1, trace_length))
    for i in np.arange(0, trace_length):
        log_f0_terms[0,i] = F0_log
        
    log_f1_terms_saved = np.zeros((1, trace_length))
    for i in np.arange(0, trace_length):
        log_f1_terms_saved[0,i] = F1_log
    
    #log_f1_terms_saved2 = log_f1_terms_saved
    
    for t in np.arange(0,W-1):
        ##print('top')
        ##print(np.exp(log_f1_terms_saved[0,t]))
        ##print('bottom')
        ##print(count_reduction_manual[t,])
        ##print(abs(float(np.exp(log_f1_terms_saved[0,t])) - count_reduction_manual[t,]))
        inter = float(np.exp(log_f1_terms_saved[0,t])) - count_reduction_manual[t,]
        log_f1_terms_saved[0,t] = np.log(abs(inter[0,]))
        
    log_F_terms = []
    log_F_terms.append(log_f1_terms_saved)
    log_F_terms.append(log_f0_terms)
    
    ##print(datetime.datetime.now().time())
    return log_F_terms
