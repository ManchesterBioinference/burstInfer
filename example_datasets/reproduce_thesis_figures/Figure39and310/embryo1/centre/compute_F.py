# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:53:32 2019

@author: Jon
"""
import numpy as np

def compute_F(length, adjusted_ones, adjusted_zeros, K, W, count_reduction_manual):
    trace_length = length
    F0_log = np.log(adjusted_zeros).flatten()
    F1_log = np.log(adjusted_ones).flatten()
    
    log_f0_terms = np.zeros((K**W, trace_length))
    for i in np.arange(0, trace_length):
        log_f0_terms[:,i] = F0_log
           
    log_f1_terms_saved = np.zeros((K**W, trace_length))
    for i in np.arange(0, trace_length):
        log_f1_terms_saved[:,i] = F1_log
    
    log_f1_terms_saved = np.flip(log_f1_terms_saved, axis = 0)
    for t in np.arange(0,W-1):
        log_f1_terms_saved[:,t] = np.log(abs(np.exp(log_f1_terms_saved[:,t]) - count_reduction_manual[t,]))
    
    log_f1_terms = log_f1_terms_saved
    log_f0_terms = np.flip(log_f0_terms, axis = 0)
    
    log_F_terms = []
    log_F_terms.append(log_f1_terms)
    log_F_terms.append(log_f0_terms)
    
    return log_F_terms
