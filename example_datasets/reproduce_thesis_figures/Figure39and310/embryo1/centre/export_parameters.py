# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:05:49 2019

@author: Jon
"""
import numpy as np

def export_parameters(parameters_array):
    
    results_array = np.zeros((1,11))
    transitions = parameters_array['A']
    pi0i = parameters_array['pi0']
    mu = parameters_array['v']
    
    results_array[0,0] = parameters_array['EM seed']
    results_array[0,1] = transitions[0,0]
    results_array[0,2] = transitions[1,0]
    results_array[0,3] = transitions[0,1]
    results_array[0,4] = transitions[1,1]
    results_array[0,5] = pi0i[0,0]
    results_array[0,6] = pi0i[0,1]
    results_array[0,7] = mu[0,]
    results_array[0,8] = mu[1,]
    results_array[0,9] = parameters_array['noise']
    results_array[0,10] = parameters_array['logL']
    
    return results_array
    
    
# =============================================================================
#     results_array[0,0] = parameters_array['seed_setter']
#     results_array[0,1] = np.exp(A_log[0,0])
#     results_array[0,2] = np.exp(A_log[1,0])
#     results_array[0,3] = np.exp(A_log[0,1])
#     results_array[0,4] = np.exp(A_log[1,1])
#     results_array[0,5] = np.exp(pi0_log[0,0])
#     results_array[0,6] = np.exp(pi0_log[0,1])
#     results_array[0,7] = np.exp(v_logs[0,])
#     results_array[0,8] = np.exp(v_logs[1,])
#     results_array[0,9] = np.exp(noise_log)
#     results_array[0,10] = seed_setter
# =============================================================================
