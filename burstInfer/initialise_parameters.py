# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:53:59 2019

@author: Jon
"""
import numpy as np

def initialise_parameters(K, W, matrix_max, matrix_mean):
    
    v_init = np.ones((K, 1))
    v_init[0, 0] = np.random.uniform(0,0.1) * (2/(W)) * matrix_max
    v_init[1,0] = np.random.uniform(0.2,0.6) * (2/(W)) * matrix_max
    noise_init = np.random.uniform(0.2,0.8) * matrix_mean
    pi0_init = np.zeros((2,1))
    pi0_init[0,0] = np.random.uniform(0.2,0.8)
    pi0_init[1,0] = 1 - pi0_init[0,0]
    A_init = np.zeros((2,2))
    
# =============================================================================
#     for j in np.arange(0,K):
#         A_init[:,j] = np.random.gamma(shape=1,scale=1,size=(1,2))
#         A_init[:,j] = A_init[:,j] / np.sum(A_init[:,j])
# =============================================================================
    
    A_init = np.zeros((2,2))
    A_init[0,0] = np.random.uniform(0.2,0.8)
    A_init[1,0] = 1 - A_init[0,0]
    A_init[0,1] = np.random.uniform(0.2,0.8)
    A_init[1,1] = 1- A_init[0,1]
    
    lambda_init = -2 * np.log(noise_init)
    
    # Further initialisation
    pi0_log = np.log(pi0_init)
    A_temp = A_init
    A_log = np.log(A_temp)
    v = v_init
    lambda_log = lambda_init
    noise_temp = noise_init
    v_logs = np.log(abs(v_init))   
    
    print('v_init')
    print(v_init)
    print('A_init')
    print(A_init)
    print('noise_init')
    print(noise_init)
    
    parameter_dict = {}
    parameter_dict['pi0_log'] = pi0_log
    parameter_dict['A_temp'] = A_temp
    parameter_dict['A_log'] = A_log
    parameter_dict['v'] = v
    parameter_dict['lambda_log'] = lambda_log
    parameter_dict['noise_temp'] = noise_temp
    parameter_dict['v_logs'] = v_logs
    
    return parameter_dict
