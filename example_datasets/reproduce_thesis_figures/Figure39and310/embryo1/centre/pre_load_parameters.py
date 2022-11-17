# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:21:00 2019

@author: Jon
"""
import numpy as np

def pre_load_parameters():
    
    preloaded_parameters = {}
    preloaded_parameters['EM seed'] = 609660386
    preloaded_parameters['logL'] = -676.4833314175047
    preloaded_parameters['noise'] = 0.22079566786856117
    preloaded_A = np.ones((2,2))
    preloaded_A[0,0] = 0.888311
    preloaded_A[1,0] = 0.111689
    preloaded_A[0,1] = 0.095028
    preloaded_A[1,1] = 0.904972
    preloaded_pi0 = np.ones((1,2))
    preloaded_pi0[0,0] = 1
    preloaded_pi0[0,1] = 0
    preloaded_v = np.ones((2,1))
    preloaded_v[0,0] = 0.00548203
    preloaded_v[1,0] = 0.145519
    preloaded_parameters['A'] = preloaded_A
    preloaded_parameters['pi0'] = preloaded_pi0
    preloaded_parameters['v'] = preloaded_v
    
    return preloaded_parameters