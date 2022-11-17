# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:53:13 2019

@author: Jon
"""
import numpy as np

def ms2_loading_coeff(kappa,W):
    alpha = kappa
    coeff = np.ones((1,W), dtype=float)
    
    alpha_ceil = np.ceil(alpha)
    alpha_floor = np.floor(alpha)
    
    coeff[0:int(alpha_floor):1,0] = (np.linspace(1,int(alpha_floor), endpoint=True, num=int(alpha_floor)) - 0.5) / alpha
    
    coeff[0,int(alpha_ceil)-1] = (alpha_ceil - alpha) + (alpha**2 - (alpha_ceil-1)**2) / (2*alpha)
    
    return coeff
