# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:42:01 2020

@author: Jon
"""
from numba import jit
import numpy as np
from burstInfer.get_adjusted import get_adjusted

@jit(nopython=True)
def calcObservationLikelihood(lambda_logF, noise_tempF, dataF, veef,
                              INPUT_STATE, K, W, ms2_coeff_flipped):
    
    ms2_coeff = ms2_coeff_flipped # HACK
    
    adjusted_list = get_adjusted(INPUT_STATE, K, W, ms2_coeff)
    
    eta = 0.5 * (lambda_logF - np.log(2*np.pi)) - 0.5 * \
    (1 / noise_tempF**2) * (dataF - (adjusted_list[1] * veef[0, 0] \
    + adjusted_list[0] * veef[1, 0]))**2
    
    ##print(eta)
    return eta
