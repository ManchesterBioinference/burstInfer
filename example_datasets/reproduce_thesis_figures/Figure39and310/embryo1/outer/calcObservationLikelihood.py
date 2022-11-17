# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:52:49 2019

@author: Jon
"""
import numpy as np

def calcObservationLikelihood(lambda_logF, noise_tempF, dataF, veef,
                              INPUT_STATE, adjusted_zeros, adjusted_ones):
    
    eta = 0.5 * (lambda_logF - np.log(2*np.pi)) - 0.5 * \
    (1 / noise_tempF**2) * (dataF - (adjusted_zeros[INPUT_STATE,0] * veef[0, 0] \
    + adjusted_ones[INPUT_STATE,0] * veef[1, 0]))**2

    return eta
