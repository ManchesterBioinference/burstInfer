# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:30:19 2019

@author: Jon
"""
def calcObservationLikelihood_long(lambda_logF, noise_tempF, dataF, veef,
                              INPUT_STATE):
    
    adjusted_list = get_adjusted(INPUT_STATE, K, W)
    
    eta = 0.5 * (lambda_logF - np.log(2*np.pi)) - 0.5 * \
    (1 / noise_tempF**2) * (dataF - (adjusted_list[1] * veef[0, 0] \
    + adjusted_list[0] * veef[1, 0]))**2
    
    #print(eta)
    return eta
