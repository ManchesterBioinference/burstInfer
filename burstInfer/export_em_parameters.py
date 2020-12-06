# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:55:09 2020

@author: Jon
"""
import numpy as np
import pandas as pd

def export_em_parameters(parameters_array):
    
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
    
    results_df = pd.DataFrame(results_array)
    results_df.columns = ['Random Seed', 'p_off_off', 'p_off_on', 'p_on_off', 'p_on_on',
                    'pi0_on', 'pi0_off', 'mu_off', 'mu_on', 'noise', 'logL']
    
    return results_df
