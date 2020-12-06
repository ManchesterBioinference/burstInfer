# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:38:06 2020

@author: Jon
"""
import numpy as np
import scipy
from scipy import linalg

def calculate_single_cell_transition_rates(max_posterior_list, deltaT, A_parameter):
    
    # old style kon / koff 
    off_on = []
    on_off = []
    
    off_off = []
    on_on = []
    
    length_holder = []
    
    for j in np.arange(0, len(max_posterior_list)):
        fetched_posterior = max_posterior_list[j]
        fetched_posterior2 = fetched_posterior[~np.isnan(fetched_posterior)]
        off_on_interior = 0
        on_off_interior = 0
        length_holder.append(len(fetched_posterior2)-1)
        
        off_off_interior = 0
        on_on_interior = 0
        for k in np.arange(1, len(fetched_posterior2)):
            digit = fetched_posterior2[k,]
            digit_m1 = fetched_posterior2[k-1]
        
            if digit == 1 and digit_m1 == 0:
                off_on_interior +=1
            elif digit == 0 and digit_m1 == 1:
                on_off_interior +=1
            elif digit == 0 and digit_m1 == 0:
                off_off_interior +=1
            elif digit == 1 and digit_m1 == 1:
                on_on_interior +=1
        
          
        off_on.append(off_on_interior)
        on_off.append(on_off_interior)
        off_off.append(off_off_interior)
        on_on.append(on_on_interior)
        
    
    on_on_array = np.array(on_on)
    on_off_array = np.array(on_off)
    off_on_array = np.array(off_on)
    off_off_array = np.array(off_off)
    
    on_on_sum = np.sum(on_on_array)
    on_off_sum = np.sum(on_off_array)
    off_on_sum = np.sum(off_on_array)
    off_off_sum = np.sum(off_off_array)
    
    n_trials = sum(length_holder)
    
    A = A_parameter # TODO
    on_on_frac = (on_on_sum / n_trials) * A
    on_off_frac = (on_off_sum / n_trials) * A
    off_on_frac = (off_on_sum / n_trials) * A
    off_off_frac = (off_off_sum / n_trials) * A
    
    
    real_kon = np.divide(off_on_array + off_on_frac, (off_off_array + off_on_array + off_off_frac + off_on_frac))
    real_koff = np.divide(on_off_array + on_off_frac, (on_on_array + on_off_array + on_off_frac + on_on_frac))

        
    artificial_transition_list = []
    art_trans_holder = []
    art_trans_processed_holder = []
    
    for i in np.arange(0, len(real_kon)):
        inner_list = []
        art_trans = np.zeros((2,2))
        ind_kon = real_kon[i,]
        ind_koff = real_koff[i,]
        art_trans[1,0] = ind_kon
        art_trans[0,0] = 1 - ind_kon
        art_trans[0,1] = ind_koff
        art_trans[1,1] = 1 - ind_koff
        
    
        inner_list.append((scipy.linalg.logm(art_trans) / deltaT) * 60)
        artificial_transition_list.append(inner_list)
        art_trans_processed_holder.append((scipy.linalg.logm(art_trans) / deltaT) * 60)
        art_trans_holder.append(art_trans)
        
    
    kon_rates = []
    koff_rates = []

    
    rate_problem_list = []
    for j in np.arange(0, len(artificial_transition_list)):
        #print(artificial_transition_list[j])
        if np.any(np.iscomplex(artificial_transition_list[j])) != True:
            trans_fetched = artificial_transition_list[j]
            #cell_tag = artificial_transition_list[j][0]
            kon_rates.append(trans_fetched[0][1,0])
            koff_rates.append(trans_fetched[0][0,1])
            #cell_tags.append(cell_tag[6,])
            #midline_tags.append(cell_tag[4,])
        else:
            rate_problem_list.append(j) # HACK
            
    rate_output = {}
    rate_output['Single Cell kon'] = kon_rates
    rate_output['Single Cell koff'] = koff_rates
    rate_output['Problematic Cells'] = rate_problem_list
    
    return rate_output