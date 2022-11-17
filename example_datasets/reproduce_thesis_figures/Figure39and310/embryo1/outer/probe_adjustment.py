# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:48:36 2019

@author: Jon
"""
from ms2_loading_coeff import ms2_loading_coeff
from compute_F import compute_F
import numpy as np

def probe_adjustment(K, W, kappa, unique_lengths):
    # MS2 coefficient calculation
    ms2_coeff = ms2_loading_coeff(kappa, W) 
    count_reduction_manual = np.zeros((1,W-1))
    for t in np.arange(0,W-1):
        count_reduction_manual[0,t] = np.sum(ms2_coeff[0,t+1:])
    count_reduction_manual = np.reshape(count_reduction_manual, (W-1,1))
    just_states = np.zeros((K**W,1))
    for d in np.arange(0,K**W):
        just_states[d,0] = d
    
    new_states_table = np.zeros((K**W,W))
    controller = '0' + str(W) + 'b'
    for s in np.arange(0,K**W):
        focus_state = int(just_states[s,0])
        state_bin = format(focus_state, controller)
        for co in np.arange(0,W):
            new_states_table[s,co] = state_bin[co]
    
            
    ms2_coeff_flipped = np.flip(ms2_coeff, 1)
    
    adjusted_ones = np.full((K**W,1), np.NINF)
    adjusted_ones[0,0] = 0
    for b in np.arange(0,K**W):
        b_row = new_states_table[b,]
        finder = np.where(b_row == 1)
        finder_contents = finder[0]
        adjusted_list = []
        inner_list = []
        for c in np.arange(0,len(finder_contents)):
            if len(finder_contents) < 1:
                adjusted_ones[b,0] = 0
            elif len(finder_contents) == 1:
                adjusted_list.append(ms2_coeff_flipped[0,finder_contents])
                adjusted_ones[b,0] = adjusted_list[0]
            else:
                inner_list.append(ms2_coeff_flipped[0,finder_contents[c,]])
        if len(inner_list) > 0:
            adjusted_ones[b,0] = sum(inner_list)
    
    adjusted_zeros = np.flip(adjusted_ones,0)
    
    F_dict = {}
    for lv in unique_lengths:
        #print(lv)
        F_dict[lv] = compute_F(lv, adjusted_ones, adjusted_zeros, K, W, count_reduction_manual)
        
    output_list = []
    output_list.append(adjusted_ones)
    output_list.append(adjusted_zeros)
    output_list.append(F_dict)
    
    return output_list
