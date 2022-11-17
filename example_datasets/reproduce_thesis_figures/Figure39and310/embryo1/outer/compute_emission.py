# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:22:14 2019

@author: Jon
"""
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.linalg import logm
from scipy import spatial
import itertools

from log_sum_exp import log_sum_exp
from v_log_solve import v_log_solve
from ms2_loading_coeff import ms2_loading_coeff
from compute_F import compute_F

def compute_emission(signals, posterior, t_MS2, deltaT, kappa, K, W):
    
    # Data re-arranging here
    
    signals_centre = signals[:,7:]
    centre_posterior2 = posterior
    
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
    
    
    length_container = []

    for i in np.arange(0, len(centre_posterior2)):#
        #fetched_length = centre_posterior2[i,:]#
        fetched_length = centre_posterior2[i] # HACK
        fetched_length2 = fetched_length[~np.isnan(fetched_length)]
        length_container.append(len(fetched_length2))
    
    unique_lengths = np.unique(length_container)
    
    F_dict = {}
    for lv in unique_lengths:
        #print(lv)
        F_dict[lv] = compute_F(lv, adjusted_ones, adjusted_zeros, K, W, count_reduction_manual)
    
    
    mu_container = []
    
    for p in np.arange(0, len(centre_posterior2)):#
        #test_trace = centre_posterior2[p,:]#
        test_trace = centre_posterior2[p] # HACK
        test_trace2 = np.reshape(test_trace, (len(test_trace), 1))
        test_trace2 = test_trace2[~np.isnan(test_trace2)]
        test_trace2 = np.reshape(test_trace2, (len(test_trace2),1))
        
        test_signal = signals_centre[p,:]#
        test_signal2 = np.reshape(test_signal, (len(test_signal), 1)) # log this
        test_signal2 = test_signal2[~np.isnan(test_signal2)]
        test_signal2 = np.reshape(test_signal2, (len(test_signal2),1))
        
        
        
        fluo_logs_abs = np.log(np.abs(test_signal2))
        x_term_logs = fluo_logs_abs
        
        xsign = np.sign(test_signal2)
        x_term_signs = xsign
        
        v_b_terms_log = np.full((1, K), np.NINF)
        v_b_terms_sign = np.ones((1, K))
        
        
        log_F_terms = F_dict[len(test_trace2)]
        
        first_state = test_trace2[0,0]
        previous_state = int(first_state)
        
        v_M_terms = np.zeros((2,2))
        
        mask = np.int32((2**W)-1)
        for m in np.arange(0,K):
            for n in np.arange(0,K):
                terms_ith = [] 
                for t in np.arange(0, len(test_trace2)):
                    cell = test_trace2[t,0]
                    if cell == 0:
                        F_state = np.bitwise_and(previous_state << 1, mask)
                    else:
                        F_state = np.bitwise_and((previous_state << 1) + 1, mask)
                    #print(F_state)
                    previous_state = F_state
                    
                    result = log_F_terms[n][F_state,t] + log_F_terms[m][F_state,t]
                    terms_ith.append(result)
                    
                v_M_terms[m,n] = scipy.special.logsumexp(terms_ith) # Do you need an extra concatentation in here, like the original?
        
        
        terms_b_log_ith = []
        sign_list = []
        tmp = np.ones((K,1))
        for m in np.arange(0,K):
            terms_b_log_ith = []
            sign_list = []
            for t in np.arange(0, len(test_trace2)):
                
                cell = test_trace2[t,0]
                if cell == 0:
                    F_state = np.bitwise_and(previous_state << 1, mask)
                else:
                    F_state = np.bitwise_and((previous_state << 1) + 1, mask)
                previous_state = F_state
                
                terms_b_log_ith.append(x_term_logs[t,0] + log_F_terms[m][F_state,t])
                #terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + log_F_terms[m][key,t])
                sign_list.append(x_term_signs[t,0])
                
                
            reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
            reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
            signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
            signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
            #assign1 = np.concatenate((np.reshape(np.array(v_b_terms_log[0,m]), (1,1)), reshaped2), axis = 1)
            #assign1 = reshaped
            #assign2 = np.concatenate((np.reshape(np.array(v_b_terms_sign[0,m]), (1,1)), signs2), axis = 1)
            #assign2 = reshaped2
            assign1 = reshaped2
            assign2 = signs2
            tmp = log_sum_exp(assign1, assign2)
            v_b_terms_log[0,m] = tmp[0,]
            v_b_terms_sign[0,m] = tmp[1,]
            
        #print(np.exp(v_log_solve(v_M_terms, np.ones((K,K)), v_b_terms_log, v_b_terms_sign)))
        mu_in = np.exp(v_log_solve(v_M_terms, np.ones((K,K)), v_b_terms_log, v_b_terms_sign))
        mu_container.append(mu_in[0,1])
    
    mu_array = np.array(mu_container)
        
    return mu_array
