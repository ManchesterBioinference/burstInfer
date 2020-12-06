# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:09:03 2020

@author: Jon
"""
import numpy as np
import scipy

from burstInfer.log_sum_exp import log_sum_exp
from burstInfer.v_log_solve import v_log_solve
from burstInfer.ms2_loading_coeff import ms2_loading_coeff
from numba import jit

#%%
@jit(nopython=True)
def get_adjusted(state, K, W, ms2_coeff):
    
    #ms2_coeff_flipped = np.flip(ms2_coeff_flipped, 1)
    ms2_coeff_flipped = ms2_coeff
    
    one_accumulator = 0
    zero_accumulator = 0
    for count in np.arange(0,W):
        #print(count)
        #print(state&1)
        if state & 1 == 1:
            #print('one')
            one_accumulator = one_accumulator + ms2_coeff_flipped[0,count]
        else:
            #print('zero')
            zero_accumulator = zero_accumulator + ms2_coeff_flipped[0,count]    
        state = state >> 1
        #print(state)
        
    return_list = []
    return_list.append(one_accumulator)
    return_list.append(zero_accumulator)
    
    return return_list



def get_single_cell_emission(K, W, kappa, posterior, signals):
    
    @jit(nopython=True)
    def compute_dynamic_F(state, length, W, K, ms2_coeff_flipped, count_reduction_manual):
        #print(datetime.datetime.now().time())
        trace_length = length
        
        state_flipped = K**W - state - 1
        
        adjust = get_adjusted(state_flipped, K, W, ms2_coeff)
        adjust_ones = adjust[0]
        adjust_zeros = adjust[1]
        
        F1_log = np.log(adjust_ones)
        F0_log = np.log(adjust_zeros)
        
        log_f0_terms = np.zeros((1, trace_length))
        for i in np.arange(0, trace_length):
            log_f0_terms[0,i] = F0_log
            
        log_f1_terms_saved = np.zeros((1, trace_length))
        for i in np.arange(0, trace_length):
            log_f1_terms_saved[0,i] = F1_log
        
        #log_f1_terms_saved2 = log_f1_terms_saved
        
        for t in np.arange(0,W-1):
            #print('top')
            #print(np.exp(log_f1_terms_saved[0,t]))
            #print('bottom')
            #print(count_reduction_manual[t,])
            #print(abs(float(np.exp(log_f1_terms_saved[0,t])) - count_reduction_manual[t,]))
            inter = float(np.exp(log_f1_terms_saved[0,t])) - count_reduction_manual[t,]
            log_f1_terms_saved[0,t] = np.log(abs(inter[0,]))
            
        log_F_terms = []
        log_F_terms.append(log_f1_terms_saved)
        log_F_terms.append(log_f0_terms)
        
        #print(datetime.datetime.now().time())
        return log_F_terms
    
    # MS2 coefficient calculation
    ms2_coeff = ms2_loading_coeff(kappa, W)
    ms2_coeff_flipped = np.flip(ms2_coeff, 1)
    count_reduction_manual = np.zeros((1,W-1))
    for t in np.arange(0,W-1):
        count_reduction_manual[0,t] = np.sum(ms2_coeff[0,t+1:])
    count_reduction_manual = np.reshape(count_reduction_manual, (W-1,1))
    
    posterior_traces = posterior
    signal_traces = signals
    
    length_container = []
    for i in np.arange(0, len(posterior_traces)): # TODO
        fetched_length = posterior_traces[i,:] # TODO
        fetched_length2 = fetched_length[~np.isnan(fetched_length)]
        length_container.append(len(fetched_length2))
        
    mu_container = []
    
    for p in np.arange(0, len(posterior_traces)): #TODO
        test_trace = posterior_traces[p,:] #TODO
        test_trace2 = np.reshape(test_trace, (len(test_trace), 1))
        test_trace2 = test_trace2[~np.isnan(test_trace2)]
        test_trace2 = np.reshape(test_trace2, (len(test_trace2),1))
        
        test_signal = signal_traces[p,:] #TODO
        test_signal2 = np.reshape(test_signal, (len(test_signal), 1))
        test_signal2 = test_signal2[~np.isnan(test_signal2)]
        test_signal2 = np.reshape(test_signal2, (len(test_signal2),1))
        
        
        
        fluo_logs_abs = np.log(np.abs(test_signal2))
        x_term_logs = fluo_logs_abs
        
        xsign = np.sign(test_signal2)
        x_term_signs = xsign
        
        v_b_terms_log = np.full((1, K), np.NINF)
        v_b_terms_sign = np.ones((1, K))
        
        
        #log_F_terms = F_dict[len(test_trace2)]
        
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
                    previous_state = F_state
                    
                    #result = log_F_terms[n][F_state,t] + log_F_terms[m][F_state,t]
                    result = compute_dynamic_F(F_state,length_container[p], W, K, ms2_coeff_flipped, count_reduction_manual)[n][0,t] + compute_dynamic_F(F_state,length_container[p], W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t]
                    terms_ith.append(result)
                    
                v_M_terms[m,n] = scipy.special.logsumexp(terms_ith)
        
        
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
                
                #terms_b_log_ith.append(x_term_logs[t,0] + log_F_terms[m][F_state,t])
                terms_b_log_ith.append(x_term_logs[t,0] + compute_dynamic_F(F_state,length_container[p], W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t])
                sign_list.append(x_term_signs[t,0])
                
                
            reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
            reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
            signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
            signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
            assign1 = reshaped2
            assign2 = signs2
            tmp = log_sum_exp(assign1, assign2)
            v_b_terms_log[0,m] = tmp[0,]
            v_b_terms_sign[0,m] = tmp[1,]
            
        #print(np.exp(v_log_solve(v_M_terms, np.ones((K,K)), v_b_terms_log, v_b_terms_sign)))
        mu_in = np.exp(v_log_solve(v_M_terms, np.ones((K,K)), v_b_terms_log, v_b_terms_sign))
        mu_container.append(mu_in[0,1])

#%%
    mu_array = np.array(mu_container)
    
    return mu_array
