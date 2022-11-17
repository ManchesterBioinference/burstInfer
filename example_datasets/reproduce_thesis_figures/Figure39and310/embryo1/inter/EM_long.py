# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:14:33 2019

@author: Jon
"""
import numpy as np
import scipy

from ms2_loading_coeff import ms2_loading_coeff
#from calcObservationLikelihood_long import calcObservationLikelihood_long
from v_log_solve import v_log_solve
from log_sum_exp import log_sum_exp

def calcObservationLikelihood_long(lambda_logF, noise_tempF, dataF, veef,
                              INPUT_STATE, K, W, ms2_coeff_flipped):
    
    adjusted_list = get_adjusted(INPUT_STATE, K, W, ms2_coeff_flipped)
    
    eta = 0.5 * (lambda_logF - np.log(2*np.pi)) - 0.5 * \
    (1 / noise_tempF**2) * (dataF - (adjusted_list[1] * veef[0, 0] \
    + adjusted_list[0] * veef[1, 0]))**2
    
    #print(eta)
    return eta

def get_adjusted(state, K, W, ms2_coeff_flipped):
    
    new_states_table = np.zeros((1,W))
    controller = '0' + str(W) + 'b'
    focus_state = int(state)
    state_bin = format(focus_state, controller)
    for co in np.arange(0,W):
        new_states_table[0,co] = state_bin[co]    
    
    #print(new_states_table)
    adjusted_ones = 0
    #adjusted_ones = np.full((K**W,1), np.NINF)
    #adjusted_ones[0,0] = 0
    #for b in np.arange(0,K**W):
    b_row = new_states_table
    b_row = np.reshape(new_states_table, (W,1))
    finder = np.where(b_row == 1)
    #print(finder)
    finder_contents = finder[0]
    #print(finder_contents)
    adjusted_list = []
    inner_list = []
    for c in np.arange(0,len(finder_contents)):
        if len(finder_contents) < 1:
            adjusted_ones = 0
        elif len(finder_contents) == 1:
            adjusted_list.append(ms2_coeff_flipped[0,finder_contents])
            adjusted_ones = adjusted_list[0]
        else:
            inner_list.append(ms2_coeff_flipped[0,finder_contents[c,]])
    if len(inner_list) > 0:        
        adjusted_ones = sum(inner_list)
        
    #print(adjusted_ones)
    #print(isinstance(adjusted_ones, np.ndarray))
    if isinstance(adjusted_ones, np.ndarray) == True:
        adjusted_ones = adjusted_ones[0,]
    
    state0 = K**W - state - 1
    
    new_states_table = np.zeros((1,W))
    controller = '0' + str(W) + 'b'
    focus_state = int(state0)
    state_bin = format(focus_state, controller)
    for co in np.arange(0,W):
        new_states_table[0,co] = state_bin[co]    
    
    #print(new_states_table)
    adjusted_zeros = 0
    #adjusted_ones = np.full((K**W,1), np.NINF)
    #adjusted_ones[0,0] = 0
    #for b in np.arange(0,K**W):
    b_row = new_states_table
    b_row = np.reshape(new_states_table, (W,1))
    finder = np.where(b_row == 1)
    #print(finder)
    finder_contents = finder[0]
    #print(finder_contents)
    adjusted_list = []
    inner_list = []
    for c in np.arange(0,len(finder_contents)):
        if len(finder_contents) < 1:
            adjusted_zeros = 0
        elif len(finder_contents) == 1:
            adjusted_list.append(ms2_coeff_flipped[0,finder_contents])
            adjusted_zeros = adjusted_list[0]
        else:
            inner_list.append(ms2_coeff_flipped[0,finder_contents[c,]])
    if len(inner_list) > 0:        
        adjusted_zeros = sum(inner_list)
    
    if isinstance(adjusted_zeros, np.ndarray) == True:
        adjusted_zeros = adjusted_zeros[0,]
    
    #print(adjusted_zeros)
    
    return_list = []
    return_list.append(adjusted_ones)
    return_list.append(adjusted_zeros)
    
    return return_list 
    
#%%
def compute_dynamic_F(state, length, W, K, ms2_coeff_flipped, count_reduction_manual):
    trace_length = length
    
    state_flipped = K**W - state - 1
    
    adjust = get_adjusted(state_flipped, K, W, ms2_coeff_flipped)
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
    
    log_f1_terms_saved2 = log_f1_terms_saved
    
    for t in np.arange(0,W-1):
        log_f1_terms_saved2[0,t] = np.log(abs(np.exp(log_f1_terms_saved2[0,t]) - count_reduction_manual[t,]))
        
    log_F_terms = []
    log_F_terms.append(log_f1_terms_saved2)
    log_F_terms.append(log_f0_terms)
    
    return log_F_terms
    

def EM_long(initialised_parameters, n_steps, n_traces, signal_struct, compound_states, K, PERMITTED_MEMORY,
                     W, eps, seed_setter, kappa):
    
    A_log = initialised_parameters['A_log']
    lambda_log = initialised_parameters['lambda_log']
    noise_temp = initialised_parameters['noise_temp']
    pi0_log = initialised_parameters['pi0_log']
    v = initialised_parameters['v']
    v_logs = initialised_parameters['v_logs']
    
    # MS2 coefficient calculation
    ms2_coeff = ms2_loading_coeff(kappa, W)
    ms2_coeff_flipped = np.flip(ms2_coeff, 1)
    count_reduction_manual = np.zeros((1,W-1))
    for t in np.arange(0,W-1):
        count_reduction_manual[0,t] = np.sum(ms2_coeff[0,t+1:])
    count_reduction_manual = np.reshape(count_reduction_manual, (W-1,1))
    
    logL_tot = np.full((1, n_steps), np.NINF)

    fluo_length_total = 0
    for gh in signal_struct:
        fluo_length_total = fluo_length_total + len(np.transpose(gh))
    
    one_more = 0
    
    log_likelihoods = np.full((1, n_traces), np.NINF)
    for i_tr in np.arange(0, n_traces):
        log_likelihoods[0, i_tr] = np.NINF
    logL_tot = np.full((1, n_steps), np.NINF)

    for baum_welch in range(n_steps):
        print('baum_welch: ')
        print(baum_welch)
        logL_tot[0, baum_welch] = 0
        
        # Declare EM terms
        pi0_terms = np.full((1, K), np.NINF)
        A_terms = np.full((K, K), np.NINF)
        lambda_terms = np.NINF
        v_M_terms = np.full((K, K), np.NINF)
        v_b_terms_log = np.full((1, K), np.NINF)
        v_b_terms_sign = np.ones((1, K))
    
        #trace_adder = 0
        
        for i_tr in range(n_traces):
            print(i_tr)
            #print(datetime.datetime.now())
            data = signal_struct[i_tr]
    
            trace_length = len(np.transpose(data))
        
            states_container = []
            off_off = A_log[0, 0]
            off_on = A_log[1, 0]
            on_off = A_log[0, 1]
            on_on = A_log[1, 1]
            pi0_log = np.reshape(pi0_log, (2,1))
            v = np.reshape(v, (2,1))
            
            fluo_logs_abs = np.log(np.abs(data))
            x_term_logs = fluo_logs_abs
            
            xsign = np.sign(data)
            x_term_signs = xsign
    
    
            #compound_states_vector = np.arange(0, compound_states)
            #compound_states_vector = np.int32(compound_states_vector)
    
            # Complete first two 'anomalous' steps manually
            # Step One
            t = 0
            expansion_counter = 0
            RAM = 2
            updater = tuple([[], [0, \
                             1], [pi0_log[0, 0], \
                                                   pi0_log[1, 0]], \
                            [pi0_log[0, 0] + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, \
                                                      data[0, 0], v, 0, K, W, ms2_coeff_flipped), \
                            pi0_log[1, 0] + \
                            calcObservationLikelihood_long(lambda_log, noise_temp,
                                                      data[0, 0], v, 1, K, W, ms2_coeff_flipped)],
                                                      []])             
            states_container.append(updater)
    
            # Step Two
            t = 1
            expansion_counter = 1
            RAM = 4
    
            new_alphas = [states_container[0][3][0] + off_off + \
                          calcObservationLikelihood_long(lambda_log, noise_temp, data[0, 1], v, 0, K, W, ms2_coeff_flipped),
                          states_container[0][3][0] + off_on + \
                          calcObservationLikelihood_long(lambda_log, noise_temp, data[0, 1], v, 1, K, W, ms2_coeff_flipped),
                          states_container[0][3][1] + on_off + \
                          calcObservationLikelihood_long(lambda_log, noise_temp, data[0, 1], v, 2, K, W, ms2_coeff_flipped),
                          states_container[0][3][1] + on_on + calcObservationLikelihood_long(lambda_log,
                                          noise_temp, data[0, 1], v, 3, K, W, ms2_coeff_flipped)]
    
            updater = tuple([[0, 1],
                             [0, 1,
                              2, 3],
                             [off_off, off_on, on_off, on_on], new_alphas, [0, 0, 1, 1]])
    
            states_container.append(updater)
    #%%
            # Expansion Phase
            while RAM < PERMITTED_MEMORY:
                t = t + 1
                expansion_counter = expansion_counter + 1
                RAM = 2 * len(states_container[t-1][1])
                previous_states = states_container[t-1][1]
                previous_states2 = np.asarray(previous_states)
                allowed_states = np.zeros((len(previous_states2), 2))
                for i in range(len(previous_states2)):
                    allowed_states[i, 0] = previous_states2[i] << 1
                    allowed_states[i, 1] = (previous_states2[i] << 1) + 1
                allowed_states = allowed_states.astype(int)
                expanded_alphas = []
                previous_alphas = states_container[t-1][3]
                involved_transitions = []
    
                for k in range(len(previous_states2)):
                    for i in np.arange(0, 2):
                        input_state = previous_states2[k,]
                        target_state = allowed_states[k, i]
                        for_counting = np.int64(target_state)
                        if input_state % 2 == 0 and target_state % 2 == 0:
                            expanded_alphas.append(previous_alphas[k] + off_off + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(off_off)
                        elif input_state % 2 == 0 and target_state % 2 != 0:
                            expanded_alphas.append(previous_alphas[k] + off_on + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(off_on)
                        elif input_state % 2 != 0 and target_state % 2 == 0:
                            expanded_alphas.append(previous_alphas[k] + on_off + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(on_off)
                        elif input_state % 2 != 0 and target_state % 2 != 0:
                            expanded_alphas.append(previous_alphas[k] + on_on + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(on_on)
    
                old_states = list(previous_states2)
                present_states = np.reshape(allowed_states, (2*len(previous_states2), ))
                present_states_list = list(present_states)
                path_variable = []
                for i in range(len(previous_states2)):
                    path_variable.append(i)
                    path_variable.append(i)
                states_container.append(tuple([old_states, present_states_list, involved_transitions,
                                               expanded_alphas, path_variable]))
    
    #%%
            # First Expansion and Contraction
            mask = np.int64((2**W)-1)
            t = t + 1
    
            previous_states = states_container[t-1][1]
            previous_states2 = np.asarray(previous_states)
            previous_states2 = np.reshape(previous_states2, (len(previous_states2), 1))
            allowed_states = np.zeros((len(previous_states2), 2))
    
            for i in range(len(previous_states2)):
                allowed_states[i, 0] = previous_states2[i] << 1
                allowed_states[i, 1] = (previous_states2[i] << 1) + 1
    
                unique_states = np.unique(allowed_states)
                integrated_states = np.concatenate((previous_states2, allowed_states), axis=1)
    
            saved_integrated_states1 = integrated_states.copy()
            rowfind_list = []
            for u in unique_states:
                selector = (integrated_states[:, 1:3] == u)
                rowfind, colfind = np.where(selector == True)
                rowfind_list.append(rowfind)
    
            expanded_alphas = []
            previous_alphas = states_container[t-1][3]
            involved_transitions = []
    
            previous_alphas_matrix = np.zeros((len(previous_alphas), 2))
            for r in range(len(previous_alphas)):
                previous_alphas_matrix[r, 0] = r
                previous_alphas_matrix[r, 1] = previous_alphas[r]
    
    
            for s in range(len(unique_states)):
                lookup = rowfind_list[s]
                if len(lookup) == 1:
                    target_state = unique_states[s]
                    input_state = previous_states2[int(lookup)]
                    for_counting = np.int64(target_state)
    
                    selector2 = (previous_alphas_matrix[:, 0:1] == input_state)
                    rowfind2, colfind2 = np.where(selector2 == True)
                    rowfind2 = int(rowfind2)
    
    
                    if input_state % 2 == 0 and target_state % 2 == 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_off + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(off_off)
                    elif input_state % 2 == 0 and target_state % 2 != 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_on + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(off_on)
                    elif input_state % 2 != 0 and target_state % 2 == 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_off + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(on_off)
                    elif input_state % 2 != 0 and target_state % 2 != 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_on + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(on_on)
    
            accumulator = np.concatenate((np.asarray(rowfind_list), np.reshape(unique_states,
                                          (len(unique_states), 1)),
                                          np.reshape(np.asarray(involved_transitions),
                                                     (len(unique_states), 1)),
                                          np.reshape(np.asarray(expanded_alphas),
                                                     (len(unique_states), 1))), axis = 1)
    
            accumulator2 = accumulator[accumulator[:,3].argsort()[::-1]]
    
            accumulator3 = accumulator2[0:PERMITTED_MEMORY, :]
    
            addition_tuple = tuple([list(previous_states), list(accumulator3[:, 1].astype(int)),
                                   list(accumulator3[:, 2]), list(accumulator3[:, 3]),
                                   list(accumulator3[:, 0].astype(int))])
    
            states_container.append(addition_tuple)
            
    #%%
            # First vanilla expansion and contraction
            t = t + 1
            mask = np.int64((2**W)-1)
            
            previous_states = states_container[t-1][1]
            previous_states2 = np.asarray(previous_states)
            previous_states2 = np.reshape(previous_states2, (len(previous_states2), 1))
            allowed_states = np.zeros((len(previous_states2), 2))
    
            for i in range(len(previous_states2)):
                allowed_states[i, 0] = np.bitwise_and(previous_states2[i] << 1, mask)
                allowed_states[i, 1] = np.bitwise_and((previous_states2[i] << 1) + 1, mask)
    
            unique_states = np.unique(allowed_states)
            integrated_states = np.concatenate((previous_states2, allowed_states), axis=1)
            
            
            rowfind_list = []
            for u in unique_states:
                selector = (integrated_states[:, 1:3] == u)
                rowfind, colfind = np.where(selector == True)
                rowfind_list.append(rowfind)
    
            expanded_alphas = []
            previous_alphas = states_container[t-1][3]
            involved_transitions = []
    
            previous_alphas_matrix = np.zeros((len(previous_alphas), 2))
            for r in range(len(previous_alphas)):
                previous_alphas_matrix[r, 0] = previous_states2[r]
                previous_alphas_matrix[r, 1] = previous_alphas[r]
    
    
            for s in range(len(unique_states)):
                lookup = rowfind_list[s]
                if len(lookup) == 1:
                    target_state = unique_states[s]
                    input_state = previous_states2[int(lookup)]
                    for_counting = np.int64(target_state)
    
                    selector2 = (previous_alphas_matrix[:, 0:1] == input_state)
                    rowfind2, colfind2 = np.where(selector2 == True)
                    rowfind2 = int(rowfind2)
    
    
                    if input_state % 2 == 0 and target_state % 2 == 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_off + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(off_off)
                    elif input_state % 2 == 0 and target_state % 2 != 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_on + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(off_on)
                    elif input_state % 2 != 0 and target_state % 2 == 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_off + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(on_off)
                    elif input_state % 2 != 0 and target_state % 2 != 0:
                        expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_on + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))
                        involved_transitions.append(on_on)
    
    
                elif len(lookup) == 2:
                    double_holder = []
                    temp = []
                    target_state = unique_states[s]
                    for_counting = np.int64(target_state)
                    for v2 in lookup:
                        input_state = previous_states2[int(v2)]
                        selector2 = (previous_alphas_matrix[:, 0:1] == input_state)
                        rowfind2, colfind2 = np.where(selector2 == True)
                        rowfind2 = int(rowfind2)
    
    
                        if input_state % 2 == 0 and target_state % 2 == 0:
                            temp.append(previous_alphas_matrix[rowfind2, 1] + off_off)
                            double_holder.append(off_off)
                        elif input_state % 2 == 0 and target_state % 2 != 0:
                            temp.append(previous_alphas_matrix[rowfind2, 1] + off_on)
                            double_holder.append(off_on)
                        elif input_state % 2 != 0 and target_state % 2 == 0:
                            temp.append(previous_alphas_matrix[rowfind2, 1] + on_off)
                            double_holder.append(on_off)
                        elif input_state % 2 != 0 and target_state % 2 != 0:
                            temp.append(previous_alphas_matrix[rowfind2, 1] + on_on)
                            double_holder.append(on_on)
                    involved_transitions.append(double_holder)
                    expanded_alphas.append(np.logaddexp(temp[0], temp[1]) + \
                        calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))     
    
            holder_list = []        
            for w in rowfind_list:
                holder2 = []
                for x in w:
                    input_state = integrated_states[int(x),0]
                    holder2.append(input_state)
                holder_list.append(holder2)
          
    
            expanded_alphas_sorted_index = np.asarray(expanded_alphas).argsort()[::-1]
            expanded_alphas_sorted = np.asarray(expanded_alphas)[expanded_alphas_sorted_index]
            sources_expander = []
            for y in expanded_alphas_sorted_index:
                sources_expander.append(holder_list[y])
            transitions_expander = []    
            for y2 in expanded_alphas_sorted_index:
                transitions_expander.append(involved_transitions[y2])
                
            alphas_cut = expanded_alphas_sorted[0:PERMITTED_MEMORY]
            sources_cut = sources_expander[0:PERMITTED_MEMORY]
            transitions_cut = transitions_expander[0:PERMITTED_MEMORY]
            targ = unique_states[expanded_alphas_sorted_index]
            targets_cut = list((targ[0:PERMITTED_MEMORY]).astype(int))
            
            addition_tuple = tuple([list(previous_states), targets_cut, transitions_cut,
                                    alphas_cut, sources_cut])
    
            states_container.append(addition_tuple)
    
    #%%
            # Subsequent expansions and contractions
            while(t < trace_length-1):
                t = t + 1
        
                previous_states = states_container[t-1][1]
                previous_states2 = np.asarray(previous_states)
                previous_states2 = np.reshape(previous_states2, (len(previous_states2), 1))
                allowed_states = np.zeros((len(previous_states2), 2))
        
                for i in range(len(previous_states2)):
                    allowed_states[i, 0] = np.bitwise_and(previous_states2[i] << 1, mask)
                    allowed_states[i, 1] = np.bitwise_and((previous_states2[i] << 1) + 1, mask)
        
                unique_states = np.unique(allowed_states)
                integrated_states = np.concatenate((previous_states2, allowed_states), axis=1)
        
        
                rowfind_list = []
                for u in unique_states:
                    selector = (integrated_states[:, 1:3] == u)
                    rowfind, colfind = np.where(selector == True)
                    rowfind_list.append(rowfind)
        
                expanded_alphas = []
                previous_alphas = states_container[t-1][3]
                involved_transitions = []
        
                previous_alphas_matrix = np.zeros((len(previous_alphas), 2))
                for r in range(len(previous_alphas)):
                    previous_alphas_matrix[r, 0] = previous_states2[r]
                    previous_alphas_matrix[r, 1] = previous_alphas[r]
        
        
                for s in range(len(unique_states)):
                    lookup = rowfind_list[s]
                    if len(lookup) == 1:
                        target_state = unique_states[s]
                        input_state = previous_states2[int(lookup)]
                        for_counting = np.int64(target_state)
        
                        selector2 = (previous_alphas_matrix[:, 0:1] == input_state)
                        rowfind2, colfind2 = np.where(selector2 == True)
                        rowfind2 = int(rowfind2)
                        
                        if input_state % 2 == 0 and target_state % 2 == 0:
                            expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_off + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(off_off)
                        elif input_state % 2 == 0 and target_state % 2 != 0:
                            expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_on + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(off_on)
                        elif input_state % 2 != 0 and target_state % 2 == 0:
                            expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_off + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(on_off)
                        elif input_state % 2 != 0 and target_state % 2 != 0:
                            expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_on + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                          for_counting, K, W, ms2_coeff_flipped))
                            involved_transitions.append(on_on)
                            
                            
                    elif len(lookup) == 2:
                        double_holder = []
                        temp = []
                        target_state = unique_states[s]
                        for_counting = np.int64(target_state)
                        for v8 in lookup:
                            input_state = previous_states2[int(v8)]
                            selector2 = (previous_alphas_matrix[:, 0:1] == input_state)
                            rowfind2, colfind2 = np.where(selector2 == True)
                            rowfind2 = int(rowfind2)
        
        
                            if input_state % 2 == 0 and target_state % 2 == 0:
                                temp.append(previous_alphas_matrix[rowfind2, 1] + off_off)
                                double_holder.append(off_off)
                            elif input_state % 2 == 0 and target_state % 2 != 0:
                                temp.append(previous_alphas_matrix[rowfind2, 1] + off_on)
                                double_holder.append(off_on)
                            elif input_state % 2 != 0 and target_state % 2 == 0:
                                temp.append(previous_alphas_matrix[rowfind2, 1] + on_off)
                                double_holder.append(on_off)
                            elif input_state % 2 != 0 and target_state % 2 != 0:
                                temp.append(previous_alphas_matrix[rowfind2, 1] + on_on)
                                double_holder.append(on_on)
                        involved_transitions.append(double_holder)
                        expanded_alphas.append(np.logaddexp(temp[0], temp[1]) + \
                            calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                      for_counting, K, W, ms2_coeff_flipped))     
        
                holder_list = []        
                for w in rowfind_list:
                    holder2 = []
                    for x in w:
                        input_state = integrated_states[np.int64(x),0]
                        holder2.append(input_state)
                    holder_list.append(holder2)
              
        
                expanded_alphas_sorted_index = np.asarray(expanded_alphas).argsort()[::-1]
                expanded_alphas_sorted = np.asarray(expanded_alphas)[expanded_alphas_sorted_index]
                sources_expander = []
                for y in expanded_alphas_sorted_index:
                    sources_expander.append(np.int64(holder_list[y]))
                transitions_expander = []    
                for y2 in expanded_alphas_sorted_index:
                    transitions_expander.append(involved_transitions[y2])
                    
                alphas_cut = expanded_alphas_sorted[0:PERMITTED_MEMORY]
                sources_cut = sources_expander[0:PERMITTED_MEMORY]
                transitions_cut = transitions_expander[0:PERMITTED_MEMORY]
                targ = unique_states[expanded_alphas_sorted_index]
                #targets_cut = list((targ[0:PERMITTED_MEMORY]).astype(int))
                targets_cut = list(np.int64((targ[0:PERMITTED_MEMORY])))
                
                addition_tuple = tuple([list(previous_states), targets_cut, transitions_cut,
                                        alphas_cut, sources_cut])
        
                states_container.append(addition_tuple)
                
    #%%
            # Backward algorithm
            initial_betas = np.zeros((PERMITTED_MEMORY, 1))         
            betas_container = []
            betas_container.append(initial_betas)
        
            present_states = states_container[-1][1]
            beta_targets = states_container[-1][0]
            
            new_betas = {}
            
            previous_betas_matrix = np.asarray(initial_betas)
            
            
            for f3 in np.arange(0, PERMITTED_MEMORY, 1):
                temp = []
                to_compute = beta_targets[f3]
                theoretical_sources = integrated_states[f3,1:3]
                for tsc in np.arange(0, len(theoretical_sources)):
                    if int(theoretical_sources[tsc]) in present_states:
                        if to_compute % 2 == 0 and theoretical_sources[tsc] % 2 == 0:
                            trans = off_off
                        elif to_compute % 2 == 0 and theoretical_sources[tsc] % 2 != 0:
                            trans = off_on
                        elif to_compute % 2 != 0 and theoretical_sources[tsc] % 2 == 0:
                            trans = on_off
                        elif to_compute % 2 != 0 and theoretical_sources[tsc] % 2 != 0:
                            trans = on_on
    
                        temp.append(trans + 0 + calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t], v,
                                                   int(theoretical_sources[tsc]), K, W, ms2_coeff_flipped))
                if len(temp) == 0:
                    new_betas[to_compute] = np.NINF
                if len(temp) == 1:
                    new_betas[to_compute] = float(np.asarray(temp))
                elif len(temp) == 2:
                    new_betas[to_compute] = np.logaddexp(temp[0], temp[1])
            betas_container.append(new_betas)
            beta_count = 2
    
    #%%
            # Automated further backward algorithm
            for t2 in range(trace_length-2, expansion_counter,-1):
                reverse = trace_length -1 -t2
                present_states = states_container[t2][1]
                beta_targets = states_container[t2][0]
        
                new_betas = {}
                previous_betas_matrix = betas_container[beta_count-1]
                for f3 in np.arange(0, len(beta_targets)):
                    temp = []
                    to_compute = beta_targets[f3]
                    allowed_states_beta = np.zeros((2,1))
                    allowed_states_beta[0,0] = np.bitwise_and(to_compute << 1, mask)
                    allowed_states_beta[1,0] = np.bitwise_and((to_compute << 1) + 1, mask)
                    theoretical_sources = allowed_states_beta
                    for tsc in np.arange(0, len(theoretical_sources)):
                        if (int(theoretical_sources[tsc]) in present_states) and previous_betas_matrix[int(theoretical_sources[tsc])] != np.NINF:
                            if to_compute % 2 == 0 and theoretical_sources[tsc] % 2 == 0:
                                trans = off_off
                            elif to_compute % 2 == 0 and theoretical_sources[tsc] % 2 != 0:
                                trans = off_on
                            elif to_compute % 2 != 0 and theoretical_sources[tsc] % 2 == 0:
                                trans = on_off
                            elif to_compute % 2 != 0 and theoretical_sources[tsc] % 2 != 0:
                                trans = on_on
    
                            temp.append(trans + previous_betas_matrix[int(theoretical_sources[tsc])] + \
                                    calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t2], v, \
                                                   int(theoretical_sources[tsc]), K, W, ms2_coeff_flipped))
                        else:
                             pass
                    if len(temp) == 0:
                        new_betas[to_compute] = np.NINF
                    if len(temp) == 1:
                        new_betas[to_compute] = float(np.asarray(temp))
                    elif len(temp) == 2:
                        new_betas[to_compute] = np.logaddexp(temp[0], temp[1])
                betas_container.append(new_betas)
                marker = reverse
                beta_count = beta_count + 1
                
    #%%            
            cutter = int(PERMITTED_MEMORY / 2)
            # Backward algorithm during contraction phase
            lcount = 0
            for t2 in range(expansion_counter,0,-1):
                lcount = lcount + 1
                marker = marker + 1
                
                present_states = states_container[t2][1]
                beta_targets = states_container[t2][0]
        
                new_betas = []
                if lcount == 1:
                    test2 = np.fromiter(betas_container[marker].values(), dtype=float)
                else:
                    test2 = np.reshape(np.asarray(betas_container[marker]),
                                                              (len(betas_container[marker]), 1))
                previous_betas_matrix = np.reshape(test2, (len(test2), 1))
                            
                
                cut_integrated_states = saved_integrated_states1[0:cutter,:]
                               
                
                for f3 in np.arange(0, cutter, 1):
                    temp = []
                    to_compute = beta_targets[f3]
                    sources = cut_integrated_states[f3,1:3]
                    if to_compute % 2 == 0 and sources[0,] % 2 == 0:
                        trans0 = off_off
                    elif to_compute % 2 == 0 and sources[0,] % 2 != 0:
                        trans0 = off_on
                    elif to_compute % 2 != 0 and sources[0,] % 2 == 0:
                        trans0 = on_off
                    elif to_compute % 2 != 0 and sources[0,] % 2 != 0:
                        trans0 = on_on
                        
                    if to_compute % 2 == 0 and sources[1,] % 2 == 0:
                        trans1 = off_off
                    elif to_compute % 2 == 0 and sources[1,] % 2 != 0:
                        trans1 = off_on
                    elif to_compute % 2 != 0 and sources[1,] % 2 == 0:
                        trans1 = on_off
                    elif to_compute % 2 != 0 and sources[1,] % 2 != 0:
                        trans1 = on_on    
                    
                    temp.append(trans0 + previous_betas_matrix[int(sources[0,]),] + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t2], v,
                                                       int(sources[0,]), K, W, ms2_coeff_flipped))
                    
                    temp.append(trans1 + previous_betas_matrix[int(sources[1,]),] + \
                                calcObservationLikelihood_long(lambda_log, noise_temp, data[0, t2], v,
                                                       int(sources[1,]), K, W, ms2_coeff_flipped))
                    
                    new_betas.append((np.logaddexp(temp[0], temp[1])).item(0))
                betas_container.append(new_betas)
                cutter = int(cutter / 2)
                
    #%%
            # Remodel Alpha and Beta into dictionaries       
            alphas_remodelled = []
            for n5 in np.arange(0,len(states_container)):
                s_dict = {}
                for n6 in np.arange(0,len(states_container[n5][1])):
                    lifted_state = states_container[n5][1][n6]
                    lifted_alpha = states_container[n5][3][n6]
                    s_dict[lifted_state] = lifted_alpha
                alphas_remodelled.append(s_dict)
            
            initial_dict = {}
            key_list = list(alphas_remodelled[-1].keys())
            for n7 in np.arange(0,PERMITTED_MEMORY):
                key_getter = key_list[n7]
                initial_dict[key_getter] = 0
                
            
            betas_container[0] = initial_dict
            
            final_dict = {}
            cutter2 = int(PERMITTED_MEMORY / 2)
            for n8 in np.arange(expansion_counter,0,-1):
                final_dict = {}
                for n9 in np.arange(0, cutter2):
                    final_dict[n9] = betas_container[-n8][n9]
                cutter2 = int(cutter2/2)
                betas_container[-n8] = final_dict
                
    #%%
            # New Gamma
            final_entry = states_container[-1]
            final_alphas = final_entry[3]
            ll = scipy.special.logsumexp(final_alphas)
            log_likelihoods[0,i_tr] = ll
            logL_tot[0, baum_welch] = logL_tot[0,baum_welch] + ll
    
    
            gammas = []
            gamma_reverse = trace_length
            for i8 in np.arange(0,trace_length):
                gamma_reverse = gamma_reverse - 1
                alpha_dict_keys_extracted = list(alphas_remodelled[gamma_reverse].keys())
                gammasum = {}
                for i9 in np.arange(0,len(alpha_dict_keys_extracted)):
                    ke = alpha_dict_keys_extracted[i9]
                    gammasum[ke] = alphas_remodelled[gamma_reverse][ke] + betas_container[i8][ke] - ll
                gammas.append(gammasum)                
                
                
    #%%
            
            # Calculate Xi        
            copied_betas = betas_container.copy()
            copied_betas.reverse()
            key_betas = copied_betas[1]
            key_state = states_container[0]
            next_state = states_container[1]
            
            manual_first_transition = [off_off, off_on, on_off, on_on]
            
            off_off_container = []
            off_on_container = []
            on_off_container = []
            on_on_container = []
            obs_viewer = [] 
            xi_temp = []
            for i in np.arange(0,4):
                source_state = next_state[4][i]
                key_transition = manual_first_transition[i]
                source_alpha = key_state[3][source_state]
                state_getter = next_state[1][i]
                obs = calcObservationLikelihood_long(lambda_log, noise_temp, data[0, 1], v,
                                                              int(state_getter), K, W, ms2_coeff_flipped)
                xi_temp.append(key_betas[i] + key_transition + source_alpha + \
                                     obs - ll)
                xi_result = key_betas[i] + key_transition + source_alpha + \
                                     obs - ll                 
                if key_transition == off_off:
                    off_off_container.append(xi_result)
                elif key_transition == off_on:
                    off_on_container.append(xi_result)
                elif key_transition == on_off:
                    on_off_container.append(xi_result)
                elif key_transition == on_on:
                    on_on_container.append(xi_result)
            
            xi_count = 1
    #%%        
            # Expansion Xi        
            for xi_loop in np.arange(xi_count, expansion_counter):
                key_betas = copied_betas[xi_count+1]
                key_state = states_container[xi_count]
                next_state = states_container[xi_count+1]
                xi_temp = []
                for i in np.arange(0,len(next_state[4])):
                    source_state = next_state[4][i]
                    key_transition = next_state[2][i]
                    source_alpha = key_state[3][source_state]
                    state_getter = next_state[1][i]
                    obs = calcObservationLikelihood_long(lambda_log, noise_temp, data[0, xi_count+1], v,
                                                              int(state_getter), K, W, ms2_coeff_flipped)
                    xi_result = key_betas[i] + key_transition + source_alpha + \
                                     obs - ll               
                    if key_transition == off_off:
                        off_off_container.append(xi_result)
                    elif key_transition == off_on:
                        off_on_container.append(xi_result)
                    elif key_transition == on_off:
                        on_off_container.append(xi_result)
                    elif key_transition == on_on:
                        on_on_container.append(xi_result)                 
                xi_count = xi_count + 1
    
    #%%
           # First Expansion and Contraction Xi
            #xi_count = xi_count + 1   
            for xi_loop in np.arange(expansion_counter, expansion_counter+1):
                key_betas = copied_betas[xi_count+1]
                key_state = states_container[xi_count]
                next_state = states_container[xi_count+1]
                xi_temp = []
                for i in np.arange(0,len(next_state[4])):
    
                    source_state = next_state[4][i]
                    key_transition = next_state[2][i]
                    source_alpha = key_state[3][int(source_state)]
                    state_getter = next_state[1][i]
                    obs = calcObservationLikelihood_long(lambda_log, noise_temp, data[0, xi_count+1], v,
                                                          int(state_getter), K, W, ms2_coeff_flipped)
                    xi_result = key_betas[int(state_getter)] + key_transition + source_alpha + \
                                 obs - ll
                    if key_transition == off_off:
                        off_off_container.append(xi_result)
                    elif key_transition == off_on:
                        off_on_container.append(xi_result)
                    elif key_transition == on_off:
                        on_off_container.append(xi_result)
                    elif key_transition == on_on:
                        on_on_container.append(xi_result)
                xi_count = xi_count + 1
    
    #%%
            # First Vanilla Expansion and Contraction Xi
            for xi_loop in np.arange(expansion_counter+1, expansion_counter+2):
                key_betas = copied_betas[xi_count+1]
                key_state = states_container[xi_count]
                next_state = states_container[xi_count+1]
                xi_temp = []
                for i in np.arange(0,len(next_state[4])):
                    extracted_state = next_state[4][i]
                    if len(extracted_state) == 1:
                        source_state = extracted_state[0]
                        state_getter = next_state[1][i]
                        if source_state % 2 == 0 and state_getter % 2 == 0:
                            key_transition = off_off
                        elif source_state % 2 == 0 and state_getter % 2 != 0:
                            key_transition = off_on
                        elif source_state % 2 != 0 and state_getter % 2 == 0:
                            key_transition = on_off
                        elif source_state % 2 != 0 and state_getter % 2 != 0:
                            key_transition = on_on
                        source_alpha = alphas_remodelled[xi_loop][int(source_state)]
                        obs = calcObservationLikelihood_long(lambda_log, noise_temp, data[0, xi_count+1], v,
                                                              int(state_getter), K, W, ms2_coeff_flipped)
                        xi_result = key_betas[int(state_getter)] + key_transition + source_alpha + \
                                     obs - ll
                        if key_transition == off_off:
                            off_off_container.append(xi_result)
                        elif key_transition == off_on:
                            off_on_container.append(xi_result)
                        elif key_transition == on_off:
                            on_off_container.append(xi_result)
                        elif key_transition == on_on:
                            on_on_container.append(xi_result)
                    else:
                        for k in np.arange(0,2):    
                            source_state = extracted_state[k]
                            state_getter = next_state[1][i]
                            source_alpha = alphas_remodelled[xi_loop][int(source_state)]
                            if source_state % 2 == 0 and state_getter % 2 == 0:
                                key_transition = off_off
                            elif source_state % 2 == 0 and state_getter % 2 != 0:
                                key_transition = off_on
                            elif source_state % 2 != 0 and state_getter % 2 == 0:
                                key_transition = on_off
                            elif source_state % 2 != 0 and state_getter % 2 != 0:
                                key_transition = on_on
                            
                            obs = calcObservationLikelihood_long(lambda_log, noise_temp, data[0, xi_count+1], v,
                                                                  int(state_getter), K, W, ms2_coeff_flipped)
                            obs_viewer.append(obs)
                            xi_result = key_betas[int(state_getter)] + key_transition + source_alpha + \
                                         obs - ll
                            if key_transition == off_off:
                                off_off_container.append(xi_result)
                            elif key_transition == off_on:
                                off_on_container.append(xi_result)
                            elif key_transition == on_off:
                                on_off_container.append(xi_result)
                            elif key_transition == on_on:
                                on_on_container.append(xi_result)
                xi_count = xi_count + 1
                           
                    
    #%%
            for xi_loop in np.arange(expansion_counter+2, trace_length-1):
                key_betas = copied_betas[xi_count+1]
                key_state = states_container[xi_count]
                next_state = states_container[xi_count+1]
                xi_temp = []
                for i in np.arange(0,len(next_state[4])):
                    extracted_state = next_state[4][i]
                    if len(extracted_state) == 2:
                        for k in np.arange(0,2):    
                            source_state = extracted_state[k]
                            state_getter = next_state[1][i]    
                            source_alpha = alphas_remodelled[xi_loop][int(source_state)]
                            
                            if source_state % 2 == 0 and state_getter % 2 == 0:
                                key_transition = off_off
                            elif source_state % 2 == 0 and state_getter % 2 != 0:
                                key_transition = off_on
                            elif source_state % 2 != 0 and state_getter % 2 == 0:
                                key_transition = on_off
                            elif source_state % 2 != 0 and state_getter % 2 != 0:
                                key_transition = on_on
                            
                            obs = calcObservationLikelihood_long(lambda_log, noise_temp, data[0, xi_count+1], v,
                                                                  int(state_getter), K, W, ms2_coeff_flipped)
                            obs_viewer.append(obs)
                            xi_result = key_betas[int(state_getter)] + key_transition + source_alpha + \
                                         obs - ll
                            if key_transition == off_off:
                                off_off_container.append(xi_result)
                            elif key_transition == off_on:
                                off_on_container.append(xi_result)
                            elif key_transition == on_off:
                                on_off_container.append(xi_result)
                            elif key_transition == on_on:
                                on_on_container.append(xi_result)
                    else:
                        source_state = extracted_state[0]
                        state_getter = next_state[1][i]    
                        
                        if source_state % 2 == 0 and state_getter % 2 == 0:
                            key_transition = off_off
                        elif source_state % 2 == 0 and state_getter % 2 != 0:
                            key_transition = off_on
                        elif source_state % 2 != 0 and state_getter % 2 == 0:
                            key_transition = on_off
                        elif source_state % 2 != 0 and state_getter % 2 != 0:
                            key_transition = on_on
                        
                        source_alpha = alphas_remodelled[xi_loop][int(source_state)]
                        obs = calcObservationLikelihood_long(lambda_log, noise_temp, data[0, xi_count+1], v,
                                                                  int(state_getter), K, W, ms2_coeff_flipped)                    
                        obs_viewer.append(obs)
                        xi_result = key_betas[int(state_getter)] + key_transition + source_alpha + \
                                     obs - ll
                        if key_transition == off_off:
                            off_off_container.append(xi_result)
                        elif key_transition == off_on:
                            off_on_container.append(xi_result)
                        elif key_transition == on_off:
                            on_off_container.append(xi_result)
                        elif key_transition == on_on:
                            on_on_container.append(xi_result)               
                xi_count = xi_count + 1 
    #%%
            # Update pi0
            for m in np.arange(0, K):
               pi0_terms[0,m] = np.logaddexp(pi0_terms[0,m], gammas[-1][m])
            # Update A
            off_off_array = np.expand_dims(np.asarray(off_off_container), axis = 1)
            off_on_array = np.expand_dims(np.asarray(off_on_container), axis = 1)
            on_off_array = np.expand_dims(np.asarray(on_off_container), axis = 1)
            on_on_array = np.expand_dims(np.asarray(on_on_container), axis = 1)
            
            off_off_array = off_off_array[off_off_array> -10000000]
            off_on_array = off_on_array[off_on_array> -10000000]
            on_off_array = on_off_array[on_off_array> -10000000]
            on_on_array = on_on_array[on_on_array> -10000000]
            off_off_array = np.reshape(off_off_array, (len(off_off_array), 1))
            off_on_array = np.reshape(off_on_array, (len(off_on_array), 1))
            on_off_array = np.reshape(on_off_array, (len(on_off_array), 1))
            on_on_array = np.reshape(on_on_array, (len(on_on_array), 1))
            ###################################################################
            # Experimental Block
            ###################################################################
            
            A_terms[0,0] = scipy.special.logsumexp(np.concatenate(
                    (np.reshape(A_terms[0,0], (1,1)), off_off_array), axis = 0)) 
            A_terms[1,0] = scipy.special.logsumexp(np.concatenate(
                    (np.reshape(A_terms[1,0], (1,1)), off_on_array), axis = 0)) 
            A_terms[0,1] = scipy.special.logsumexp(np.concatenate(
                    (np.reshape(A_terms[0,1], (1,1)), on_off_array), axis = 0)) 
            A_terms[1,1] = scipy.special.logsumexp(np.concatenate(
                    (np.reshape(A_terms[1,1], (1,1)), on_on_array), axis = 0)) 
            gammas_copy = gammas.copy()
            gammas_copy.reverse()
            
            term_ith = []
            for t in np.arange(0, trace_length):
                for key in gammas_copy[t]:
                    adjusted_list = get_adjusted(int(key), K, W, ms2_coeff_flipped)
                    term_ith.append(gammas_copy[t][key] + np.log((data[0,t] - \
                                   (adjusted_list[1] * v[0, 0] + \
                                   adjusted_list[0] * v[1, 0]))**2))
                
            flattened = np.asarray(term_ith)
            flattened = np.expand_dims(flattened, axis=0)
            test2 = np.expand_dims(np.expand_dims(np.array(lambda_terms), axis = 0), axis = 0)
            #test3 = np.concatenate((test2, flattened), axis = 0)
            test3 = np.concatenate((test2, flattened), axis = 1)
            test4 = scipy.special.logsumexp(test3)
            lambda_terms = test4
            
            
            #log_F_terms = F_dict[trace_length]
            #print(datetime.datetime.now())
            terms_ith = []
            for m in np.arange(0,K):
                for n in np.arange(0,K):
                    terms_ith = []                
                    for t in np.arange(0, trace_length):
                        for key in gammas_copy[t]:
                            #i_result = gammas_copy[t][key] + log_F_terms[n][key,t] + log_F_terms[m][key,t]
                            i_result = gammas_copy[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[n][0,t] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t]
                            if i_result > -100000000:
                                terms_ith.append(i_result)
                    filler = np.ones((1,1))
                    filler[0,0] = v_M_terms[m,n]
                    v_M_terms[m,n] = scipy.special.logsumexp(np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0))
            #print(datetime.datetime.now())       
            
    
            terms_b_log_ith = []
            sign_list = []
            tmp = np.ones((K,1))
            for m in np.arange(0,K):
                terms_b_log_ith = []
                sign_list = []
                for t in np.arange(0, trace_length):
                    for key in gammas_copy[t]:
                        terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t])
                        #terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + log_F_terms[m][key,t])
                        sign_list.append(x_term_signs[0,t])
                        
                reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
                reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
                signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
                signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
                assign1 = np.concatenate((np.reshape(np.array(v_b_terms_log[0,m]), (1,1)), reshaped2), axis = 1)
                assign2 = np.concatenate((np.reshape(np.array(v_b_terms_sign[0,m]), (1,1)), signs2), axis = 1)
                tmp = log_sum_exp(assign1, assign2)
                v_b_terms_log[0,m] = tmp[0,]
                v_b_terms_sign[0,m] = tmp[1,]
                #print(v_b_terms_sign)
               
    #%%
        # Maximisation Step
        # pi0_log
        pi0_old = np.exp(pi0_log)
        pi0_log = pi0_terms - np.log(n_traces)
        
        pi0_norm_rel_change = abs(np.linalg.norm(pi0_old,2) - np.linalg.norm(np.exp(pi0_log),2)) \
        /  np.linalg.norm(pi0_old)
        
        # A_log
        A_old = np.exp(A_log)
        A_log = A_terms
        arr = np.zeros((K,0))
        for n in np.arange(0, K):
            arr = A_log[:,n]
            arr_max = max(arr)
            A_log[:,n] = A_log[:,n] - (arr_max + np.log(np.sum(np.exp(arr[:,] - arr_max))))
       
        A_norm_rel_change = abs(np.linalg.norm(A_old,2) - np.linalg.norm(np.exp(A_log),2)) / np.linalg.norm(A_old,2)    
        
        # lambda_log
        lambda_log_old = lambda_log
        
    
        lambda_log = np.log(n_traces*trace_length) - lambda_terms
        noise_log_old = -0.5 * lambda_log_old
        noise_log = -0.5 * lambda_log
        noise_temp = np.exp(noise_log)
    
           
        noise_change = np.multiply(np.exp(noise_log), abs(np.exp(noise_log_old - noise_log)- 1))
        noise_rel_change = noise_change / np.exp(noise_log_old)
        
        # v
        v_logs_old = v_logs
        m_sign = np.ones((K,K))
        m_log = v_M_terms
        
        
        b_sign = v_b_terms_sign
        b_log = v_b_terms_log
        
        v_updated = v_log_solve(m_log, m_sign, b_log, b_sign)
        v_logs = v_updated[0,:]
        #v_signs = v_updated[1,:]
        v = np.exp(v_logs)
        v = np.reshape(v, (2,1))
        
        v_norm_change = abs(np.linalg.norm(np.exp(v_logs_old), 2) - np.linalg.norm(np.exp(v_logs), 2))
        v_norm_rel_change = v_norm_change / np.linalg.norm(np.exp(v_logs_old))
        
        
        # Change in logL per time step
        logL_norm_change = 0
        if baum_welch > 0:
            logL_norm_change = logL_tot[0,baum_welch] - logL_tot[0,baum_welch - 1]
            logL_norm_change = abs(logL_norm_change) / fluo_length_total
            
        print(pi0_norm_rel_change)
        print(A_norm_rel_change)
        print(noise_rel_change)    
        print(v_norm_rel_change)
        print(logL_norm_change)
        print('A: ')
        print(np.exp(A_log))
        print('pi0: ')
        print(np.exp(pi0_log))
        print('noise: ')
        print(np.exp(noise_log))
        print('v: ')
        print(np.exp(v_logs))
        print('lltot: ')
        print(logL_tot[0,baum_welch])
        
        if one_more == 1:
            break
        
        if (np.max(np.array([pi0_norm_rel_change, A_norm_rel_change, noise_rel_change, \
                        v_norm_rel_change, logL_norm_change]))) < eps and (one_more == 0):
            logL_tot = logL_tot[0:baum_welch]
            print('EXCEEDED')
            one_more = 1
            break
        
    output_dict = {}
    output_dict['A'] = np.exp(A_log)
    output_dict['pi0'] = np.exp(pi0_log)
    output_dict['v'] = np.exp(v_logs)
    output_dict['noise'] = np.exp(noise_log)
    output_dict['logL'] = logL_tot[0, baum_welch]
    output_dict['EM seed'] = seed_setter

    return output_dict