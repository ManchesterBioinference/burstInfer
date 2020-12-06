# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:48:33 2020

@author: Jon
"""
import numpy as np
import scipy
from scipy import special
from burstInfer.calcObservationLikelihood import calcObservationLikelihood
from numba import jit

#@jit(nopython=False)
def exact_forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container, off_off, off_on,
                     on_off, on_on, trace_length,
                     log_likelihoods, logL_tot, baum_welch, i_tr):

    
    states_container = []
    #pi0_log = np.reshape(pi0_log, (2,1))
    #v = np.reshape(v, (2,1))
    
    fluo_logs_abs = np.log(np.abs(data))
    #fluo_logs_abs = np.reshape(fluo_logs_abs, (len(fluo_logs_abs),1))
    x_term_logs = np.ones((K**W, trace_length))
    for i in np.arange(0,K**W):
        x_term_logs[i,:] = fluo_logs_abs
    x_term_signer = np.sign(data)
    x_term_signs = np.ones((K**W, trace_length))
    for i in np.arange(0,K**W):
        x_term_signs[i,:] = x_term_signer

    compound_states = K**W
    compound_states_vector = np.arange(0, compound_states)
    compound_states_vector = np.int32(compound_states_vector)
    
    ram_holder = []
    
    PERMITTED_MEMORY = K**W
    

    # Complete first two 'anomalous' steps manually
    # Step One
    t = 0
    expansion_counter = 0
    RAM = 2
    #print('pi0_log at alpha matrix calc: ')
    #print(pi0_log)
    updater = tuple([[], [compound_states_vector[0], \
                     compound_states_vector[1]], [pi0_log[0, 0], \
                                           pi0_log[1, 0]], \
                    [pi0_log[0, 0] + \
                    calcObservationLikelihood(lambda_log, noise_temp, \
                                              data[0, 0], v, 0, K, W, ms2_coeff_flipped), \
                    pi0_log[1, 0] + \
                    calcObservationLikelihood(lambda_log, noise_temp,
                                              data[0, 0], v, 1, K, W, ms2_coeff_flipped)],
                                              []])
    #print(updater)                
    # Idea for Viterbi
    states_container.append(updater)
    ram_holder.append(RAM)
# =============================================================================
#         viterbi_delta = np.zeros((2, 1))
#         #viterbi_storage = np.zeros((2,1))
#         viterbi_delta[0, 0] = pi0_log[0, 0] + calcObservationLikelihood(
#             lambda_log, noise_temp, data[0, 0], v, 3, 0)
#         viterbi_delta[1, 0] = pi0_log[1, 0] + \
#         calcObservationLikelihood(lambda_log, noise_temp, data[0, 0], v,
#                                   2, 1)
#         viterbi_storage = np.amax(viterbi_delta, axis=0)
#         viterbi_delta_list = list(viterbi_delta)
#         viterbi_storage_list = list(viterbi_storage)
# =============================================================================

    # Step Two
    t = 1
    expansion_counter = 1
    RAM = 4

    new_alphas = [states_container[0][3][0] + off_off + \
                  calcObservationLikelihood(lambda_log, noise_temp, data[0, 1], v, 0, K, W, ms2_coeff_flipped),
                  states_container[0][3][0] + off_on + \
                  calcObservationLikelihood(lambda_log, noise_temp, data[0, 1], v, 1, K, W, ms2_coeff_flipped),
                  states_container[0][3][1] + on_off + \
                  calcObservationLikelihood(lambda_log, noise_temp, data[0, 1], v, 2, K, W, ms2_coeff_flipped),
                  states_container[0][3][1] + on_on + calcObservationLikelihood(lambda_log,
                                  noise_temp, data[0, 1], v, 3, K, W, ms2_coeff_flipped)]

    updater = tuple([[compound_states_vector[0], compound_states_vector[1]],
                     [compound_states_vector[0], compound_states_vector[1],
                      compound_states_vector[2], compound_states_vector[3]],
                     [off_off, off_on, on_off, on_on], new_alphas, [0, 0, 1, 1]])

    states_container.append(updater)
    ram_holder.append(RAM)
    #print(updater)
#%%
    # Expansion Phase
    while RAM < PERMITTED_MEMORY:
        t = t + 1
        expansion_counter = expansion_counter + 1
        RAM = 2 * len(states_container[t-1][1])
        ram_holder.append(RAM)
        previous_states = states_container[t-1][1]
        #print(t)
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
                for_counting = np.int32(target_state)
                if input_state % 2 == 0 and target_state % 2 == 0:
                    expanded_alphas.append(previous_alphas[k] + off_off + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))
                    involved_transitions.append(off_off)
                elif input_state % 2 == 0 and target_state % 2 != 0:
                    expanded_alphas.append(previous_alphas[k] + off_on + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))
                    involved_transitions.append(off_on)
                elif input_state % 2 != 0 and target_state % 2 == 0:
                    expanded_alphas.append(previous_alphas[k] + on_off + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))
                    involved_transitions.append(on_off)
                elif input_state % 2 != 0 and target_state % 2 != 0:
                    expanded_alphas.append(previous_alphas[k] + on_on + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))
                    involved_transitions.append(on_on)

        old_states = list(previous_states2)
        present_states = np.reshape(allowed_states, (2*len(previous_states2), ))
        present_states_list = list(present_states)
        #path_variable = np.zeros((2* len(previous_states2), 1))
        path_variable = []
        for i in range(len(previous_states2)):
            path_variable.append(i)
            path_variable.append(i)
        states_container.append(tuple([old_states, present_states_list, involved_transitions,
                                       expanded_alphas, path_variable]))

#%%
    if PERMITTED_MEMORY == K**W:
        # Forward algorithm
        while(t < trace_length-1):                
        
            t = t + 1
            ram_holder.append(RAM)
            previous_states = states_container[t-1][1]
            previous_states2 = np.asarray(previous_states)
            allowed_states = np.zeros((len(previous_states2), 2))
            for i in range(int(len(previous_states2)/2)):
                allowed_states[i, 0] = previous_states2[i] << 1
                allowed_states[i, 1] = (previous_states2[i] << 1) + 1
                
            for i in np.arange(int(len(previous_states2)/2), len(previous_states2)):
                allowed_states[i, 0] = previous_states2[i - int(len(previous_states2)/2)] << 1
                allowed_states[i, 1] = (previous_states2[i - int(len(previous_states2)/2)] << 1) + 1
            allowed_states = allowed_states.astype(int)
            expanded_alphas = []
            previous_alphas = states_container[t-1][3]
            involved_transitions = []
            
            unique_states = np.unique(allowed_states)
            integrated_states = np.concatenate((np.reshape(previous_states2, \
                                (len(previous_states2), 1)), allowed_states), axis=1)
  
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
                double_holder = []
                temp = []
                target_state = unique_states[s]
                for_counting = np.int32(target_state)
                for v2 in lookup:
                    input_state = previous_states2[int(v2)] #!!!!!
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
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                               for_counting, K, W, ms2_coeff_flipped))     
            
            
            expanded_alphas_sorted_index = np.zeros((PERMITTED_MEMORY,))
            for i in np.arange(0, PERMITTED_MEMORY):
                expanded_alphas_sorted_index[i,] = i
            holder_list = []        
            for w in rowfind_list:
                holder2 = []
                for x in w:
                    input_state = integrated_states[int(x),0]
                    holder2.append(input_state)
                holder_list.append(holder2)
            sources_expander = []
            for y in expanded_alphas_sorted_index:
                 #print(y)
                 sources_expander.append(holder_list[int(y)])    
                
            
            addition_tuple = tuple([list(previous_states), list(previous_states), 
                                      involved_transitions, expanded_alphas,
                                      sources_expander])
            states_container.append(addition_tuple)

        # Backward algorithm
        initial_betas = np.zeros((PERMITTED_MEMORY, 1))         
        betas_container = []
        betas_container.append(initial_betas)
    
        present_states = states_container[-1][1]
        target_states = states_container[-1][4]
        transitions = states_container[-1][2]
        beta_targets = states_container[-1][0]
    
        new_betas = []
        
        previous_betas_matrix = np.asarray(initial_betas)
        
        
        for f3 in np.arange(0, PERMITTED_MEMORY, 1):
            temp = []
            to_compute = beta_targets[f3]
            sources = integrated_states[f3,1:3]
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
            
            temp.append(trans0 + 0 + calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                               int(sources[0,]), K, W, ms2_coeff_flipped))
            
            temp.append(trans1 + 0 + calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                               int(sources[1,]), K, W, ms2_coeff_flipped))
            
            new_betas.append(np.logaddexp(temp[0], temp[1]))
            #print(new_betas)
        betas_container.append(new_betas)
        #print(betas_container[1])
        
        # Automated further backward algorithm
        for t2 in range(trace_length-2, expansion_counter,-1):
            #print(t2)
            reverse = trace_length -1 -t2
            #print(reverse)
            present_states = states_container[t2][1]
            beta_targets = states_container[t2][0]
    
            new_betas = []
    
            previous_betas_matrix = np.reshape(np.asarray(betas_container[reverse]),
                                                      (len(betas_container[reverse]), 1))
            
            for f3 in np.arange(0, PERMITTED_MEMORY, 1):
                temp = []
                to_compute = beta_targets[f3]
                sources = integrated_states[f3,1:3]
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
                
                temp.append(trans0 + previous_betas_matrix[sources[0,],] + \
                            calcObservationLikelihood(lambda_log, noise_temp, data[0, t2], v,
                                                   int(sources[0,]), K, W, ms2_coeff_flipped))
                
                temp.append(trans1 + previous_betas_matrix[sources[1,],] + \
                            calcObservationLikelihood(lambda_log, noise_temp, data[0, t2], v,
                                                   int(sources[1,]), K, W, ms2_coeff_flipped))
                
                new_betas.append((np.logaddexp(temp[0], temp[1])).item(0))
                #print(new_betas)
            betas_container.append(new_betas)
            marker = reverse
            

        cutter = int(PERMITTED_MEMORY / 2)
        # Backward algorithm during contraction phase
        for t2 in range(expansion_counter,0,-1):
            #print(t2)
            #reverse = trace_length -1 -t2
            marker = marker + 1
            #print(marker)
            
            present_states = states_container[t2][1]
            beta_targets = states_container[t2][0]
    
            new_betas = []
    
            previous_betas_matrix = np.reshape(np.asarray(betas_container[marker]),
                                                      (len(betas_container[marker]), 1))
            
            
            cut_integrated_states = integrated_states[0:cutter,:]
                           
            
            for f3 in np.arange(0, cutter, 1):
                #print(f3)
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
                
                temp.append(trans0 + previous_betas_matrix[sources[0,],] + \
                            calcObservationLikelihood(lambda_log, noise_temp, data[0, t2], v,
                                                   int(sources[0,]), K, W, ms2_coeff_flipped))
                
                temp.append(trans1 + previous_betas_matrix[sources[1,],] + \
                            calcObservationLikelihood(lambda_log, noise_temp, data[0, t2], v,
                                                   int(sources[1,]), K, W, ms2_coeff_flipped))
                
                new_betas.append((np.logaddexp(temp[0], temp[1])).item(0))
                #print(new_betas)
            betas_container.append(new_betas)
            cutter = int(cutter / 2)
            
        #stop = time.time()        
    else:
        print('nothing')            
        
# =============================================================================
#     # Calculate Gamma
#     betas_container[0] = [0] * PERMITTED_MEMORY    
#     final_entry = states_container[-1]
#     final_alphas = final_entry[3]
#     # This will be hard
#     ll = np.float64(scipy.special.logsumexp(final_alphas))
#     log_likelihoods[0,i_tr] = ll
#     logL_tot[0, baum_welch] = logL_tot[0,baum_welch] + ll
#     #print('value of ll: ')
#     #print(ll)
#     
#     final_gammas = np.asarray(final_alphas) + np.asarray(betas_container[0]) - ll
#     penultimate_gammas = states_container[-2][3] + np.asarray(betas_container[1]) - ll
#     
#     gammas = []
#     gamma_reverse = trace_length
#     for i in range(0,trace_length):
#         gamma_reverse = gamma_reverse - 1
#         gammas.append(np.asarray(betas_container[i], dtype = np.float64) + np.asarray(states_container[gamma_reverse][3], dtype = np.float64) - ll)
# =============================================================================
    
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
    # Tentative first Xi loop
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
    obs_viewer = [] ###
    xi_container = []
    xi_temp = []
    for i in np.arange(0,4):
        source_state = next_state[4][i]
        key_transition = manual_first_transition[i]
        source_alpha = key_state[3][source_state]
        state_getter = next_state[1][i]
        obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, 1], v,
                                                      int(state_getter), K, W, ms2_coeff_flipped)
        xi_temp.append(key_betas[i] + key_transition + source_alpha + \
                             obs - ll)
        xi_result = key_betas[i] + key_transition + source_alpha + \
                             obs - ll
        obs_viewer.append(obs)                     
        if key_transition == off_off:
            off_off_container.append(xi_result)
        elif key_transition == off_on:
            off_on_container.append(xi_result)
        elif key_transition == on_off:
            on_off_container.append(xi_result)
        elif key_transition == on_on:
            on_on_container.append(xi_result)
    xi_container.append(xi_temp)
    #print(xi_container)
    
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
            obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_loop], v,
                                                      int(state_getter), K, W, ms2_coeff_flipped)
            xi_temp.append(key_betas[i] + key_transition + source_alpha + 
                             obs - ll)
            xi_result = key_betas[i] + key_transition + source_alpha + \
                             obs - ll
            #obs_viewer.append(obs)                 
            if key_transition == off_off:
                off_off_container.append(xi_result)
            elif key_transition == off_on:
                off_on_container.append(xi_result)
            elif key_transition == on_off:
                on_off_container.append(xi_result)
            elif key_transition == on_on:
                on_on_container.append(xi_result)                 
        xi_container.append(xi_temp)
        xi_count = xi_count + 1
        #print('COUNT XI')
        #print(xi_count)
#%%
   # Main Phase Xi
    #xi_count = xi_count + 1   
    for xi_loop in np.arange(xi_count+1, trace_length):
        key_betas = copied_betas[xi_count+1]
        key_state = states_container[xi_count]
        next_state = states_container[xi_count+1]
        xi_temp = []
        for i in np.arange(0,len(next_state[4])):
            xi_temp_double = []
            for k in np.arange(0,2):
                source_state = next_state[4][i][k]
                key_transition = next_state[2][i][k]
                source_alpha = key_state[3][source_state]
                state_getter = next_state[1][i]
                #print('xi_count: ')
                #print(xi_count)
                #print('xi_loop: ')
                #print(xi_loop)
                obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_loop], v,
                                                      int(state_getter), K, W, ms2_coeff_flipped)
                #print(obs)
                #print(source_alpha)
                obs_viewer.append(obs)
                xi_temp_double.append(key_betas[i] + key_transition + source_alpha + 
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
                xi_temp.append(xi_temp_double)
        xi_container.append(xi_temp)
        xi_count = xi_count + 1
        #print('xi_count: ')
        #print(xi_count)
        #print('xi_loop: ')
        #print(xi_loop)
        
    forward_backward_results = {}
    forward_backward_results['Gamma'] = gammas
    forward_backward_results['off_off_results'] = off_off_container
    forward_backward_results['off_on_results'] = off_on_container
    forward_backward_results['on_off_results'] = on_off_container
    forward_backward_results['on_on_results'] = on_on_container
    
    #print(forward_backward_results)
    
    return forward_backward_results
