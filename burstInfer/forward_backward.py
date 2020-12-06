# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:39:49 2020

@author: Jon
"""
import numpy as np
import scipy
from scipy import special
from burstInfer.calcObservationLikelihood import calcObservationLikelihood

def forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container, off_off, off_on,
                     on_off, on_on, PERMITTED_MEMORY, trace_length,
                     log_likelihoods, logL_tot, baum_welch, i_tr):
    t = 0
    expansion_counter = 0
    RAM = 2
    updater = tuple([[], [0, \
                     1], [pi0_log[0, 0], \
                                           pi0_log[1, 0]], \
                    [pi0_log[0, 0] + \
                    calcObservationLikelihood(lambda_log, noise_temp, \
                                              data[0, 0], v, 0, K, W, ms2_coeff_flipped), \
                    pi0_log[1, 0] + \
                    calcObservationLikelihood(lambda_log, noise_temp,
                                              data[0, 0], v, 1, K, W, ms2_coeff_flipped)],
                                              []])
    #print(updater)
    states_container.append(updater)

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
    #print(new_alphas)
    updater = tuple([[0, 1],
                     [0, 1,
                      2, 3],
                     [off_off, off_on, on_off, on_on], new_alphas, [0, 0, 1, 1]])
    #print(updater)
    states_container.append(updater)
#%%
    # Expansion Phase
    #print(datetime.datetime.now().time())
    #print('start forward')
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
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                              for_counting, K, W, ms2_coeff_flipped))
                involved_transitions.append(off_off)
            elif input_state % 2 == 0 and target_state % 2 != 0:
                expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_on + \
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                              for_counting, K, W, ms2_coeff_flipped))
                involved_transitions.append(off_on)
            elif input_state % 2 != 0 and target_state % 2 == 0:
                expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_off + \
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                              for_counting, K, W, ms2_coeff_flipped))
                involved_transitions.append(on_off)
            elif input_state % 2 != 0 and target_state % 2 != 0:
                expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_on + \
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
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
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                              for_counting, K, W, ms2_coeff_flipped))
                involved_transitions.append(off_off)
            elif input_state % 2 == 0 and target_state % 2 != 0:
                expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_on + \
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                              for_counting, K, W, ms2_coeff_flipped))
                involved_transitions.append(off_on)
            elif input_state % 2 != 0 and target_state % 2 == 0:
                expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_off + \
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                              for_counting, K, W, ms2_coeff_flipped))
                involved_transitions.append(on_off)
            elif input_state % 2 != 0 and target_state % 2 != 0:
                expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_on + \
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
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
                calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
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
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))
                    involved_transitions.append(off_off)
                elif input_state % 2 == 0 and target_state % 2 != 0:
                    expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + off_on + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))
                    involved_transitions.append(off_on)
                elif input_state % 2 != 0 and target_state % 2 == 0:
                    expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_off + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
                                                  for_counting, K, W, ms2_coeff_flipped))
                    involved_transitions.append(on_off)
                elif input_state % 2 != 0 and target_state % 2 != 0:
                    expanded_alphas.append(previous_alphas_matrix[rowfind2, 1] + on_on + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
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
                    calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
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
    #print(datetime.datetime.now().time())
    #print('start backward')
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

                temp.append(trans + 0 + calcObservationLikelihood(lambda_log, noise_temp, data[0, t], v,
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
                            calcObservationLikelihood(lambda_log, noise_temp, data[0, t2], v, \
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
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t2], v,
                                               int(sources[0,]), K, W, ms2_coeff_flipped))
            
            temp.append(trans1 + previous_betas_matrix[int(sources[1,]),] + \
                        calcObservationLikelihood(lambda_log, noise_temp, data[0, t2], v,
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
    #print(datetime.datetime.now().time())
    #print('start xi')
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
        obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, 1], v,
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
            obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_count+1], v,
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
            obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_count+1], v,
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
                obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_count+1], v,
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
                    
                    obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_count+1], v,
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
                    
                    obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_count+1], v,
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
                obs = calcObservationLikelihood(lambda_log, noise_temp, data[0, xi_count+1], v,
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
        
    forward_backward_results = {}
    forward_backward_results['Gamma'] = gammas
    forward_backward_results['off_off_results'] = off_off_container
    forward_backward_results['off_on_results'] = off_on_container
    forward_backward_results['on_off_results'] = on_off_container
    forward_backward_results['on_on_results'] = on_on_container
    
    return forward_backward_results
