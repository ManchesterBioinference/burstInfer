# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:10:19 2020

@author: Jon
"""
import numpy as np
import scipy
from burstInfer.v_log_solve import v_log_solve
from burstInfer.log_sum_exp import log_sum_exp
from burstInfer.ms2_loading_coeff import ms2_loading_coeff

#from numba import jit
from burstInfer.forward_backward import forward_backward
from burstInfer.get_adjusted import get_adjusted
from burstInfer.compute_dynamic_F import compute_dynamic_F
from burstInfer.logsumexp_numba import logsumexp_numba
#from burstinfer.calcObservationLikelihood_long import calcObservationLikelihood_long
from burstInfer.exact_forward_backward import exact_forward_backward
#from burstInfer.exact_forward_backward_numba import exact_forward_backward_numba

class HMM:
    
    def __init__(self, K, W, t_MS2, deltaT, kappa, compound_states,
                 processed_signals):
        self.K = K
        self.W = W
        self.t_MS2 = t_MS2
        self.deltaT = deltaT
        self.kappa = kappa
        self.compound_states = compound_states
        self.processed_signals = processed_signals
    
    
    def initialise_parameters(self):
        
        W = self.W
        K = self.K
        matrix_max = self.processed_signals['Matrix Max']
        matrix_mean = self.processed_signals['Matrix Mean']
    
        v_init = np.ones((K, 1))
        v_init[0, 0] = np.random.uniform(0,0.1) * (2/(W)) * matrix_max
        v_init[1,0] = np.random.uniform(0.2,0.6) * (2/(W)) * matrix_max
        noise_init = np.random.uniform(0.2,0.8) * matrix_mean
        pi0_init = np.zeros((2,1))
        pi0_init[0,0] = np.random.uniform(0.2,0.8)
        pi0_init[1,0] = 1 - pi0_init[0,0]
        A_init = np.zeros((2,2))
        
    # =============================================================================
    #     for j in np.arange(0,K):
    #         A_init[:,j] = np.random.gamma(shape=1,scale=1,size=(1,2))
    #         A_init[:,j] = A_init[:,j] / np.sum(A_init[:,j])
    # =============================================================================
        
        A_init = np.zeros((2,2))
        A_init[0,0] = np.random.uniform(0.2,0.8)
        A_init[1,0] = 1 - A_init[0,0]
        A_init[0,1] = np.random.uniform(0.2,0.8)
        A_init[1,1] = 1- A_init[0,1]
        
        lambda_init = -2 * np.log(noise_init)
        
        # Further initialisation
        pi0_log = np.log(pi0_init)
        A_temp = A_init
        A_log = np.log(A_temp)
        v = v_init
        lambda_log = lambda_init
        noise_temp = noise_init
        v_logs = np.log(abs(v_init))   
        
        print('v_init')
        print(v_init)
        print('A_init')
        print(A_init)
        print('noise_init')
        print(noise_init)
        
        parameter_dict = {}
        parameter_dict['pi0_log'] = pi0_log
        parameter_dict['A_temp'] = A_temp
        parameter_dict['A_log'] = A_log
        parameter_dict['v'] = v
        parameter_dict['lambda_log'] = lambda_log
        parameter_dict['noise_temp'] = noise_temp
        parameter_dict['v_logs'] = v_logs
        
        return parameter_dict

#%%    
# =============================================================================
#     @jit(nopython=True)
#     def get_adjusted(self, state, K, W, ms2_coeff):
#         
#         #ms2_coeff_flipped = np.flip(ms2_coeff_flipped, 1)
#         ms2_coeff_flipped = ms2_coeff
#         
#         one_accumulator = 0
#         zero_accumulator = 0
#         for count in np.arange(0,W):
#             ##print(count)
#             ##print(state&1)
#             if state & 1 == 1:
#                 ##print('one')
#                 one_accumulator = one_accumulator + ms2_coeff_flipped[0,count]
#             else:
#                 ##print('zero')
#                 zero_accumulator = zero_accumulator + ms2_coeff_flipped[0,count]    
#             state = state >> 1
#             ##print(state)
#             
#         return_list = []
#         return_list.append(one_accumulator)
#         return_list.append(zero_accumulator)
#         
#         return return_list
#     
#     
#     @jit(nopython=True)
#     def logsumexp_numba(self, X):
#         r = 0.0
#         for x in X:
#             r += np.exp(x)  
#         return np.log(r)
# =============================================================================
        
    
    def EM(self, initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
                         eps, seed_setter):
        
        K = self.K
        W = self.W
        kappa = self.kappa
        signal_struct = self.processed_signals['Processed Signals']
        
        A_log = initialised_parameters['A_log']
        lambda_log = initialised_parameters['lambda_log']
        noise_temp = initialised_parameters['noise_temp']
        pi0_log = initialised_parameters['pi0_log']
        v = initialised_parameters['v']
        v_logs = initialised_parameters['v_logs']
        
        # MS2 coefficient calculation
        ms2_coeff = ms2_loading_coeff(kappa, W)
        ms2_coeff_flipped = np.flip(ms2_coeff, 1)
        
        ms2_coeff_flipped = ms2_coeff #!!!!!!!!!!!!!!!!! HACK
        
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
            print('EM step number: ')
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
                #print(i_tr)
                #print(datetime.datetime.now())
                #print('start trace')
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
        
                
                # START FORWARD BACKWARD HERE
                
                forward_backward_results = forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container, off_off, off_on,
                     on_off, on_on, PERMITTED_MEMORY, trace_length,
                     log_likelihoods, logL_tot, baum_welch, i_tr)
                
                gammas = forward_backward_results['Gamma']
                off_off_container = forward_backward_results['off_off_results']
                off_on_container = forward_backward_results['off_on_results']
                on_off_container = forward_backward_results['on_off_results']
                on_on_container = forward_backward_results['on_on_results']
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
                
                #print(datetime.datetime.now().time())
                #print('before term_ith')
                term_ith = []
                for t in np.arange(0, trace_length):
                    for key in gammas_copy[t]:
                        adjusted_list = get_adjusted(int(key), K, W, ms2_coeff)
                        term_ith.append(gammas_copy[t][key] + np.log((data[0,t] - \
                                       (adjusted_list[1] * v[0, 0] + \
                                       adjusted_list[0] * v[1, 0]))**2))
                #print(datetime.datetime.now().time())
                #print('after term_ith')
                    
                flattened = np.asarray(term_ith)
                flattened = np.expand_dims(flattened, axis=0)
                test2 = np.expand_dims(np.expand_dims(np.array(lambda_terms), axis = 0), axis = 0)
                #test3 = np.concatenate((test2, flattened), axis = 0)
                test3 = np.concatenate((test2, flattened), axis = 1)
                test4 = scipy.special.logsumexp(test3)
                lambda_terms = test4
                
                gammas_cleaned = gammas_copy.copy() # HACK
                gammas_cleaned2 = []
                for g in gammas_cleaned: # HACK
                    g = {key:val for key, val in g.items() if val != np.NINF} # HACK
                    gammas_cleaned2.append(g)
                
                #log_F_terms = F_dict[trace_length]
                #print(datetime.datetime.now())
                #print('before v_M')
                terms_ith = []
                failure = 0 # HACK
                success = 0 # HACK
                success_viewer = [] # HACK
                failure_viewer = [] # HACK
                for m in np.arange(0,K):
                    for n in np.arange(0,K):
                        terms_ith = []                
                        for t in np.arange(0, trace_length):
                            for key in gammas_cleaned2[t]:
                                i_result = gammas_cleaned2[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[n][0,t] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t]
                                terms_ith.append(i_result)
        
        
                        filler = np.ones((1,1))
                        filler[0,0] = v_M_terms[m,n]
                        #v_M_terms[m,n] = scipy.special.logsumexp(np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0))
                        v_M_input = np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0) 
                        v_M_input = v_M_input[v_M_input != np.NINF]
                        v_M_terms[m,n] = logsumexp_numba(v_M_input)
                #print(datetime.datetime.now())       
                #print('after VM')
        
                #print(datetime.datetime.now().time())
                #print('before terms_b_log_ith')
                terms_b_log_ith = []
                sign_list = []
                tmp = np.ones((K,1))
                for m in np.arange(0,K):
                    terms_b_log_ith = []
                    sign_list = []
                    for t in np.arange(0, trace_length):
                        for key in gammas_copy[t]:
                            if gammas_copy[t][key] != np.NINF:
                                terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t])
                                #terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + log_F_terms[m][key,t])
                                sign_list.append(x_term_signs[0,t])
                    #print(datetime.datetime.now().time())
                    #print('mid terms_b_log_ith')        
                    reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
                    reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
                    signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
                    signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
                    assign1 = np.concatenate((np.reshape(np.array(v_b_terms_log[0,m]), (1,1)), reshaped2), axis = 1)
                    assign2 = np.concatenate((np.reshape(np.array(v_b_terms_sign[0,m]), (1,1)), signs2), axis = 1)
                    tmp = log_sum_exp(assign1, assign2)
                    v_b_terms_log[0,m] = tmp[0,]
                    v_b_terms_sign[0,m] = tmp[1,]
                    ##print(v_b_terms_sign)
                    ##print(datetime.datetime.now().time())
                    ##print('mid 2 terms_b_log_ith')
                #print(datetime.datetime.now().time())
                #print('after terms_b_log_ith')
                   
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
                #print('EXCEEDED')
                one_more = 1
                break
            
        output_dict = {}
        output_dict['A'] = np.exp(A_log)
        output_dict['pi0'] = np.exp(pi0_log)
        output_dict['v'] = np.exp(v_logs)
        output_dict['noise'] = np.exp(noise_log)
        output_dict['logL'] = logL_tot[0, baum_welch]
        output_dict['EM seed'] = seed_setter
        
        #print(datetime.datetime.now().time())
        #print('end program')
        return output_dict


#%%
    
    def EM_fixed(self, initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
                         eps, seed_setter):
        
        K = self.K
        W = self.W
        kappa = self.kappa
        signal_struct = self.processed_signals['Processed Signals']
        
        A_log = initialised_parameters['A_log']
        lambda_log = initialised_parameters['lambda_log']
        noise_temp = initialised_parameters['noise_temp']
        pi0_log = initialised_parameters['pi0_log']
        v = initialised_parameters['v']
        v_logs = initialised_parameters['v_logs']
        
        # MS2 coefficient calculation
        ms2_coeff = ms2_loading_coeff(kappa, W)
        ms2_coeff_flipped = np.flip(ms2_coeff, 1)
        
        ms2_coeff_flipped = ms2_coeff #!!!!!!!!!!!!!!!!! HACK
        
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
            print('EM (fixed) step number: ')
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
                #print(i_tr)
                #print(datetime.datetime.now())
                #print('start trace')
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
        
                # START FORWARD BACKWARD HERE
                
                forward_backward_results = forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container, off_off, off_on,
                     on_off, on_on, PERMITTED_MEMORY, trace_length,
                     log_likelihoods, logL_tot, baum_welch, i_tr)
                
                gammas = forward_backward_results['Gamma']
                off_off_container = forward_backward_results['off_off_results']
                off_on_container = forward_backward_results['off_on_results']
                on_off_container = forward_backward_results['on_off_results']
                on_on_container = forward_backward_results['on_on_results']

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
                
                #print(datetime.datetime.now().time())
                #print('before term_ith')
                term_ith = []
                for t in np.arange(0, trace_length):
                    for key in gammas_copy[t]:
                        adjusted_list = get_adjusted(int(key), K, W, ms2_coeff)
                        term_ith.append(gammas_copy[t][key] + np.log((data[0,t] - \
                                       (adjusted_list[1] * v[0, 0] + \
                                       adjusted_list[0] * v[1, 0]))**2))
                #print(datetime.datetime.now().time())
                #print('after term_ith')
                    
                flattened = np.asarray(term_ith)
                flattened = np.expand_dims(flattened, axis=0)
                test2 = np.expand_dims(np.expand_dims(np.array(lambda_terms), axis = 0), axis = 0)
                #test3 = np.concatenate((test2, flattened), axis = 0)
                test3 = np.concatenate((test2, flattened), axis = 1)
                test4 = scipy.special.logsumexp(test3)
                lambda_terms = test4
                
                gammas_cleaned = gammas_copy.copy() # HACK
                gammas_cleaned2 = []
                for g in gammas_cleaned: # HACK
                    g = {key:val for key, val in g.items() if val != np.NINF} # HACK
                    gammas_cleaned2.append(g)
                
                #log_F_terms = F_dict[trace_length]
                #print(datetime.datetime.now())
                #print('before v_M')
                terms_ith = []
                #failure = 0 # HACK
                #success = 0 # HACK
                #success_viewer = [] # HACK
                #failure_viewer = [] # HACK
                for m in np.arange(0,K):
                    for n in np.arange(0,K):
                        terms_ith = []                
                        for t in np.arange(0, trace_length):
                            for key in gammas_cleaned2[t]:
                                i_result = gammas_cleaned2[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[n][0,t] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t]
                                terms_ith.append(i_result)
        
        
                        filler = np.ones((1,1))
                        filler[0,0] = v_M_terms[m,n]
                        #v_M_terms[m,n] = scipy.special.logsumexp(np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0))
                        v_M_input = np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0) 
                        v_M_input = v_M_input[v_M_input != np.NINF]
                        v_M_terms[m,n] = logsumexp_numba(v_M_input)
                #print(datetime.datetime.now())       
                #print('after VM')
        
                #print(datetime.datetime.now().time())
                #print('before terms_b_log_ith')
                terms_b_log_ith = []
                sign_list = []
                tmp = np.ones((K,1))
                for m in np.arange(0,K):
                    terms_b_log_ith = []
                    sign_list = []
                    for t in np.arange(0, trace_length):
                        for key in gammas_copy[t]:
                            if gammas_copy[t][key] != np.NINF:
                                terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t])
                                #terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + log_F_terms[m][key,t])
                                sign_list.append(x_term_signs[0,t])
                    #print(datetime.datetime.now().time())
                    #print('mid terms_b_log_ith')        
                    reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
                    reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
                    signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
                    signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
                    assign1 = np.concatenate((np.reshape(np.array(v_b_terms_log[0,m]), (1,1)), reshaped2), axis = 1)
                    assign2 = np.concatenate((np.reshape(np.array(v_b_terms_sign[0,m]), (1,1)), signs2), axis = 1)
                    tmp = log_sum_exp(assign1, assign2)
                    v_b_terms_log[0,m] = tmp[0,]
                    v_b_terms_sign[0,m] = tmp[1,]
                    ##print(v_b_terms_sign)
                    ##print(datetime.datetime.now().time())
                    ##print('mid 2 terms_b_log_ith')
                #print(datetime.datetime.now().time())
                #print('after terms_b_log_ith')
                #print(v_b_terms_log)
                   
        #%%
            # Maximisation Step
            # pi0_log
            pi0_old = np.exp(pi0_log)
            pi0_log = pi0_terms - np.log(n_traces)
            
            pi0_norm_rel_change = abs(np.linalg.norm(pi0_old,2) - np.linalg.norm(np.exp(pi0_log),2)) \
            /  np.linalg.norm(pi0_old)
            
            # A_log
            A_old = np.exp(A_log)
            #A_log = A_terms
            A_log = np.log(A_old)
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
                #print('EXCEEDED')
                one_more = 1
                break
            
        output_dict = {}
        output_dict['A'] = np.exp(A_log)
        output_dict['pi0'] = np.exp(pi0_log)
        output_dict['v'] = np.exp(v_logs)
        output_dict['noise'] = np.exp(noise_log)
        output_dict['logL'] = logL_tot[0, baum_welch]
        output_dict['EM seed'] = seed_setter
        
        output_dict['lambda_log'] = lambda_log
        output_dict['v_logs'] = v_logs
        output_dict['noise_temp'] = noise_temp
        output_dict['pi0_log'] = pi0_log
        
        #print(datetime.datetime.now().time())
        #print('end program')
        return output_dict
    
#%%
    def EM_with_priors(self, initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
                         eps, seed_setter):
        
        K = self.K
        W = self.W
        kappa = self.kappa
        signal_struct = self.processed_signals['Processed Signals']
        
# =============================================================================
#         A_log = initialised_parameters['A_log']
#         lambda_log = initialised_parameters['lambda_log']
#         noise_temp = initialised_parameters['noise_temp']
#         pi0_log = initialised_parameters['pi0_log']
#         v = initialised_parameters['v']
#         v_logs = initialised_parameters['v_logs']
# =============================================================================
        #######################################################################
            #A_log = initialised_parameters['A_log']
        A_init = np.zeros((2,2))
# =============================================================================
#     for j in np.arange(0,K):
#         A_init[:,j] = np.random.gamma(shape=1,scale=1,size=(1,2))
#         A_init[:,j] = A_init[:,j] / np.sum(A_init[:,j])
# =============================================================================
        A_init[0,0] = np.random.uniform(0.2,0.8)
        A_init[1,0] = 1 - A_init[0,0]
        A_init[0,1] = np.random.uniform(0.2,0.8)
        A_init[1,1] = 1- A_init[0,1]
        A_log = np.log(A_init)
        print('Main EM A init')
        print(np.exp(A_log))
        
        lambda_log = initialised_parameters['lambda_log']
        
        #noise_temp = initialised_parameters['noise_temp']
        noise_init = initialised_parameters['noise_temp']
        noise_min = 0.5 * noise_init
        noise_max = 2.0 * noise_init
        noise_range = noise_max - noise_min
        noise_temp = noise_min + np.random.uniform(0.1, 1) * noise_range
        
        pi0_log = initialised_parameters['pi0_log']
        v = initialised_parameters['v']
        
        v = np.reshape(v, (2,1))
        
        v[0,0] = (v[0,0] * (0.7 + 0.6 * np.random.uniform(0.1, 1))) * np.random.uniform(0.1, 0.2)
        v[1,0] = (v[1,0] * (0.7 + 0.6 * np.random.uniform(0.1, 1))) * np.random.uniform(0.1, 0.6)
        v_logs = initialised_parameters['v_logs']
        
        v_logs = np.reshape(v_logs, (2,1))
        #######################################################################
        # MS2 coefficient calculation
        ms2_coeff = ms2_loading_coeff(kappa, W)
        ms2_coeff_flipped = np.flip(ms2_coeff, 1)
        
        ms2_coeff_flipped = ms2_coeff #!!!!!!!!!!!!!!!!! HACK
        
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
            print('EM step number: ')
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
                #print(i_tr)
                #print(datetime.datetime.now())
                #print('start trace')
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
        
        
                # START FORWARD BACKWARD HERE
                gammas = {}
                off_off_container = []
                off_on_container = []
                on_off_container = []
                on_on_container = []
                
                forward_backward_results = forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container, off_off, off_on,
                     on_off, on_on, PERMITTED_MEMORY, trace_length,
                     log_likelihoods, logL_tot, baum_welch, i_tr)
                
                gammas = forward_backward_results['Gamma']
                off_off_container = forward_backward_results['off_off_results']
                off_on_container = forward_backward_results['off_on_results']
                on_off_container = forward_backward_results['on_off_results']
                on_on_container = forward_backward_results['on_on_results']
                
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
                
                #print(datetime.datetime.now().time())
                #print('before term_ith')
                term_ith = []
                for t in np.arange(0, trace_length):
                    for key in gammas_copy[t]:
                        adjusted_list = get_adjusted(int(key), K, W, ms2_coeff)
                        term_ith.append(gammas_copy[t][key] + np.log((data[0,t] - \
                                       (adjusted_list[1] * v[0, 0] + \
                                       adjusted_list[0] * v[1, 0]))**2))
                #print(datetime.datetime.now().time())
                #print('after term_ith')
                    
                flattened = np.asarray(term_ith)
                flattened = np.expand_dims(flattened, axis=0)
                test2 = np.expand_dims(np.expand_dims(np.array(lambda_terms), axis = 0), axis = 0)
                #test3 = np.concatenate((test2, flattened), axis = 0)
                test3 = np.concatenate((test2, flattened), axis = 1)
                test4 = scipy.special.logsumexp(test3)
                lambda_terms = test4
                
                gammas_cleaned = gammas_copy.copy() # HACK
                gammas_cleaned2 = []
                for g in gammas_cleaned: # HACK
                    g = {key:val for key, val in g.items() if val != np.NINF} # HACK
                    gammas_cleaned2.append(g)
                
                #log_F_terms = F_dict[trace_length]
                #print(datetime.datetime.now())
                #print('before v_M')
                terms_ith = []
                #failure = 0 # HACK
                #success = 0 # HACK
                #success_viewer = [] # HACK
                #failure_viewer = [] # HACK
                for m in np.arange(0,K):
                    for n in np.arange(0,K):
                        terms_ith = []                
                        for t in np.arange(0, trace_length):
                            for key in gammas_cleaned2[t]:
                                i_result = gammas_cleaned2[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[n][0,t] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t]
                                terms_ith.append(i_result)
        
        
                        filler = np.ones((1,1))
                        filler[0,0] = v_M_terms[m,n]
                        #v_M_terms[m,n] = scipy.special.logsumexp(np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0))
                        v_M_input = np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0) 
                        v_M_input = v_M_input[v_M_input != np.NINF]
                        v_M_terms[m,n] = logsumexp_numba(v_M_input)
                #print(datetime.datetime.now())       
                #print('after VM')
        
                #print(datetime.datetime.now().time())
                #print('before terms_b_log_ith')
                terms_b_log_ith = []
                sign_list = []
                tmp = np.ones((K,1))
                for m in np.arange(0,K):
                    terms_b_log_ith = []
                    sign_list = []
                    for t in np.arange(0, trace_length):
                        for key in gammas_copy[t]:
                            if gammas_copy[t][key] != np.NINF:
                                terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t])
                                #terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + log_F_terms[m][key,t])
                                sign_list.append(x_term_signs[0,t])
                    #print(datetime.datetime.now().time())
                    #print('mid terms_b_log_ith')        
                    reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
                    reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
                    signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
                    signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
                    assign1 = np.concatenate((np.reshape(np.array(v_b_terms_log[0,m]), (1,1)), reshaped2), axis = 1)
                    assign2 = np.concatenate((np.reshape(np.array(v_b_terms_sign[0,m]), (1,1)), signs2), axis = 1)
                    tmp = log_sum_exp(assign1, assign2)
                    v_b_terms_log[0,m] = tmp[0,]
                    v_b_terms_sign[0,m] = tmp[1,]
                    ##print(v_b_terms_sign)
                    ##print(datetime.datetime.now().time())
                    ##print('mid 2 terms_b_log_ith')
                #print(datetime.datetime.now().time())
                #print('after terms_b_log_ith')
                   
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
                #print('EXCEEDED')
                one_more = 1
                break
            
        output_dict = {}
        output_dict['A'] = np.exp(A_log)
        output_dict['pi0'] = np.exp(pi0_log)
        output_dict['v'] = np.exp(v_logs)
        output_dict['noise'] = np.exp(noise_log)
        output_dict['logL'] = logL_tot[0, baum_welch]
        output_dict['EM seed'] = seed_setter
        
        #print(datetime.datetime.now().time())
        #print('end program')
        return output_dict
    

    #%%
    def get_promoter_traces(self, initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
                         eps, seed_setter):
        
        K = self.K
        W = self.W
        kappa = self.kappa
        signal_struct = self.processed_signals['Processed Signals']
        
        A_log = np.log(initialised_parameters['A'])
        noise_temp = initialised_parameters['noise']
        lambda_log = -2 * np.log(noise_temp)
        pi0_log = np.log(initialised_parameters['pi0'])
        v = initialised_parameters['v']
        v_logs = np.log(v)
        
        v_logs = np.reshape(v_logs, (2,1))
        #######################################################################
        # MS2 coefficient calculation
        ms2_coeff = ms2_loading_coeff(kappa, W)
        ms2_coeff_flipped = np.flip(ms2_coeff, 1)
        
        ms2_coeff_flipped = ms2_coeff #!!!!!!!!!!!!!!!!! HACK
        
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
        
        
        p_z_log_soft = {}
        for baum_welch in range(n_steps):
            #print('baum_welch: ')
            #print(baum_welch)
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
                #print('start trace')
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
        
                # START FORWARD BACKWARD HERE
                
                forward_backward_results = forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container, off_off, off_on,
                     on_off, on_on, PERMITTED_MEMORY, trace_length,
                     log_likelihoods, logL_tot, baum_welch, i_tr)
                
                gammas = forward_backward_results['Gamma']
                off_off_container = forward_backward_results['off_off_results']
                off_on_container = forward_backward_results['off_on_results']
                on_off_container = forward_backward_results['on_off_results']
                on_on_container = forward_backward_results['on_on_results']
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
                
                #print(datetime.datetime.now().time())
                #print('before term_ith')
                term_ith = []
                for t in np.arange(0, trace_length):
                    for key in gammas_copy[t]:
                        adjusted_list = get_adjusted(int(key), K, W, ms2_coeff)
                        term_ith.append(gammas_copy[t][key] + np.log((data[0,t] - \
                                       (adjusted_list[1] * v[0, 0] + \
                                       adjusted_list[0] * v[1, 0]))**2))
                #print(datetime.datetime.now().time())
                #print('after term_ith')
                    
                flattened = np.asarray(term_ith)
                flattened = np.expand_dims(flattened, axis=0)
                test2 = np.expand_dims(np.expand_dims(np.array(lambda_terms), axis = 0), axis = 0)
                #test3 = np.concatenate((test2, flattened), axis = 0)
                test3 = np.concatenate((test2, flattened), axis = 1)
                test4 = scipy.special.logsumexp(test3)
                lambda_terms = test4
                
                gammas_cleaned = gammas_copy.copy() # HACK
                gammas_cleaned2 = []
                for g in gammas_cleaned: # HACK
                    g = {key:val for key, val in g.items() if val != np.NINF} # HACK
                    gammas_cleaned2.append(g)
                
                #log_F_terms = F_dict[trace_length]
                #print(datetime.datetime.now())
                #print('before v_M')
                terms_ith = []
                failure = 0 # HACK
                success = 0 # HACK
                success_viewer = [] # HACK
                failure_viewer = [] # HACK
                for m in np.arange(0,K):
                    for n in np.arange(0,K):
                        terms_ith = []                
                        for t in np.arange(0, trace_length):
                            for key in gammas_cleaned2[t]:
                                i_result = gammas_cleaned2[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[n][0,t] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t]
                                terms_ith.append(i_result)
        
        
                        filler = np.ones((1,1))
                        filler[0,0] = v_M_terms[m,n]
                        #v_M_terms[m,n] = scipy.special.logsumexp(np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0))
                        v_M_input = np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0) 
                        v_M_input = v_M_input[v_M_input != np.NINF]
                        v_M_terms[m,n] = logsumexp_numba(v_M_input)
                #print(datetime.datetime.now())       
                #print('after VM')
        
                #print(datetime.datetime.now().time())
                #print('before terms_b_log_ith')
                terms_b_log_ith = []
                sign_list = []
                tmp = np.ones((K,1))
                for m in np.arange(0,K):
                    terms_b_log_ith = []
                    sign_list = []
                    for t in np.arange(0, trace_length):
                        for key in gammas_copy[t]:
                            if gammas_copy[t][key] != np.NINF:
                                terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t])
                                #terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + log_F_terms[m][key,t])
                                sign_list.append(x_term_signs[0,t])
                    #print(datetime.datetime.now().time())
                    #print('mid terms_b_log_ith')        
                    reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
                    reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
                    signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
                    signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
                    assign1 = np.concatenate((np.reshape(np.array(v_b_terms_log[0,m]), (1,1)), reshaped2), axis = 1)
                    assign2 = np.concatenate((np.reshape(np.array(v_b_terms_sign[0,m]), (1,1)), signs2), axis = 1)
                    tmp = log_sum_exp(assign1, assign2)
                    v_b_terms_log[0,m] = tmp[0,]
                    v_b_terms_sign[0,m] = tmp[1,]
                    ##print(v_b_terms_sign)
                    ##print(datetime.datetime.now().time())
                    ##print('mid 2 terms_b_log_ith')
                #print(datetime.datetime.now().time())
                #print('after terms_b_log_ith')
                
                p_z_log_soft_singleTrace = np.zeros((K, trace_length))
                off_storage = []
                on_storage = []
                for t in np.arange(0, trace_length):
                    off_storage = []
                    on_storage = []
                    these_gammas = gammas_copy[t]
                    for key, value in these_gammas.items():
                        if key % 2 == 0:
                            off_storage.append(value)
                        else:
                            on_storage.append(value)
                    p_z_log_soft_singleTrace[0,t] = scipy.special.logsumexp(off_storage)
                    p_z_log_soft_singleTrace[1,t] = scipy.special.logsumexp(on_storage)
                p_z_log_soft[i_tr] = p_z_log_soft_singleTrace
                   
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
                #print('EXCEEDED')
                one_more = 1
                break
            
        output_dict = {}
        output_dict['A'] = np.exp(A_log)
        output_dict['pi0'] = np.exp(pi0_log)
        output_dict['v'] = np.exp(v_logs)
        output_dict['noise'] = np.exp(noise_log)
        #output_dict['logL'] = logL_tot[0, baum_welch]
        output_dict['logL'] = logL_tot
        output_dict['EM seed'] = seed_setter
        
        #print(datetime.datetime.now().time())
        #print('end program')
        return p_z_log_soft
    
    def exact_EM(self, initialised_parameters, n_steps, n_traces,
                     eps, seed_setter):
    
        K = self.K
        W = self.W
        kappa = self.kappa
        signal_struct = self.processed_signals['Processed Signals']
        
        A_log = initialised_parameters['A_log']
        lambda_log = initialised_parameters['lambda_log']
        noise_temp = initialised_parameters['noise_temp']
        pi0_log = initialised_parameters['pi0_log']
        v = initialised_parameters['v']
        v_logs = initialised_parameters['v_logs']
        
        # MS2 coefficient calculation
        ms2_coeff = ms2_loading_coeff(kappa, W)
        ms2_coeff_flipped = np.flip(ms2_coeff, 1)
        
        ms2_coeff_flipped = ms2_coeff #!!!!!!!!!!!!!!!!! HACK
        
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
            print('EM step number: ')
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
                #print(i_tr)
                #print(datetime.datetime.now())
                #print('start trace')
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
        
                
                # START FORWARD BACKWARD HERE
                
                forward_backward_results = exact_forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container, off_off, off_on,
                     on_off, on_on, trace_length,
                     log_likelihoods, logL_tot, baum_welch, i_tr)
                
                gammas = forward_backward_results['Gamma']
                off_off_container = forward_backward_results['off_off_results']
                off_on_container = forward_backward_results['off_on_results']
                on_off_container = forward_backward_results['on_off_results']
                on_on_container = forward_backward_results['on_on_results']
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
                
                #print(datetime.datetime.now().time())
                #print('before term_ith')
                term_ith = []
                for t in np.arange(0, trace_length):
                    for key in gammas_copy[t]:
                        adjusted_list = get_adjusted(int(key), K, W, ms2_coeff)
                        term_ith.append(gammas_copy[t][key] + np.log((data[0,t] - \
                                       (adjusted_list[1] * v[0, 0] + \
                                       adjusted_list[0] * v[1, 0]))**2))
                #print(datetime.datetime.now().time())
                #print('after term_ith')
                    
                flattened = np.asarray(term_ith)
                flattened = np.expand_dims(flattened, axis=0)
                test2 = np.expand_dims(np.expand_dims(np.array(lambda_terms), axis = 0), axis = 0)
                #test3 = np.concatenate((test2, flattened), axis = 0)
                test3 = np.concatenate((test2, flattened), axis = 1)
                test4 = scipy.special.logsumexp(test3)
                lambda_terms = test4
                
                gammas_cleaned = gammas_copy.copy() # HACK
                gammas_cleaned2 = []
                for g in gammas_cleaned: # HACK
                    g = {key:val for key, val in g.items() if val != np.NINF} # HACK
                    gammas_cleaned2.append(g)
                
                #log_F_terms = F_dict[trace_length]
                #print(datetime.datetime.now())
                #print('before v_M')
                terms_ith = []
                failure = 0 # HACK
                success = 0 # HACK
                success_viewer = [] # HACK
                failure_viewer = [] # HACK
                for m in np.arange(0,K):
                    for n in np.arange(0,K):
                        terms_ith = []                
                        for t in np.arange(0, trace_length):
                            for key in gammas_cleaned2[t]:
                                i_result = gammas_cleaned2[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[n][0,t] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t]
                                terms_ith.append(i_result)
        
        
                        filler = np.ones((1,1))
                        filler[0,0] = v_M_terms[m,n]
                        #v_M_terms[m,n] = scipy.special.logsumexp(np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0))
                        v_M_input = np.concatenate((np.expand_dims(np.asarray(terms_ith), axis = 1), filler), axis = 0) 
                        v_M_input = v_M_input[v_M_input != np.NINF]
                        v_M_terms[m,n] = logsumexp_numba(v_M_input)
                #print(datetime.datetime.now())       
                #print('after VM')
        
                #print(datetime.datetime.now().time())
                #print('before terms_b_log_ith')
                terms_b_log_ith = []
                sign_list = []
                tmp = np.ones((K,1))
                for m in np.arange(0,K):
                    terms_b_log_ith = []
                    sign_list = []
                    for t in np.arange(0, trace_length):
                        for key in gammas_copy[t]:
                            if gammas_copy[t][key] != np.NINF:
                                terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + compute_dynamic_F(key,trace_length, W, K, ms2_coeff_flipped, count_reduction_manual)[m][0,t])
                                #terms_b_log_ith.append(x_term_logs[0,t] + gammas_copy[t][key] + log_F_terms[m][key,t])
                                sign_list.append(x_term_signs[0,t])
                    #print(datetime.datetime.now().time())
                    #print('mid terms_b_log_ith')        
                    reshaped = np.reshape(np.asarray(terms_b_log_ith), (1,len(np.asarray(terms_b_log_ith))))
                    reshaped2 = np.reshape(reshaped, (1,np.size(reshaped)))
                    signs_unpacked = np.reshape(np.asarray(sign_list), (1,len(np.asarray(sign_list))))
                    signs2 = np.reshape(signs_unpacked, (1,np.size(signs_unpacked)))
                    assign1 = np.concatenate((np.reshape(np.array(v_b_terms_log[0,m]), (1,1)), reshaped2), axis = 1)
                    assign2 = np.concatenate((np.reshape(np.array(v_b_terms_sign[0,m]), (1,1)), signs2), axis = 1)
                    tmp = log_sum_exp(assign1, assign2)
                    v_b_terms_log[0,m] = tmp[0,]
                    v_b_terms_sign[0,m] = tmp[1,]
                    ##print(v_b_terms_sign)
                    ##print(datetime.datetime.now().time())
                    ##print('mid 2 terms_b_log_ith')
                #print(datetime.datetime.now().time())
                #print('after terms_b_log_ith')
                   
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
                #print('EXCEEDED')
                one_more = 1
                break
            
        output_dict = {}
        output_dict['A'] = np.exp(A_log)
        output_dict['pi0'] = np.exp(pi0_log)
        output_dict['v'] = np.exp(v_logs)
        output_dict['noise'] = np.exp(noise_log)
        output_dict['logL'] = logL_tot[0, baum_welch]
        output_dict['EM seed'] = seed_setter
        
        #print(datetime.datetime.now().time())
        #print('end program')
        return output_dict
    
