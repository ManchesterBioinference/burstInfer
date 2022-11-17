# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 03:32:57 2020

@author: Jon
"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit

def ms2_loading_coeff(kappa,W):
    alpha = kappa
    coeff = np.ones((1,W), dtype=float)
    
    alpha_ceil = np.ceil(alpha)
    alpha_floor = np.floor(alpha)
    
    coeff[0:int(alpha_floor):1,0] = (np.linspace(1,int(alpha_floor), endpoint=True, num=int(alpha_floor)) - 0.5) / alpha
    
    coeff[0,int(alpha_ceil)-1] = (alpha_ceil - alpha) + (alpha**2 - (alpha_ceil-1)**2) / (2*alpha)
    
    return coeff


#seed_setter = np.random.randint(0,1000000000) # Make sure this is random for actual run
seed_setter = 957434
np.random.seed(seed_setter)
np.seterr(divide='ignore')

# Import data
signal_holder = genfromtxt('synthetic_promoter_traces_w13.csv', delimiter=',', skip_header=0)

synthetic_x = np.arange(0,100)

# =============================================================================
# plt.figure(0)
# plt.step(synthetic_x, signal_holder[40,:])
# =============================================================================

# Initialisation
K = 2
n_traces = len(signal_holder)
W = 13
compound_states = K**W

mu = np.zeros((K,1))
mu[0,0] = 282.27
mu[1,0] = 13889.8
#noise = 16414.9
noise = 32828

t_MS2 = 30
deltaT = 20
kappa = t_MS2 / deltaT

unique_lengths = np.expand_dims(np.asarray(100), axis = 0)
trace_length = 100

#%%
# More efficient stuff
# MS2 coefficient calculation
ms2_coeff = ms2_loading_coeff(kappa, W)
ms2_coeff_flipped = np.flip(ms2_coeff, 1)
count_reduction_manual = np.zeros((1,W-1))
for t in np.arange(0,W-1):
    count_reduction_manual[0,t] = np.sum(ms2_coeff[0,t+1:])
count_reduction_manual = np.reshape(count_reduction_manual, (W-1,1))


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


mask = np.int32((2**W)-1)

fluorescence_holder = np.zeros((100,100))

for i in np.arange(0, len(fluorescence_holder)):
    single_promoter = np.expand_dims(signal_holder[i,:], axis = 0)
    
    single_trace = np.zeros((1,100))
    
    t = 0
    
    window_storage = int(single_promoter[0,0])
    #single_trace[0,t] = ((F_on_viewer[window_storage, t] * mu[1,0]) + (F_off_viewer[window_storage, t] * mu[0,0])) + np.random.normal(0, noise)
    single_trace[0,t] = ((get_adjusted(window_storage, K, W, ms2_coeff)[0] * mu[1,0]) + (get_adjusted(window_storage, K, W, ms2_coeff)[1] * mu[0,0])) +  + np.random.normal(0, noise)
    
    window_storage = 0
    t = 1
    present_state_list = []
    present_state_list.append(int(single_promoter[0,0]))
    #while t < W:
    while t < 100:
        present_state = int(single_promoter[0,t])
        #print('present state')
        #print(present_state)
        #present_state_list.append(present_state)
        window_storage = np.bitwise_and((present_state_list[t-1] << 1) + present_state, mask)
        #print('window storage')
        #print(window_storage)
        present_state_list.append(window_storage)
         
        #single_trace[0,t] = ((F_on_viewer[window_storage, t] * mu[1,0]) + (F_off_viewer[window_storage, t] * mu[0,0])) + np.random.normal(0, noise)
        single_trace[0,t] = ((get_adjusted(window_storage, K, W, ms2_coeff)[0] * mu[1,0]) + (get_adjusted(window_storage, K, W, ms2_coeff)[1] * mu[0,0])) +  + np.random.normal(0, noise)
        
        t = t + 1
        
    fluorescence_holder[i,:] = single_trace

# =============================================================================
# plt.figure(2)
# plt.plot(synthetic_x, single_trace.flatten(), c='b')
# =============================================================================


sampling_dataframe = pd.DataFrame(fluorescence_holder)

sampling_dataframe.to_csv("synthetic_fluorescent_traces_w13.csv")


#transition_probabilities = [0.9 0.1;0.35 0.65];

#%%
# =============================================================================
# for j in np.arange(3,15):
#     plt.figure(j)
#     plt.plot(synthetic_x, fluorescence_holder[j,:].flatten())
#     
# plt.figure(15)
# plt.step(synthetic_x, signal_holder[40,:])
# plt.figure(16)
# plt.plot(synthetic_x, fluorescence_holder[40,:].flatten())
# 
# plt.figure(17)
# plt.step(synthetic_x, signal_holder[42,:])
# plt.figure(18)
# plt.plot(synthetic_x, fluorescence_holder[42,:].flatten())
# 
# =============================================================================
plt.figure(4)
plt.plot(synthetic_x, fluorescence_holder[26,:].flatten())
plt.show()