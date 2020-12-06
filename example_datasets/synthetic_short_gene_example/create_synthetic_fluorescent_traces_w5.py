# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 03:32:57 2020

@author: Jon
"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
#from numba import jit
#from burstinfer.compute_dynamic_F import compute_dynamic_F
from burstInfer.get_adjusted import get_adjusted
from burstInfer.ms2_loading_coeff import ms2_loading_coeff


#seed_setter = np.random.randint(0,1000000000) # Make sure this is random for actual run
seed_setter = 633463
np.random.seed(seed_setter)
np.seterr(divide='ignore')

# Import data
signal_holder = genfromtxt('very_short_gene_w5.csv', delimiter=',', skip_header=0)

synthetic_x = np.arange(0,100)

plt.figure(0)
plt.step(synthetic_x, signal_holder[40,:])

# Initialisation
K = 2
n_traces = len(signal_holder)
W = 5
compound_states = K**W

mu = np.zeros((K,1))
mu[0,0] = 150
mu[1,0] = 8000
noise = 3000

t_MS2 = 30
deltaT = 20
kappa = t_MS2 / deltaT

unique_lengths = np.expand_dims(np.asarray(100), axis = 0)
trace_length = 100

#%%
# MS2 coefficient calculation
ms2_coeff = ms2_loading_coeff(kappa, W)
ms2_coeff_flipped = np.flip(ms2_coeff, 1)
count_reduction_manual = np.zeros((1,W-1))
for t in np.arange(0,W-1):
    count_reduction_manual[0,t] = np.sum(ms2_coeff[0,t+1:])
count_reduction_manual = np.reshape(count_reduction_manual, (W-1,1))


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

plt.figure(2)
plt.plot(synthetic_x, single_trace.flatten(), c='b')


sampling_dataframe = pd.DataFrame(fluorescence_holder)

sampling_dataframe.to_csv("very_short_gene_w5_fluorescent_traces.csv")


#transition_probabilities = [0.9 0.1;0.35 0.65];

#%%
for j in np.arange(3,15):
    plt.figure(j)
    plt.plot(synthetic_x, fluorescence_holder[j,:].flatten())
    
# =============================================================================
# plt.figure(15)
# plt.step(synthetic_x, signal_holder[40,:])
# plt.figure(16)
# plt.plot(synthetic_x, fluorescence_holder[40,:].flatten())
# 
# plt.figure(17)
# plt.step(synthetic_x, signal_holder[42,:])
# plt.figure(18)
# plt.plot(synthetic_x, fluorescence_holder[42,:].flatten())
# =============================================================================