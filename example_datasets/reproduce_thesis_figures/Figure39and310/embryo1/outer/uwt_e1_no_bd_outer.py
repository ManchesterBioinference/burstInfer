# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 04:24:55 2020

@author: MBGM9JBC
"""
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from process_data import process_data
from initialise_parameters import initialise_parameters
from EM import EM
from fixed_EM import fixed_EM
from probe_adjustment import probe_adjustment
from pre_load_parameters import pre_load_parameters
from EM_with_estimates import EM_with_estimates
from export_parameters import export_parameters
from process_raw_data import process_raw_data

# Configuration
#seed_setter = 609660386 # TODO
seed_setter = np.random.randint(0,1000000000)
np.random.seed(seed_setter)
np.seterr(divide='ignore')
plt.style.use('dark_background')

# Import and process data
signal_holder = genfromtxt('uwt_e1_no_bd.csv', delimiter=',', skip_header=1) # TODO
signal_holder = signal_holder[:,1:]

###############################

sorted_by_means = signal_holder[signal_holder[:,2].argsort()]

non_excluded = sorted_by_means[21:,:]


sorted_by_midline = non_excluded[np.sqrt(np.abs(non_excluded[:,3])).argsort()]
non_excluded = sorted_by_midline

centre_mean = 15
inter_mean = 30
outer_mean = 45

centre = non_excluded[(np.sqrt(np.abs(non_excluded[:,3])) < centre_mean)]
inter = non_excluded[(np.sqrt(np.abs(non_excluded[:,3])) < inter_mean) & (np.sqrt(np.abs(non_excluded[:,3])) > centre_mean)]
outer = non_excluded[(np.sqrt(np.abs(non_excluded[:,3])) > inter_mean)]


non_excluded = outer

# =============================================================================
# plt.figure(6)
# plt.scatter(np.sqrt(np.abs(centre[:,3])), centre[:,2],c='r')
# plt.scatter(np.sqrt(np.abs(inter[:,3])), inter[:,2],c='b')
# plt.scatter(np.sqrt(np.abs(outer[:,3])), outer[:,2],c='g')
# plt.title('ushWT EMBRYO 1')
# plt.xlim((0,70))
# =============================================================================



#processed_signals = process_data(non_excluded)
processed_signals = process_raw_data(non_excluded,18)
###############################

signal_struct = processed_signals['Processed Signals']
unique_lengths = processed_signals['Signal Lengths']


# Set up HMM
K = 2
n_traces = len(signal_struct)
W = 19 # TODO
eps = 10**(-3)
compound_states = K**W
n_steps = 50
PERMITTED_MEMORY = 128 # TODO
t_MS2 = 30 # TODO
deltaT = 20 # TODO
kappa = t_MS2 / deltaT


initialised_parameters = initialise_parameters(K, W, processed_signals['Matrix Max'], processed_signals['Matrix Mean'])

#%%

# Run 'Medium Window' EM
# Below can also go inside
probe_list = probe_adjustment(K, W, kappa, unique_lengths)
adjusted_ones = probe_list[0]
adjusted_zeros = probe_list[1]
F_dict = probe_list[2]

#preloaded_parameters = pre_load_parameters()

parameter_priors = fixed_EM(initialised_parameters, n_steps, n_traces, signal_struct, compound_states, K,
                PERMITTED_MEMORY, W, F_dict, adjusted_ones, adjusted_zeros, eps*10, seed_setter)


parameters = EM_with_estimates(parameter_priors, n_steps, n_traces, signal_struct, compound_states, K,
                PERMITTED_MEMORY, W, F_dict, adjusted_ones, adjusted_zeros, eps, seed_setter)

#parameters = EM(initialised_parameters, n_steps, n_traces, signal_struct, compound_states, K,
#                PERMITTED_MEMORY, W, F_dict, adjusted_ones, adjusted_zeros, eps, seed_setter)


#%%

# Export parameters
parameters_for_export = export_parameters(parameters)
results_df = pd.DataFrame(parameters_for_export)
results_df.to_csv('result_' + str(seed_setter) + '.csv', header=None, index=False)


