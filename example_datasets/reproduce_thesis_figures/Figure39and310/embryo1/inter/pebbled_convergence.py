# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:17:24 2020

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
signal_holder = genfromtxt('pebWT_full_embryo_processed_2_clusters_interpolated.csv', delimiter=',', skip_header=1) # TODO
signal_holder = signal_holder[:,1:]

###############################

sorted_by_means = signal_holder[signal_holder[:,2].argsort()]

non_excluded = sorted_by_means[20:,:]
#non_excluded = non_excluded[non_excluded[:,1] == 2]
non_excluded = non_excluded[110:,35:]

processed_signals = process_data(non_excluded)
#processed_signals = process_raw_data(non_excluded, 12)
###############################

signal_struct = processed_signals['Processed Signals']
unique_lengths = processed_signals['Signal Lengths']


# Set up HMM
K = 2
n_traces = len(signal_struct)
W = 9 # TODO
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

parameter_priors = fixed_EM(initialised_parameters, PERMITTED_MEMORY, n_traces, signal_struct, compound_states, K,
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