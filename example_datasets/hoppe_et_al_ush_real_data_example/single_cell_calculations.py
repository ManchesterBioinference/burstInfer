# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:11:27 2020

@author: Jon
"""
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt
from burstInfer.process_raw_data import process_raw_data
from burstInfer.HMM import HMM
from scipy import special

np.seterr(divide='ignore')
plt.style.use('dark_background')


# Import ML parameters
max_likelihood_estimate = pd.read_csv('model_maximum_likelihood_estimate.csv', header=None, index_col=0)

#%%
# Import and process data (synthetic fluorescent traces)
ms2_signals = genfromtxt('uwt_e1_no_bd.csv', delimiter=',', skip_header=1) # TODO
# Strip away leading column of row indices
signal_holder = ms2_signals[:,1:]

# Sort signals by mean value (to identify very weak signals)
sorted_by_means = signal_holder[signal_holder[:,2].argsort()]
# Filter out weak signals
weak_signals_removed = sorted_by_means[20:,:]
# Select only cluster / region of interest from whole embryo
filtered_by_cluster = weak_signals_removed[weak_signals_removed[:,1] == 0]
# Generate dict of signals for input to model
processed_signals = process_raw_data(filtered_by_cluster, 11)

#%%
# Set up HMM parameters. 
K = 2 # Number of allowed promoter states - always 2 ATM.
n_traces = len(processed_signals['Processed Signals'])
eps = 10**(-3) # Error tolerance
n_steps = 50 # Number of maximum EM steps
PERMITTED_MEMORY = 256 # Number of allowed compound states

# The parameters will probably need to be altered.
W = 19 # Window size
t_MS2 = 30 # Time for Pol II to traverse MS2 probe (s)
deltaT = 20 # Time resolution (s)
kappa = t_MS2 / deltaT
compound_states = K**W

seed_setter = int(max_likelihood_estimate.loc['Random Seed'])
np.random.seed(seed_setter)

#%%
demoHMM = HMM(K, W, t_MS2, deltaT, kappa, compound_states, processed_signals)

learned_parameters = {}
transitions = np.ones((2,2))
transitions[0,0] = max_likelihood_estimate.loc['p_off_off']
transitions[1,0] = max_likelihood_estimate.loc['p_off_on']
transitions[1,1] = max_likelihood_estimate.loc['p_on_on']
transitions[0,1] = max_likelihood_estimate.loc['p_on_off']
learned_parameters['A'] = transitions
pi0 = np.ones((1,2))
pi0[0,0] = max_likelihood_estimate.loc['pi0_off']
pi0[0,1] = max_likelihood_estimate.loc['pi0_on']
learned_parameters['pi0'] = pi0
v = np.ones((2,))
v[0,] = max_likelihood_estimate.loc['mu_off']
v[1,] = max_likelihood_estimate.loc['mu_on']
learned_parameters['v'] = v
learned_parameters['noise'] = float(max_likelihood_estimate.loc['noise'])
learned_parameters['logL'] = max_likelihood_estimate.loc['logL']
learned_parameters['EM seed'] = seed_setter

posterior_traces = demoHMM.get_promoter_traces(learned_parameters, 1, n_traces, PERMITTED_MEMORY,
                         eps, seed_setter)

#%%
# Convert posterior probabilities to binary sequences
exponentiated_posterior_list = []
posterior_occupancies = []
for j in np.arange(0,len(posterior_traces)):
    extracted_posterior = posterior_traces[j]
    exponentiated_posterior = np.exp(extracted_posterior)
    exponentiated_posterior_list.append(exponentiated_posterior)
    extracted_posterior_occupancies = np.mean(exponentiated_posterior,axis=1)
    posterior_occupancies.append(extracted_posterior_occupancies[1,])
    
max_posterior_list = []
for p in np.arange(0, len(exponentiated_posterior_list)):
    exp_posterior = exponentiated_posterior_list[p]
    max_posterior = np.argmax(exp_posterior, axis=0)
    max_posterior_list.append(max_posterior)
    
#%%
from burstInfer.get_single_cell_emission import get_single_cell_emission

# Calculate single cell emission
reconstituted_signals = filtered_by_cluster[:,7:]
reconstituted_posterior = np.full((np.shape(reconstituted_signals)[0], np.shape(reconstituted_signals)[1]), np.nan)
for j in np.arange(0, np.shape(reconstituted_signals)[0]):
    fetched_posterior = max_posterior_list[j]
    reconstituted_posterior[j,0:len(fetched_posterior)] = fetched_posterior
    
sc_emission = get_single_cell_emission(K, W, kappa, reconstituted_posterior, reconstituted_signals)

#%%
# Calculate single cell parameters using inferred promoter traces
from burstInfer.calculate_single_cell_transition_rates import calculate_single_cell_transition_rates

single_cell_rates = calculate_single_cell_transition_rates(max_posterior_list, deltaT, 3)

#%%
# Export SC results

emission_df = pd.DataFrame(sc_emission)
occupancy_df = pd.DataFrame(posterior_occupancies)

old_style_kon_df = pd.DataFrame(single_cell_rates['Single Cell koff'])
old_style_koff_df = pd.DataFrame(single_cell_rates['Single Cell kon'])
rate_problem_list = single_cell_rates['Problematic Cells']

emission_df.drop(emission_df.index[rate_problem_list],inplace=True)
occupancy_df.drop(occupancy_df.index[rate_problem_list],inplace=True)

export_df = pd.concat([emission_df, occupancy_df, old_style_kon_df, old_style_koff_df], axis=1)
export_df_titles=['Emission', 'Occupancy', 'kon', 'koff']
export_df.columns = export_df_titles


export_df.to_csv('SINGLE_CELL_PARAMETERS.csv')
pd.DataFrame(reconstituted_posterior).to_csv('SINGLE_CELL_POSTERIOR.csv')
pd.DataFrame(reconstituted_signals).to_csv('SINGLE_CELL_SIGNALS.csv')