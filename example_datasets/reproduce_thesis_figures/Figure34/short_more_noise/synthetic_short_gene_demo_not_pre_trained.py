# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:57:40 2020

@author: Jon
"""
#import sys
#sys.path.append("../../../burstInfer")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt
from scipy import special
#import burstInfer
from burstInfer.process_raw_data import process_raw_data
from burstInfer.HMM import HMM
from burstInfer.export_em_parameters import export_em_parameters

#%%
# Basic configuration
#seed_setter = 737978298 # Use specific seed for this demo
seed_setter = np.random.randint(0,1000000000) # When actually training set random seed
np.random.seed(seed_setter)
np.seterr(divide='ignore')
plt.style.use('dark_background')

#%%
# Import and process data (synthetic fluorescent traces)
ms2_signals = genfromtxt('very_short_gene_w5_fluorescent_traces.csv', delimiter=',', skip_header=1) # TODO
# Strip away leading column of row indices
ms2_signals = ms2_signals[:,1:]
# Create dictionary containing list of signals (integer is which column to start
# with, 0 in our case)
processed_signals = process_raw_data(ms2_signals,0)

#%%
# Set up HMM parameters. 
K = 2 # Number of allowed promoter states - always 2 ATM.
n_traces = len(processed_signals['Processed Signals'])
eps = 10**(-3) # Error tolerance
n_steps = 50 # Number of maximum EM steps
PERMITTED_MEMORY = 8 # Number of allowed compound states

# The parameters will probably need to be altered.
W = 5 # Window size
t_MS2 = 30 # Time for Pol II to traverse MS2 probe (s)
deltaT = 20 # Time resolution (s)
kappa = t_MS2 / deltaT
compound_states = K**W

#%%
# Create HMM object
demoHMM = HMM(K, W, t_MS2, deltaT, kappa, compound_states, processed_signals)

# initialise parameters
initialised_parameters = demoHMM.initialise_parameters()


# Method A
# Use initialised parameters to train model with no further estimates
#learned_parameters = demoHMM.EM(initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
#                         eps, seed_setter)


# Method B
# Fix transitions and improve initialised emission and noise parameter estimates
parameter_priors = demoHMM.EM_fixed(initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
                         eps*10, seed_setter)

# Use these estimates to train model
learned_parameters = demoHMM.EM_with_priors(parameter_priors, n_steps, n_traces, PERMITTED_MEMORY,
                         eps, seed_setter)

#%%
# Export inferred parameters as csv. The file name includes the seed used.
# Each separate script will generate one results file, which can then be
# collected together and processed.
parameters_for_export = export_em_parameters(learned_parameters)
results_df = pd.DataFrame(parameters_for_export)
results_df.to_csv('result_' + str(seed_setter) + '.csv', header=True, index=False)