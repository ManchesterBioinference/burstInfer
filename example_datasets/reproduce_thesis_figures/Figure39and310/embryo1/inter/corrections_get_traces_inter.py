# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:35:19 2021

@author: MBGM9JBC
"""
import pandas as pd
from glob import glob
import scipy
import numpy as np
from scipy import linalg

# Import and process data
df_list = []
deltaT = 20

for file in glob('result_*.csv'):
    data_row = pd.read_csv(file, header=None)
    df_list.append(data_row)
    
results_dataframe = pd.concat(df_list)
results_dataframe.columns = ['Random Seed', 'p_off_off', 'p_off_on', 'p_on_off', 'p_on_on',
                    'pi0_off', 'pi0_on', 'mu_off', 'mu_on', 'noise', 'logL']
results_dataframe.reset_index(inplace=True, drop=True)


anomaly_list1 = results_dataframe.index[results_dataframe['p_off_off'] == 0].tolist()
anomaly_list2 = results_dataframe.index[results_dataframe['p_off_off'] == 1].tolist()
anomaly_list3 = results_dataframe.index[results_dataframe['p_on_on'] == 0].tolist()
anomaly_list4 = results_dataframe.index[results_dataframe['p_on_on'] == 1].tolist()

anomaly_metalist = anomaly_list1 + anomaly_list2 + anomaly_list3 + anomaly_list4

if len(anomaly_metalist) !=0:
    max_anomaly = max(anomaly_metalist)
    results_dataframe2 = results_dataframe.iloc[0:max_anomaly,:]
else:
    results_dataframe2 = results_dataframe

filter1 = results_dataframe2[results_dataframe2['p_off_off']!= 0]
filter2 = filter1[filter1['p_off_off']!= 1]
filter3 = filter2[filter2['p_on_on']!= 0]
filter4 = filter3[filter3['p_on_on']!= 1]

filtered_dataframe = filter4

sorted_results = filtered_dataframe.sort_values(by='logL')

#max_likelihood_estimate = sorted_results.iloc[-1,:]
max_likelihood_estimate = results_dataframe.iloc[1,:]

kon_list = []
koff_list = []
trans_matrix = np.zeros((2,2))
trans_matrix[0,0] = max_likelihood_estimate.iloc[1,]
trans_matrix[1,0] = max_likelihood_estimate.iloc[2,]
trans_matrix[0,1] = max_likelihood_estimate.iloc[3,]
trans_matrix[1,1] = max_likelihood_estimate.iloc[4,]
rates = (scipy.linalg.logm(trans_matrix) / deltaT) * 60
kon_list.append(rates[1,0])
koff_list.append(rates[0,1])
max_likelihood_estimate['kon'] = kon_list[0]
max_likelihood_estimate['koff'] = koff_list[0]

print(max_likelihood_estimate)

#%%
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


non_excluded = inter

# =============================================================================
# plt.figure(6)
# plt.scatter(np.sqrt(np.abs(centre[:,3])), centre[:,2],c='r')
# plt.scatter(np.sqrt(np.abs(inter[:,3])), inter[:,2],c='b')
# plt.scatter(np.sqrt(np.abs(outer[:,3])), outer[:,2],c='g')
# plt.title('ushWT EMBRYO 1')
# plt.xlim((0,70))
# =============================================================================



#processed_signals = process_data(non_excluded)
processed_signals = process_raw_data(non_excluded,12)
###############################

signal_struct = processed_signals['Processed Signals']
unique_lengths = processed_signals['Signal Lengths']


# Set up HMM
K = 2
n_traces = len(signal_struct)
W = 19 # TODO
eps = 10**(-3)
compound_states = K**W
n_steps = 1
PERMITTED_MEMORY = 128 # TODO
t_MS2 = 30 # TODO
deltaT = 20 # TODO
kappa = t_MS2 / deltaT


#%%

seed_setter = max_likelihood_estimate['Random Seed']

# Run 'Medium Window' EM
# Below can also go inside
probe_list = probe_adjustment(K, W, kappa, unique_lengths)
adjusted_ones = probe_list[0]
adjusted_zeros = probe_list[1]
F_dict = probe_list[2]

max_pi0 = np.zeros((2,1))
max_pi0[0,0] = max_likelihood_estimate['pi0_off']
max_pi0[1,0] = max_likelihood_estimate['pi0_on']
v_est = np.zeros((2,1))
v_est[0,0] = max_likelihood_estimate['mu_off']
v_est[1,0] = max_likelihood_estimate['mu_on']

initialised_parameters = {}
initialised_parameters['A'] = trans_matrix
initialised_parameters['noise'] = max_likelihood_estimate['noise'] 
initialised_parameters['pi0'] = max_pi0
initialised_parameters['v'] = v_est

from get_posterior_complete import get_posterior_complete
print('Computing posterior')
posterior_traces = get_posterior_complete(initialised_parameters, n_steps, n_traces, signal_struct, compound_states, K,
                PERMITTED_MEMORY, W, F_dict, adjusted_ones, adjusted_zeros, eps, seed_setter)

#%%
exponentiated_posterior_list = []
posterior_occupancies = []
for j in np.arange(0,len(posterior_traces)):
    extracted_posterior = posterior_traces[j]
    exponentiated_posterior = np.exp(extracted_posterior)
    exponentiated_posterior_list.append(exponentiated_posterior)
    extracted_posterior_occupancies = np.mean(exponentiated_posterior,axis=1)
    posterior_occupancies.append(extracted_posterior_occupancies[1,])
    
#%%
max_posterior_list = []
for p in np.arange(0, len(exponentiated_posterior_list)):
    exp_posterior = exponentiated_posterior_list[p]
    max_posterior = np.argmax(exp_posterior, axis=0)
    max_posterior_list.append(max_posterior)
    
export_posterior_df = pd.DataFrame(max_posterior_list)
export_posterior_df.to_csv('corrections_traces_inter.csv')