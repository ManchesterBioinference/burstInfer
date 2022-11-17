# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:38:49 2020

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
max_likelihood_estimate = results_dataframe.iloc[46,:]

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
from get_p_zz import get_p_zz

transition_probabilities = get_p_zz(initialised_parameters, n_steps, n_traces, signal_struct, compound_states, K,
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
exponentiated_transition_probabilities_list = []
p_off_list = []
p_on_list = []
for m in np.arange(0, len(transition_probabilities)):
    extracted_transition_probabilities = transition_probabilities[m]
    exponentiated_transition_probabilities = np.exp(extracted_transition_probabilities)
    exponentiated_transition_probabilities_list.append(exponentiated_transition_probabilities)
    p_off = exponentiated_transition_probabilities[0,] + exponentiated_transition_probabilities[1,]
    p_on = exponentiated_transition_probabilities[2,] + exponentiated_transition_probabilities[3,]
    p_off_list.append(p_off)
    p_on_list.append(p_on)

posterior_transition_occupancies = []
for n in np.arange(0, len(p_on_list)):
    extracted_on_probabilities = p_on_list[n]
    mean_extracted_on_probability = np.mean(extracted_on_probabilities)
    posterior_transition_occupancies.append(mean_extracted_on_probability)

#%%
max_posterior_list = []
for p in np.arange(0, len(exponentiated_posterior_list)):
    exp_posterior = exponentiated_posterior_list[p]
    max_posterior = np.argmax(exp_posterior, axis=0)
    max_posterior_list.append(max_posterior)

#%%
on_prob_list = []
off_prob_list = []
transition_matrix = np.zeros((2,2))
for i in np.arange(0, len(exponentiated_transition_probabilities_list)):
    print(i)
    brought = exponentiated_transition_probabilities_list[i]
    brought_rate = np.mean(brought, axis=1)
    transition_matrix[0,0] = brought_rate[0,]
    transition_matrix[1,0] = brought_rate[1,]
    transition_matrix[0,1] = brought_rate[2,]
    transition_matrix[1,1] = brought_rate[3,]
    rate_matrix = (scipy.linalg.logm(transition_matrix) / deltaT) * 60
    on_prob_list.append(rate_matrix[1,0])
    off_prob_list.append(rate_matrix[0,1])
    
plt.figure(8)
plt.scatter(non_excluded[:,5], on_prob_list,c='g')

plt.figure(9)
plt.scatter(non_excluded[:,5], off_prob_list,c='g')

#%%
from get_single_cell_emission import get_single_cell_emission

signal_keys = non_excluded[:,0:7]
signal_body = non_excluded[:,18:]
reconstituted_signals = np.concatenate((signal_keys, signal_body), axis=1)

reconstituted_posterior = np.full((np.shape(reconstituted_signals)[0], np.shape(reconstituted_signals)[1]), np.nan)
reconstituted_posterior[:,0:7] = reconstituted_signals[:,0:7]

for j in np.arange(0, np.shape(reconstituted_signals)[0]):
    fetched_posterior = max_posterior_list[j]
    reconstituted_posterior[j,7:len(fetched_posterior)+7] = fetched_posterior


emission_old_technique = get_single_cell_emission(K, W, kappa, reconstituted_posterior[:,7:], reconstituted_signals[:,7:])

#%%
# old style kon / koff 
off_on = []
on_off = []

off_off = []
on_on = []

length_holder = []

for j in np.arange(0, len(max_posterior_list)):
    fetched_posterior = max_posterior_list[j]
    fetched_posterior2 = fetched_posterior[~np.isnan(fetched_posterior)]
    off_on_interior = 0
    on_off_interior = 0
    length_holder.append(len(fetched_posterior2)-1)
    
    off_off_interior = 0
    on_on_interior = 0
    for k in np.arange(1, len(fetched_posterior2)):
        digit = fetched_posterior2[k,]
        digit_m1 = fetched_posterior2[k-1]
    
        if digit == 1 and digit_m1 == 0:
            off_on_interior +=1
        elif digit == 0 and digit_m1 == 1:
            on_off_interior +=1
        elif digit == 0 and digit_m1 == 0:
            off_off_interior +=1
        elif digit == 1 and digit_m1 == 1:
            on_on_interior +=1
    
      
    off_on.append(off_on_interior)
    on_off.append(on_off_interior)
    off_off.append(off_off_interior)
    on_on.append(on_on_interior)
    

on_on_array = np.array(on_on)
on_off_array = np.array(on_off)
off_on_array = np.array(off_on)
off_off_array = np.array(off_off)

on_on_sum = np.sum(on_on_array)
on_off_sum = np.sum(on_off_array)
off_on_sum = np.sum(off_on_array)
off_off_sum = np.sum(off_off_array)

n_trials = sum(length_holder)

A = 3 # TODO
on_on_frac = (on_on_sum / n_trials) * A
on_off_frac = (on_off_sum / n_trials) * A
off_on_frac = (off_on_sum / n_trials) * A
off_off_frac = (off_off_sum / n_trials) * A


real_kon = np.divide(off_on_array + off_on_frac, (off_off_array + off_on_array + off_off_frac + off_on_frac))
real_koff = np.divide(on_off_array + on_off_frac, (on_on_array + on_off_array + on_off_frac + on_on_frac))
p_on_on = np.divide(on_on_array + on_on_frac, (on_on_array + on_off_array + on_on_frac + on_off_frac))
p_off_off = np.divide(off_off_array + off_off_frac, (off_off_array + off_on_array + off_off_frac + off_on_frac))

default_kon = off_on_frac / (off_off_frac + off_on_frac)
default_koff = on_off_frac / (on_off_frac + on_on_frac)
default_p_on_on = on_on_frac / (on_on_frac + on_off_frac)
default_p_off_off = off_off_frac / (off_off_frac + off_on_frac)


artificial_transition_list = []
art_trans_holder = []
art_trans_processed_holder = []
expression_list = []
occupancy_list2 = []
label_holder = []

for i in np.arange(0, len(real_kon)):
    inner_list = []
    art_trans = np.zeros((2,2))
    ind_kon = real_kon[i,]
    ind_koff = real_koff[i,]
    art_trans[1,0] = ind_kon
    art_trans[0,0] = 1 - ind_kon
    art_trans[0,1] = ind_koff
    art_trans[1,1] = 1 - ind_koff
    
    #data_row = number_array[i,:]
    
    #if ~(np.isnan(art_trans).any()):
    #inner_list.append(data_row)
    inner_list.append((scipy.linalg.logm(art_trans) / 19.60784) * 60)
    artificial_transition_list.append(inner_list)
    art_trans_processed_holder.append((scipy.linalg.logm(art_trans) / 19.60784) * 60)
    art_trans_holder.append(art_trans)
    #expression_list.append(number_array[i,3])
    #occupancy_list2.append(occupancy_list[i])
    #label_holder.append(number_array[i,2])
    

kon_rates = []
koff_rates = []
cell_tags = []

midline_tags = []

rate_problem_list = []
for j in np.arange(0, len(artificial_transition_list)):
    print(artificial_transition_list[j])
    if np.any(np.iscomplex(artificial_transition_list[j])) != True:
        trans_fetched = artificial_transition_list[j]
        #cell_tag = artificial_transition_list[j][0]
        kon_rates.append(trans_fetched[0][1,0])
        koff_rates.append(trans_fetched[0][0,1])
        #cell_tags.append(cell_tag[6,])
        #midline_tags.append(cell_tag[4,])
    else:
        rate_problem_list.append(j) # HACK

#%%
on_rate_df = pd.DataFrame(on_prob_list)
off_rate_df = pd.DataFrame(off_prob_list)
keys_df = pd.DataFrame(signal_keys)
emission_df = pd.DataFrame(emission_old_technique * (20 / 19.60784))
occupancy_df = pd.DataFrame(posterior_transition_occupancies)

old_style_kon_df = pd.DataFrame(kon_rates)
old_style_koff_df = pd.DataFrame(koff_rates)

#export_df = pd.concat([keys_df, on_rate_df, off_rate_df, emission_df, occupancy_df, old_style_kon_df, old_style_koff_df], axis=1)


keys_df.drop(keys_df.index[rate_problem_list],inplace=True) # HACK
emission_df.drop(emission_df.index[rate_problem_list],inplace=True)
occupancy_df.drop(occupancy_df.index[rate_problem_list],inplace=True)


export_df = pd.concat([keys_df, emission_df, occupancy_df, old_style_kon_df, old_style_koff_df], axis=1)
export_df_titles=['Key', 'Cluster', 'Mean Value', 'Midline distance', 'AP position', 'Lateral position', 'Z position', 'Emission', 'Occupancy', 'kon', 'koff']
export_df.columns = export_df_titles

#5 = Lateral position (across midline)
#4 = AP position
#6 = z position
#3 = distance from midline

#export_df.drop(export_df.index[4],inplace=True)

export_df.to_csv('uwt_e1_mutant_comparison_xi_outer_TRC.csv')
pd.DataFrame(reconstituted_posterior).to_csv('uwt_e1_mutant_comparison_xi_outer_posterior_TRC.csv')
pd.DataFrame(reconstituted_signals).to_csv('uwt_e1_mutant_comparison_xi_outer_signals_TRC.csv')

