# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:51:48 2020

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

non_excluded = sorted_by_means[20:,:]
non_excluded = non_excluded[non_excluded[:,1] == 2]

nonzeros_list = []
nonzeroviewer = []
for i in np.arange(0, len(non_excluded)):
    selected_row = non_excluded[i,7:]
    nonzero = np.nonzero(selected_row)
    nonzeros_list.append(nonzero[0][0,])
    nonzeroviewer.append(nonzero)
    
decile = np.percentile(nonzeros_list, 10)


#processed_signals = process_data(non_excluded)
processed_signals = process_raw_data(non_excluded,7)
###############################

signal_struct = processed_signals['Processed Signals']
unique_lengths = processed_signals['Signal Lengths']


centre_crude = non_excluded[:,7:]
centre_crude_mean = np.mean(centre_crude[:,0:85],axis=0)


final_mean = centre_crude_mean[-1]
print(final_mean)
pol_count = 150.35
print(pol_count)
conv_factor = final_mean / pol_count
print(conv_factor)

converted = centre_crude_mean / conv_factor

plt.figure(0)
plt.plot(centre_crude_mean)
plt.title('Mean of ushWT Centre Cluster over time')
plt.ylabel('Fluorescence')
plt.xlabel('Time Point')

plt.figure(1)
plt.plot(converted)
plt.title('Mean of ushWT Centre Cluster over time (converted)')
plt.ylabel('Pol II')
plt.xlabel('Time Point')

#%%

strip1_v = 13664.4 * 3
loading_rate = strip1_v / conv_factor
print(loading_rate)


very_final_column = centre_crude[:,-3]
very_final_column_no_nans = very_final_column[~np.isnan(very_final_column)]
very_final_mean = np.mean(very_final_column_no_nans)

print(final_mean)
pol_count = 150.35
print(pol_count)
very_final_conv_factor = very_final_mean / pol_count
print(very_final_conv_factor)

very_final_converted = centre_crude_mean / conv_factor