# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:31:28 2020

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

# =============================================================================
# centre_mean = np.mean(centre[:,8:], axis=0)
# inter_mean = np.mean(inter[:,12:], axis=0)
# outer_mean = np.mean(outer[:,18:], axis=0)
# 
# plt.figure(0)
# plt.plot(centre_mean)
# plt.title('mean fluorescence, uwt e1 centre (distance)')
# 
# plt.figure(1)
# plt.plot(inter_mean)
# plt.title('mean fluorescence, uwt e1 inter (distance)')
# 
# plt.figure(2)
# plt.plot(outer_mean)
# plt.title('mean fluorescence, uwt e1 outer (distance)')
# =============================================================================

non_excluded = centre

# =============================================================================
# plt.figure(6)
# plt.scatter(np.sqrt(np.abs(centre[:,3])), centre[:,2],c='r')
# plt.scatter(np.sqrt(np.abs(inter[:,3])), inter[:,2],c='b')
# plt.scatter(np.sqrt(np.abs(outer[:,3])), outer[:,2],c='g')
# plt.title('ushWT EMBRYO 1')
# plt.xlim((0,70))
# =============================================================================



#processed_signals = process_data(non_excluded)
processed_signals = process_raw_data(non_excluded,8)
###############################

signal_struct = processed_signals['Processed Signals']
unique_lengths = processed_signals['Signal Lengths']


mean_list = []

for j in signal_struct:
    mean_list.append(np.mean(j))

plt.figure(0)
plt.scatter(non_excluded[:,5], non_excluded[:,2])

plt.figure(1)
plt.scatter(non_excluded[:,5], mean_list)

plt.figure(2)
plt.scatter(sorted_by_midline[:,5], sorted_by_midline[:,2])