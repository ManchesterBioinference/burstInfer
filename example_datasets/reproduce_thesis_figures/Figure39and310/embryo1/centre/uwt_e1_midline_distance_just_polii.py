# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:55:47 2020

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


non_excluded = centre


non_excluded_df = pd.DataFrame(non_excluded)
non_excluded_df.to_csv('uwt_e1_midline_distance_training_data_centre.csv')