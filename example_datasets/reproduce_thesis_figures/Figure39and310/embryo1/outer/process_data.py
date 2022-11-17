# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:37:25 2019

@author: Jon
"""
import numpy as np

def process_data(signals):

    # Build list of fluorescence signals
    number_of_signals = len(signals)
    signal_struct = []
    mean_matrix = np.zeros((number_of_signals,1))
    max_matrix = np.zeros((number_of_signals,1))
    length_container = []
    for u in np.arange(0,number_of_signals):
        requested_signal = signals[u,]
        requested_signal2 = requested_signal[~np.isnan(requested_signal)]
        mean_matrix[u,] = np.mean(requested_signal2, axis = 0)
        max_matrix[u,] = np.max(requested_signal2, axis = 0)
        signal_struct.append(np.reshape(requested_signal2, (1,len(requested_signal2))))
        length_container.append(len(requested_signal2))
        
    matrix_mean = np.mean(mean_matrix)
    matrix_max = np.max(max_matrix)
    unique_lengths = np.unique(length_container)
    
    output_dict = {}
    output_dict['Processed Signals'] = signal_struct
    output_dict['Matrix Mean'] = matrix_mean
    output_dict['Matrix Max'] = matrix_max
    output_dict['Signal Lengths'] = unique_lengths
    
    return output_dict
