# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:54:28 2019

@author: Jon
"""

def produce_signals_dict(ms2_data, dummy_clusters, sort, threshold, remove_top_row,
                          remove_first_column, cluster_number):
    
    if remove_first_column == True:
        ms2_data = ms2_data[:,1:]
    else:
        ms2_data = ms2_data
        
    if remove_top_row == True:
        ms2_data = ms2_data[1:,:]
    else:
        ms2_data = ms2_data
        
    if dummy_clusters == True:
        ms2_data[:,1] = 1
        
    if sort == True:
        sorted_by_means = ms2_data[ms2_data[:,2].argsort()]
        ms2_data = sorted_by_means[threshold:,:]
        
    if dummy_clusters == True:
        ms2_data = ms2_data[ms2_data[:,1] == 1]
    else:
        ms2_data = ms2_data[ms2_data[:,1] == cluster_number]
    
    signals_only = ms2_data[:,7:]
    signals_and_metadata = ms2_data
    
    return_dict = {}
    return_dict['signals_and_metadata'] = signals_and_metadata
    return_dict['signals_only'] = signals_only
    
    return return_dict