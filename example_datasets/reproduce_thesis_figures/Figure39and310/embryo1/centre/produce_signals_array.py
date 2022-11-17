# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:19:53 2019

@author: Jon
"""

def produce_signals_array(ms2_data):
    ms2_data = ms2_data[:,1:]
    sorted_by_means = ms2_data[ms2_data[:,2].argsort()]
    
    non_excluded = sorted_by_means[20:,:]
    
    cluster_subset = non_excluded[non_excluded[:,1] == 0]
    
    return cluster_subset