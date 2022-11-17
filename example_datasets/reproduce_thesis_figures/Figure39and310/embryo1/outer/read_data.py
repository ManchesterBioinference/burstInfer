# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:29:44 2019

@author: Jon
"""

def read_data(ms2_data):
    ms2_data = ms2_data[:,1:]
    sorted_by_means = ms2_data[ms2_data[:,2].argsort()]
    
    non_excluded = sorted_by_means[20:,:]
    
    cluster_subset = non_excluded[non_excluded[:,1] == 0]
    signals_only = cluster_subset[:,7:]
    
    return signals_only