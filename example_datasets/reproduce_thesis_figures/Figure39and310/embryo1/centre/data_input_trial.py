# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:11:17 2019

@author: Jon
"""
#%%
import pandas as pd
from glob import glob
import scipy
import numpy as np

# Import and process data
df_list = []
deltaT = 20

for file in glob('result_*.csv'):
    data_row = pd.read_csv(file, header=None)
    df_list.append(data_row)
    
results_dataframe = pd.concat(df_list)
results_dataframe.columns = ['Random Seed', 'p_off_off', 'p_off_on', 'p_on_off', 'p_on_on',
                    'pi0_on', 'pi0_off', 'mu_off', 'mu_on', 'noise', 'logL']
results_dataframe.reset_index(inplace=True, drop=True)

max_likelihood = results_dataframe['logL'].idxmax()
max_likelihood_estimate = results_dataframe.iloc[max_likelihood,:]

trans_matrix = np.zeros((2,2))
trans_matrix[0,0] = max_likelihood_estimate['p_off_off']
trans_matrix[1,0] = max_likelihood_estimate['p_off_on']
trans_matrix[0,1] = max_likelihood_estimate['p_on_off']
trans_matrix[1,1] = max_likelihood_estimate['p_on_on'] 
rates = (scipy.linalg.logm(trans_matrix) / deltaT) * 60


#%%

#Get results
emission = max_likelihood_estimate['mu_on'] * 3

kon = trans_matrix[1,0]
koff = trans_matrix[0,1]

burst_size = emission / koff

burst_frequency = (kon * koff) / (kon + koff)

occupancy = kon / (kon + koff)