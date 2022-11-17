# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:18:18 2020

@author: Jon
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
    data_row = pd.read_csv(file, header=0)
    df_list.append(data_row)

results_dataframe = pd.concat(df_list)
results_dataframe.columns = ['Random Seed', 'p_off_off', 'p_off_on', 'p_on_off', 'p_on_on',
                    'pi0_off', 'pi0_on', 'mu_off', 'mu_on', 'noise', 'logL']
results_dataframe.reset_index(inplace=True, drop=True)


sorted_results = results_dataframe.sort_values(by='logL')
mu_column = sorted_results.iloc[:,8]
problematic_mu = sorted_results[sorted_results.iloc[:,8] > mu_column.quantile(0.95)]

filtered_results = sorted_results[~sorted_results.isin(problematic_mu)].dropna()


max_likelihood_estimate = filtered_results.iloc[-1,:]

# Calculate and append kon / koff
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

max_likelihood_estimate_export = pd.DataFrame(max_likelihood_estimate)
max_likelihood_estimate_export.to_csv('model_maximum_likelihood_estimate.csv', header=False, index=True)
