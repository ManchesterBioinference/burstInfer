# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 22:18:44 2020

@author: MBGM9JBC
"""
import pandas as pd
import scipy
import numpy as np
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('dark_background')
plt.rcParams["figure.figsize"] = (6,6)

centre = pd.read_csv('uwt_e1_mutant_comparison_xi_centre.csv')
inter = pd.read_csv('uwt_e1_mutant_comparison_xi_inter.csv')
outer = pd.read_csv('uwt_e1_mutant_comparison_xi_outer.csv')

collected = pd.concat([outer, inter, centre], axis = 0)
#collected = collected.iloc[33:,1:]
collected = collected.iloc[:,1:]

collected.dropna(axis=0,how='any',inplace=True)
collected.drop(collected.iloc[:,0] == 1000110000, inplace=True)

mean = collected.iloc[:,2]
position = collected.iloc[:,5]
#emission = collected.iloc[6:,7]
emission = collected.iloc[:,7]
occupancy = collected.iloc[:,8]
old_kon = collected.iloc[:,9]
old_koff = collected.iloc[:,10]



#%%
import seaborn as sns
import scipy

occupancy_df = occupancy
mean_expression_df = mean

occupancy_mean = pd.concat((occupancy_df, mean_expression_df), axis=1)
occupancy_mean.columns = ["Occupancy", "Mean Expression"]

plt.figure(6)
sns.regplot(list(mean), list(occupancy))
plt.xlabel("Mean Expression Level (Observed Fluorescence)")
plt.ylabel("Occupancy")
plt.title("Occupancy / Expression Correlation")

print(occupancy_mean.corr(method="pearson"))

kon_df = old_kon
kon_mean = pd.concat((kon_df, mean_expression_df), axis=1)
kon_mean.columns = ["kon", "Mean Expression"]

plt.figure(7)
sns.regplot(list(mean), list(old_kon))
plt.xlabel("Mean Expression Level (Observed Fluorescence)")
plt.ylabel("kon")
plt.title("k_on / Expression Correlation")

print(kon_mean.corr(method="pearson"))

koff_df = old_koff
koff_mean = pd.concat((koff_df, mean_expression_df), axis=1)
koff_mean.columns = ["koff", "Mean Expression"]

plt.figure(8)
sns.regplot(list(mean), list(old_koff))
plt.xlabel("Mean Expression Level (Observed Fluorescence)")
plt.ylabel("koff")
plt.title("k_off / Expression Correlation")

print(koff_mean.corr(method="pearson"))

# =============================================================================
# frequency_df = pd.DataFrame(burst_frequency_proper_rate)
# frequency_mean = pd.concat((frequency_df, mean_expression_df), axis=1)
# frequency_mean.columns = ["Burst Frequency", "Mean Expression"]
# 
# plt.figure(9)
# sns.regplot(expression_list, burst_frequency_proper_rate)
# plt.xlabel("Mean Expression Level (Observed Fluorescence)")
# plt.ylabel("Burst Frequency")
# plt.title("Burst Frequency / Expression Correlation")
# 
# print(frequency_mean.corr(method="pearson"))
# =============================================================================

# Do burst size
#burst_size = emission / old_koff[6:,]
burst_size = emission / old_koff
#size_mean = pd.concat((burst_size, mean_expression_df.iloc[6:,]), axis=1)
size_mean = pd.concat((burst_size, mean_expression_df), axis=1)
size_mean.columns = ["burst size", "Mean Expression"]

plt.figure(9)
#sns.regplot(list(mean.iloc[6:,]), list(burst_size))
sns.regplot(list(mean), list(burst_size))
plt.xlabel("Mean Expression Level (Observed Fluorescence)")
plt.ylabel("burst size")
plt.title("burst size / Expression Correlation")

print(size_mean.corr(method="pearson"))

#%%

# Single-cell emission correlation analysis
# TODO
#single_cell_emission_df_truncated = pd.DataFrame(inbetween2_emission[25:,])
#single_cell_emission_array_truncated = np.array(single_cell_emission_df_truncated)

#emission_mean = pd.concat((emission, mean.iloc[6:,]), axis=1)
emission_mean = pd.concat((emission, mean), axis=1)
emission_mean.columns = ["Single Cell Emission", "Mean Expression"]

plt.figure(11)
#sns.regplot(expression_list, single_cell_emission_array_truncated.flatten())
#sns.regplot(list(mean.iloc[6:,]), list(emission*3))
sns.regplot(list(mean), list(emission*3))
plt.xlabel("Mean Expression Level (Observed Fluorescence)")
plt.ylabel("Single Cell Emission")
plt.title("Single Cell Emission / Expression Correlation")
#plt.ylim((5000,35000))

print(emission_mean.corr(method="pearson"))

# Visualisation

plt.figure(12)
plt.scatter(position, mean)
plt.title('mean')

plt.figure(13)
plt.scatter(position, occupancy)
plt.title('occupancy')

plt.figure(14)
#plt.scatter(position.iloc[6:,], emission)
plt.scatter(position, emission)
plt.title('emission')

plt.figure(15)
plt.scatter(position, old_kon)
plt.title('kon')

plt.figure(16)
plt.scatter(position, old_koff)
plt.title('koff')

plt.figure(17)
plt.scatter(collected.iloc[:,4], position, c=old_koff, cmap='plasma', s=80)




