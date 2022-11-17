# -*- coding: utf-8 -*-
"""
Created on Mon May 17 23:06:57 2021

@author: MBGM9JBC
"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

centre = genfromtxt('corrections_traces_centre.csv', delimiter=',',
                           skip_header=1)
centre = centre[:,1:]

max_posterior_list = list(centre)
viewer = max_posterior_list

off_on = []
on_off = []

off_off = []
on_on = []

length_holder = []

for j in np.arange(0, len(max_posterior_list)):
    fetched_posterior = max_posterior_list[j]
    fetched_posterior2 = fetched_posterior[~np.isnan(fetched_posterior)]
    off_on_interior = 0
    on_off_interior = 0
    length_holder.append(len(fetched_posterior2)-1)
    
    off_off_interior = 0
    on_on_interior = 0
    for k in np.arange(1, len(fetched_posterior2)):
        digit = fetched_posterior2[k,]
        digit_m1 = fetched_posterior2[k-1]
    
        if digit == 1 and digit_m1 == 0:
            off_on_interior +=1
        elif digit == 0 and digit_m1 == 1:
            on_off_interior +=1
        elif digit == 0 and digit_m1 == 0:
            off_off_interior +=1
        elif digit == 1 and digit_m1 == 1:
            on_on_interior +=1
    
      
    off_on.append(off_on_interior)
    on_off.append(on_off_interior)
    off_off.append(off_off_interior)
    on_on.append(on_on_interior)
    
centre_on_on_array = np.array(on_on)
centre_on_off_array = np.array(on_off)
centre_off_on_array = np.array(off_on)
centre_off_off_array = np.array(off_off)

on_on_sum = np.sum(centre_on_on_array)
on_off_sum = np.sum(centre_on_off_array)
off_on_sum = np.sum(centre_off_on_array)
off_off_sum = np.sum(centre_off_off_array)

n_trials = sum(length_holder)

A = 3
on_on_frac = (on_on_sum / n_trials) * A
on_off_frac = (on_off_sum / n_trials) * A
off_on_frac = (off_on_sum / n_trials) * A
off_off_frac = (off_off_sum / n_trials) * A

centre_p_on = np.divide(centre_off_on_array + off_on_frac, (centre_off_off_array + centre_off_on_array + off_off_frac + off_on_frac))
centre_p_off = np.divide(centre_on_off_array + on_off_frac, (centre_on_on_array + centre_on_off_array + on_off_frac + on_on_frac))

centre_occ= centre_p_on / (centre_p_on + centre_p_off)

#%%
inter = genfromtxt('corrections_traces_inter.csv', delimiter=',',
                           skip_header=1)
inter = inter[:,1:]

max_posterior_list = list(inter)

off_on = []
on_off = []

off_off = []
on_on = []

length_holder = []

for j in np.arange(0, len(max_posterior_list)):
    fetched_posterior = max_posterior_list[j]
    fetched_posterior2 = fetched_posterior[~np.isnan(fetched_posterior)]
    off_on_interior = 0
    on_off_interior = 0
    length_holder.append(len(fetched_posterior2)-1)
    
    off_off_interior = 0
    on_on_interior = 0
    for k in np.arange(1, len(fetched_posterior2)):
        digit = fetched_posterior2[k,]
        digit_m1 = fetched_posterior2[k-1]
    
        if digit == 1 and digit_m1 == 0:
            off_on_interior +=1
        elif digit == 0 and digit_m1 == 1:
            on_off_interior +=1
        elif digit == 0 and digit_m1 == 0:
            off_off_interior +=1
        elif digit == 1 and digit_m1 == 1:
            on_on_interior +=1
    
      
    off_on.append(off_on_interior)
    on_off.append(on_off_interior)
    off_off.append(off_off_interior)
    on_on.append(on_on_interior)
    

inter_on_on_array = np.array(on_on)
inter_on_off_array = np.array(on_off)
inter_off_on_array = np.array(off_on)
inter_off_off_array = np.array(off_off)

on_on_sum = np.sum(inter_on_on_array)
on_off_sum = np.sum(inter_on_off_array)
off_on_sum = np.sum(inter_off_on_array)
off_off_sum = np.sum(inter_off_off_array)

n_trials = sum(length_holder)

A = 3
on_on_frac = (on_on_sum / n_trials) * A
on_off_frac = (on_off_sum / n_trials) * A
off_on_frac = (off_on_sum / n_trials) * A
off_off_frac = (off_off_sum / n_trials) * A

inter_p_on = np.divide(inter_off_on_array + off_on_frac, (inter_off_off_array + inter_off_on_array + off_off_frac + off_on_frac))
inter_p_off = np.divide(inter_on_off_array + on_off_frac, (inter_on_on_array + inter_on_off_array + on_off_frac + on_on_frac))

inter_occ= inter_p_on / (inter_p_on + inter_p_off)

#%%
outer = genfromtxt('corrections_traces_outer.csv', delimiter=',',
                           skip_header=1)
outer = outer[:,1:]

max_posterior_list = list(outer)

off_on = []
on_off = []

off_off = []
on_on = []

length_holder = []

for j in np.arange(0, len(max_posterior_list)):
    fetched_posterior = max_posterior_list[j]
    fetched_posterior2 = fetched_posterior[~np.isnan(fetched_posterior)]
    off_on_interior = 0
    on_off_interior = 0
    length_holder.append(len(fetched_posterior2)-1)
    
    off_off_interior = 0
    on_on_interior = 0
    for k in np.arange(1, len(fetched_posterior2)):
        digit = fetched_posterior2[k,]
        digit_m1 = fetched_posterior2[k-1]
    
        if digit == 1 and digit_m1 == 0:
            off_on_interior +=1
        elif digit == 0 and digit_m1 == 1:
            on_off_interior +=1
        elif digit == 0 and digit_m1 == 0:
            off_off_interior +=1
        elif digit == 1 and digit_m1 == 1:
            on_on_interior +=1
    
      
    off_on.append(off_on_interior)
    on_off.append(on_off_interior)
    off_off.append(off_off_interior)
    on_on.append(on_on_interior)
    

outer_on_on_array = np.array(on_on)
outer_on_off_array = np.array(on_off)
outer_off_on_array = np.array(off_on)
outer_off_off_array = np.array(off_off)

on_on_sum = np.sum(outer_on_on_array)
on_off_sum = np.sum(outer_on_off_array)
off_on_sum = np.sum(outer_off_on_array)
off_off_sum = np.sum(outer_off_off_array)

n_trials = sum(length_holder)

A = 3
on_on_frac = (on_on_sum / n_trials) * A
on_off_frac = (on_off_sum / n_trials) * A
off_on_frac = (off_on_sum / n_trials) * A
off_off_frac = (off_off_sum / n_trials) * A

outer_p_on = np.divide(outer_off_on_array + off_on_frac, (outer_off_off_array + outer_off_on_array + off_off_frac + off_on_frac))
outer_p_off = np.divide(outer_on_off_array + on_off_frac, (outer_on_on_array + outer_on_off_array + on_off_frac + on_on_frac))

outer_occ= outer_p_on / (outer_p_on + outer_p_off)

#%%
on_counts = np.concatenate((centre_off_on_array, inter_off_on_array, outer_off_on_array), axis=0)
off_counts = np.concatenate((centre_on_off_array, inter_on_off_array, outer_on_off_array), axis=0)
occupancy = np.concatenate((centre_occ, inter_occ, outer_occ), axis=0)

imported_centre_traces = genfromtxt('centre_traces.csv', delimiter=',',
                           skip_header=1)
imported_centre_traces = imported_centre_traces[:,1:]
imported_inter_traces = genfromtxt('inter_traces.csv', delimiter=',',
                           skip_header=1)
imported_inter_traces = imported_inter_traces[:,1:]
imported_outer_traces = genfromtxt('outer_traces.csv', delimiter=',',
                           skip_header=1)
imported_outer_traces = imported_outer_traces[:,1:]

centre_x = imported_centre_traces[:,4]
centre_y = imported_centre_traces[:,5]
inter_x = imported_inter_traces[:,4]
inter_y = imported_inter_traces[:,5]
outer_x = imported_outer_traces[:,4]
outer_y = imported_outer_traces[:,5]

centre_expression = imported_centre_traces[:,2]
inter_expression = imported_inter_traces[:,2]
outer_expression = imported_outer_traces[:,2]

combined_x = np.concatenate((centre_x, inter_x, outer_x), axis=0)
combined_y = np.concatenate((centre_y, inter_y, outer_y), axis=0)
combined_expression = np.concatenate((centre_expression, inter_expression, outer_expression), axis=0)

plt.figure(0)
plt.scatter(combined_x, combined_y, c=on_counts, cmap='magma')
plt.show()

plt.figure(1)
plt.scatter(combined_x, combined_y, c=off_counts, cmap='magma')
plt.show()

plt.figure(2)
plt.scatter(combined_x, combined_y, c=occupancy, cmap='magma')
plt.show()

plt.figure(3)
plt.scatter(combined_x, combined_y, c=combined_expression, cmap='magma')
plt.show()


plt.figure(4)
plt.scatter(combined_y, on_counts)
plt.title('number of off - on transitions')
plt.show()

plt.figure(5)
plt.scatter(combined_y, off_counts)
plt.title('number of on - off transitions')
plt.show()

plt.figure(6)
plt.scatter(combined_y, occupancy)
plt.title('occupancy')
plt.show()

plt.figure(7)
plt.scatter(combined_y, combined_expression)
plt.title('mean expression')
plt.show()

combined_p_on = np.concatenate((centre_p_on, inter_p_on, outer_p_on), axis=0)
combined_p_off = np.concatenate((centre_p_off, inter_p_off, outer_p_off), axis=0)

# =============================================================================
# plt.figure(4)
# plt.scatter(combined_y, combined_p_on)
# plt.title('probability of off - on transition')
# 
# plt.figure(5)
# plt.scatter(combined_y, combined_p_off)
# plt.title('probability of on - off transition')
# 
# plt.figure(6)
# plt.scatter(combined_y, occupancy)
# plt.title('occupancy')
# 
# plt.figure(7)
# plt.scatter(combined_y, combined_expression)
# plt.title('mean expression')
# =============================================================================

import statsmodels as sm
import pandas as pd
from statsmodels.stats import proportion

on_counts_df = pd.DataFrame(on_counts)
#on_counts_df = on_counts_df.replace(0, 1)
off_off_counts = np.concatenate((centre_off_off_array, inter_off_off_array, outer_off_off_array), axis=0)
off_off_counts_df = pd.DataFrame(off_off_counts)

off_on_transitions_counts = np.sum(on_counts)
off_off_transitions_counts = np.sum(off_off_counts)
n = off_off_transitions_counts + off_on_transitions_counts

# not n_trials!
bin_confs_on = sm.stats.proportion.proportion_confint(on_counts, n,
                                                   method='jeffreys', alpha=0.05)
bin_confs_on_lower = bin_confs_on[0]
bin_confs_on_upper = bin_confs_on[1]

plt.figure(8)
plt.scatter(combined_y, on_counts)
plt.scatter(combined_y, bin_confs_on_lower)
plt.scatter(combined_y, bin_confs_on_upper)
plt.show()

trial = sm.stats.proportion.proportion_confint(5, (off_off_counts[0,]
                                               + on_counts[0,]),
                                                   method='jeffreys', alpha=0.05)

loop_holder_low = []
loop_holder_high = []
for i in np.arange(0, len(on_counts)):
    inner_result = sm.stats.proportion.proportion_confint(on_counts[i,],
                                                          (off_off_counts[i,]
                                                            + on_counts[i,]),
                                                   method='jeffreys', alpha=0.05)
    loop_holder_low.append(inner_result[0])
    loop_holder_high.append(inner_result[1])

on_proportions = on_counts / (off_off_counts + on_counts)
plt.figure(9)
#plt.scatter(combined_y, on_proportions)
#plt.scatter(combined_y, loop_holder_low)
#plt.scatter(combined_y, loop_holder_high)
plt.errorbar(combined_y, on_proportions,
             yerr=[loop_holder_low, loop_holder_high], fmt='o')
plt.show()



##%
on_on_counts = np.concatenate((centre_on_on_array, inter_on_on_array, outer_on_on_array), axis=0)

loop_holder_low_off = []
loop_holder_high_off = []
for i in np.arange(0, len(on_counts)):
    inner_result = sm.stats.proportion.proportion_confint(off_counts[i,],
                                                          (on_on_counts[i,]
                                                            + off_counts[i,]),
                                                   method='jeffreys', alpha=0.05)
    loop_holder_low_off.append(inner_result[0])
    loop_holder_high_off.append(inner_result[1])

off_proportions = off_counts / (on_on_counts + off_counts)
plt.figure(10)
#plt.scatter(combined_y, on_proportions)
#plt.scatter(combined_y, loop_holder_low)
#plt.scatter(combined_y, loop_holder_high)
plt.errorbar(combined_y, off_proportions,
             yerr=[loop_holder_low_off, loop_holder_high_off], fmt='o')
plt.show()

"""
combined_y_cut = combined_y[0:40,]
on_proportions_cut = on_proportions[0:40]
loop_holder_low_cut = loop_holder_low[0:40]
loop_holder_high_cut = loop_holder_high[0:40]
print('debugger')
plt.figure(11)
plt.errorbar(combined_y_cut, on_proportions_cut,
             yerr=[loop_holder_low_cut, loop_holder_high_cut], fmt='o')
print('debugger2')
plt.show()
"""

combined_on = np.stack([combined_y, on_proportions,
                        np.array(loop_holder_low), np.array(loop_holder_high)], axis=1)
combined_on2 = combined_on.copy()
combined_on2 = combined_on2[combined_on2[:,0].argsort()]
#plt.figure(12)
#plt.scatter(combined_on2[:,0], combined_on2[:,1])
#plt.title('sorted')
cutoff = 80
plt.errorbar(combined_on2[0:cutoff,0], combined_on2[0:cutoff,1],
             yerr=[combined_on2[0:cutoff,2], combined_on2[0:cutoff,3]], fmt='o')
plt.title('on selection')
plt.show()
