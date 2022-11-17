import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

centre = genfromtxt('corrections_traces_centre.csv', delimiter=',',
                    skip_header=1)
centre = centre[:, 1:]

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
    length_holder.append(len(fetched_posterior2) - 1)

    off_off_interior = 0
    on_on_interior = 0
    for k in np.arange(1, len(fetched_posterior2)):
        digit = fetched_posterior2[k,]
        digit_m1 = fetched_posterior2[k - 1]

        if digit == 1 and digit_m1 == 0:
            off_on_interior += 1
        elif digit == 0 and digit_m1 == 1:
            on_off_interior += 1
        elif digit == 0 and digit_m1 == 0:
            off_off_interior += 1
        elif digit == 1 and digit_m1 == 1:
            on_on_interior += 1

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

centre_p_on = np.divide(centre_off_on_array + off_on_frac,
                        (centre_off_off_array + centre_off_on_array + off_off_frac + off_on_frac))
centre_p_off = np.divide(centre_on_off_array + on_off_frac,
                         (centre_on_on_array + centre_on_off_array + on_off_frac + on_on_frac))

centre_p_on_on = np.divide(centre_on_on_array + on_on_frac,
                        (centre_on_off_array + centre_on_on_array + on_off_frac + on_on_frac))
centre_p_off_off = np.divide(centre_off_off_array + off_off_frac,
                        (centre_off_off_array + centre_off_on_array + off_off_frac + off_on_frac))

centre_occ = centre_p_on / (centre_p_on + centre_p_off)

# %%
inter = genfromtxt('corrections_traces_inter.csv', delimiter=',',
                   skip_header=1)
inter = inter[:, 1:]

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
    length_holder.append(len(fetched_posterior2) - 1)

    off_off_interior = 0
    on_on_interior = 0
    for k in np.arange(1, len(fetched_posterior2)):
        digit = fetched_posterior2[k,]
        digit_m1 = fetched_posterior2[k - 1]

        if digit == 1 and digit_m1 == 0:
            off_on_interior += 1
        elif digit == 0 and digit_m1 == 1:
            on_off_interior += 1
        elif digit == 0 and digit_m1 == 0:
            off_off_interior += 1
        elif digit == 1 and digit_m1 == 1:
            on_on_interior += 1

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

inter_p_on = np.divide(inter_off_on_array + off_on_frac,
                       (inter_off_off_array + inter_off_on_array + off_off_frac + off_on_frac))
inter_p_off = np.divide(inter_on_off_array + on_off_frac,
                        (inter_on_on_array + inter_on_off_array + on_off_frac + on_on_frac))

inter_p_on_on = np.divide(inter_on_on_array + on_on_frac,
                        (inter_on_off_array + inter_on_on_array + on_off_frac + on_on_frac))
inter_p_off_off = np.divide(inter_off_off_array + off_off_frac,
                        (inter_off_off_array + inter_off_on_array + off_off_frac + off_on_frac))

inter_occ = inter_p_on / (inter_p_on + inter_p_off)

# %%
outer = genfromtxt('corrections_traces_outer.csv', delimiter=',',
                   skip_header=1)
outer = outer[:, 1:]

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
    length_holder.append(len(fetched_posterior2) - 1)

    off_off_interior = 0
    on_on_interior = 0
    for k in np.arange(1, len(fetched_posterior2)):
        digit = fetched_posterior2[k,]
        digit_m1 = fetched_posterior2[k - 1]

        if digit == 1 and digit_m1 == 0:
            off_on_interior += 1
        elif digit == 0 and digit_m1 == 1:
            on_off_interior += 1
        elif digit == 0 and digit_m1 == 0:
            off_off_interior += 1
        elif digit == 1 and digit_m1 == 1:
            on_on_interior += 1

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

outer_p_on = np.divide(outer_off_on_array + off_on_frac,
                       (outer_off_off_array + outer_off_on_array + off_off_frac + off_on_frac))
outer_p_off = np.divide(outer_on_off_array + on_off_frac,
                        (outer_on_on_array + outer_on_off_array + on_off_frac + on_on_frac))

outer_p_on_on = np.divide(outer_on_on_array + on_on_frac,
                        (outer_on_off_array + outer_on_on_array + on_off_frac + on_on_frac))
outer_p_off_off = np.divide(outer_off_off_array + off_off_frac,
                        (outer_off_off_array + outer_off_on_array + off_off_frac + off_on_frac))

outer_occ = outer_p_on / (outer_p_on + outer_p_off)

# %%
on_counts = np.concatenate((centre_off_on_array, inter_off_on_array, outer_off_on_array), axis=0)
off_counts = np.concatenate((centre_on_off_array, inter_on_off_array, outer_on_off_array), axis=0)
occupancy = np.concatenate((centre_occ, inter_occ, outer_occ), axis=0)

imported_centre_traces = genfromtxt('centre_traces.csv', delimiter=',',
                                    skip_header=1)
imported_centre_traces = imported_centre_traces[:, 1:]
imported_inter_traces = genfromtxt('inter_traces.csv', delimiter=',',
                                   skip_header=1)
imported_inter_traces = imported_inter_traces[:, 1:]
imported_outer_traces = genfromtxt('outer_traces.csv', delimiter=',',
                                   skip_header=1)
imported_outer_traces = imported_outer_traces[:, 1:]

centre_x = imported_centre_traces[:, 4]
centre_y = imported_centre_traces[:, 5]
inter_x = imported_inter_traces[:, 4]
inter_y = imported_inter_traces[:, 5]
outer_x = imported_outer_traces[:, 4]
outer_y = imported_outer_traces[:, 5]

centre_expression = imported_centre_traces[:, 2]
inter_expression = imported_inter_traces[:, 2]
outer_expression = imported_outer_traces[:, 2]

combined_x = np.concatenate((centre_x, inter_x, outer_x), axis=0)
combined_y = np.concatenate((centre_y, inter_y, outer_y), axis=0)
combined_expression = np.concatenate((centre_expression, inter_expression, outer_expression), axis=0)


#%%
import statsmodels as sm
import pandas as pd
from statsmodels.stats import proportion

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
plt.figure(0)
# plt.scatter(combined_y, on_proportions)
# plt.scatter(combined_y, loop_holder_low)
# plt.scatter(combined_y, loop_holder_high)
plt.errorbar(combined_y, off_proportions,
             yerr=[loop_holder_low_off, loop_holder_high_off], fmt='o')
plt.show()

#%%

# Try with centre_p_on, centre_p_off etc.

combined_p_on = np.concatenate((centre_p_on, inter_p_on, outer_p_on), axis=0)
combined_p_off = np.concatenate((centre_p_off, inter_p_off, outer_p_off), axis=0)

combined_p_on_on = np.concatenate((centre_p_on_on, inter_p_on_on, outer_p_on_on), axis=0)

loop_holder_low_off = []
loop_holder_high_off = []
for i in np.arange(0, len(on_counts)):
    inner_result = sm.stats.proportion.proportion_confint(off_counts[i,],
                                                          (on_on_counts[i,]
                                                           + off_counts[i,]),
                                                          method='jeffries', alpha=0.05)
    #inner_result = sm.stats.proportion.proportion_confint(combined_p_off[i,],
    #                                                      (combined_p_on_on[i,]
    #                                                       + combined_p_off[i,]),
    #                                                      method='jeffries', alpha=0.05)
    loop_holder_low_off.append(inner_result[0])
    loop_holder_high_off.append(inner_result[1])

upper_error = np.abs(loop_holder_high_off - combined_p_off)
lower_error = np.abs(loop_holder_low_off - combined_p_off)

# #off_proportions = off_counts / (on_on_counts + off_counts)
# plt.figure(1)
# plt.errorbar(combined_y, combined_p_off,
#              yerr=[lower_error, upper_error], fmt='o', c='r')
# #plt.scatter(combined_y, combined_p_off)
# #plt.scatter(combined_y, loop_holder_low_off, s=6)
# #plt.scatter(combined_y, loop_holder_high_off, s=6)
# plt.title('TEST')
# plt.show()

vi = np.array((loop_holder_low_off, loop_holder_high_off))

#off_counts = np.concatenate((centre_on_off_array, inter_on_off_array, outer_on_off_array), axis=0)

#centre_p_off = np.divide(centre_on_off_array + on_off_frac,
#                         (centre_on_on_array + centre_on_off_array + on_off_frac + on_on_frac))

#centre_on_off_array = np.array(on_off)

#combined_p_off = np.concatenate((centre_p_off, inter_p_off, outer_p_off), axis=0)

params = {'legend.fontsize': 18}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'arial'

# off error bars
plt.figure(2, figsize = (8,6), dpi=300)
plt.errorbar(combined_y, combined_p_off,
             yerr=[lower_error, upper_error], fmt='o', markerfacecolor='grey', markeredgecolor='black',
             ecolor='lightblue', markersize=6, capsize=0)
#plt.ylabel('Proportion of promoter on -> off transitions')
#plt.ylabel('PROPORTION PLACEHOLDER')
plt.xlabel('Embryo lateral position (A.U.)')
plt.savefig('off_error_bars.pdf', dpi=300, transparent=True)
plt.show()


#%%

# On... this section definitely needs checking

combined_p_off_off = np.concatenate((centre_p_off_off, inter_p_off_off, outer_p_off_off), axis=0)
off_off_counts = np.concatenate((centre_off_off_array, inter_off_off_array, outer_off_off_array), axis=0)

loop_holder_low = []
loop_holder_high = []
for i in np.arange(0, len(on_counts)):
    inner_result = sm.stats.proportion.proportion_confint(on_counts[i,],
                                                          (off_off_counts[i,]
                                                           + on_counts[i,]),
                                                          method='jeffries', alpha=0.05)
    #inner_result = sm.stats.proportion.proportion_confint(combined_p_off[i,],
    #                                                      (combined_p_on_on[i,]
    #                                                       + combined_p_off[i,]),
    #                                                      method='jeffries', alpha=0.05)
    loop_holder_low.append(inner_result[0])
    loop_holder_high.append(inner_result[1])

upper_error = np.abs(loop_holder_high - combined_p_off)
lower_error = np.abs(loop_holder_low - combined_p_off)



# on error bars
plt.figure(3, figsize = (8,6), dpi=300)
plt.errorbar(combined_y, combined_p_on,
             yerr=[loop_holder_low, loop_holder_high], fmt='o', markerfacecolor='grey', markeredgecolor='black',
             ecolor='lightblue', markersize=6, capsize=0)
#plt.ylabel('Proportion of promoter off -> on transitions')
#plt.ylabel('PROPORTION PLACEHOLDER')
plt.xlabel('Embryo lateral position (A.U.)')
plt.savefig('on_error_bars.pdf', dpi=300, transparent=True)
plt.show()


centre_other = imported_centre_traces[:, 3]
inter_other = imported_inter_traces[:, 3]
outer_other = imported_outer_traces[:, 3]

combined_other = np.concatenate((centre_other, inter_other, outer_other), axis=0)
#combined_other2 = np.abs(np.sqrt(combined_other))
combined_other2 = combined_y - np.mean(combined_y)

# on error bars
plt.figure(4, figsize = (8,6), dpi=300)
plt.errorbar(combined_other2, combined_p_on,
             yerr=[loop_holder_low, loop_holder_high], fmt='o', markerfacecolor='grey', markeredgecolor='black',
             ecolor='lightblue', markersize=6, capsize=0)
#plt.ylabel('Proportion of promoter off -> on transitions')
#plt.ylabel('PROPORTION PLACEHOLDER')
plt.xlabel('Embryo lateral position (A.U.) OTHER2')
#plt.savefig('on_error_bars.pdf', dpi=300, transparent=True)
plt.show()

