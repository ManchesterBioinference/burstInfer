import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
#from numba import jit
#from burstinfer.compute_dynamic_F import compute_dynamic_F
from burstInfer.get_adjusted import get_adjusted
from burstInfer.ms2_loading_coeff import ms2_loading_coeff

# Import ML parameters
max_likelihood_estimate = pd.read_csv('result_694826350.csv', header=None, index_col=0)

ms2_signals = genfromtxt('SINGLE_CELL_SIGNALS.csv', delimiter=',', skip_header=1)
ms2_signals = ms2_signals[:, 2:]

promoters = genfromtxt('SINGLE_CELL_POSTERIOR.csv', delimiter=',', skip_header=1)
promoters = promoters[:, 2:]

K = 2
W = 13

mu = np.zeros((K, 1))
mu[0, 0] = max_likelihood_estimate.iloc[1, 6]
mu[1, 0] = max_likelihood_estimate.iloc[1, 7]
#noise = max_likelihood_estimate.iloc[1, 8]
noise = float(max_likelihood_estimate.iloc[1, 8])

t_MS2 = 30
deltaT = 20
kappa = t_MS2 / deltaT

#%%
# MS2 coefficient calculation
ms2_coeff = ms2_loading_coeff(kappa, W)
ms2_coeff_flipped = np.flip(ms2_coeff, 1)
count_reduction_manual = np.zeros((1,W-1))
for t in np.arange(0,W-1):
    count_reduction_manual[0,t] = np.sum(ms2_coeff[0,t+1:])
count_reduction_manual = np.reshape(count_reduction_manual, (W-1,1))


mask = np.int32((2**W)-1)

#%%
fluorescence_holder = []

for i in np.arange(0, len(promoters)):
    single_promoter = np.expand_dims(promoters[i, :], axis=0)
    single_promoter = single_promoter[~np.isnan(single_promoter)] ###
    single_promoter = np.expand_dims(single_promoter, axis=0) ###

    single_trace = np.zeros((1, single_promoter.shape[1]))

    t = 0

    window_storage = int(single_promoter[0, 0])
    # single_trace[0,t] = ((F_on_viewer[window_storage, t] * mu[1,0]) + (F_off_viewer[window_storage, t] * mu[0,0])) + np.random.normal(0, noise)
    single_trace[0, t] = ((get_adjusted(window_storage, K, W, ms2_coeff)[0] * mu[1, 0]) + (
                get_adjusted(window_storage, K, W, ms2_coeff)[1] * mu[0, 0]))
                #np.random.normal(0, noise)

    window_storage = 0
    t = 1
    present_state_list = []
    present_state_list.append(int(single_promoter[0, 0]))
    # while t < W:
    while t < single_promoter.shape[1]:
        present_state = int(single_promoter[0, t])
        # print('present state')
        # print(present_state)
        # present_state_list.append(present_state)
        window_storage = np.bitwise_and((present_state_list[t - 1] << 1) + present_state, mask)
        # print('window storage')
        # print(window_storage)
        present_state_list.append(window_storage)

        # single_trace[0,t] = ((F_on_viewer[window_storage, t] * mu[1,0]) + (F_off_viewer[window_storage, t] * mu[0,0])) + np.random.normal(0, noise)
        single_trace[0, t] = ((get_adjusted(window_storage, K, W, ms2_coeff)[0] * mu[1, 0]) + (
                    get_adjusted(window_storage, K, W, ms2_coeff)[1] * mu[0, 0]))
                    #np.random.normal(0, noise)

        t = t + 1

    fluorescence_holder.append(single_trace)

#selector = 23
selector = 23
selected_promoter = promoters[selector,:]
selected_promoter = selected_promoter[~np.isnan(selected_promoter)]
selected_original = ms2_signals[selector, :]
selected_original = selected_original[~np.isnan(selected_original)]
selected_fitted = fluorescence_holder[selector]

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 6)
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'arial'

plt.figure(0)
plt.plot(selected_fitted.flatten(), c='r')
plt.plot(selected_original, c='b')
plt.show()

synthetic_x_fitted = np.arange(0,selected_fitted.shape[1])
synthetic_x_original = np.arange(0,selected_original.shape[0])


# BEWARE!!!!!!!!!
selected_fitted = selected_fitted / 10000
selected_original = selected_original / 10000
noise = noise / 10000


index = synthetic_x_fitted / 3
plt.figure(1)
plt.plot(index, selected_fitted.flatten(), color='red', lw=1.5)
#plt.plot(synthetic_x, selected_fitted.flatten()-2*noise, color='blue', lw=0.5)
#plt.plot(synthetic_x, selected_fitted.flatten()+2*noise, color='blue', lw=0.5)
plt.fill_between(index, selected_fitted.flatten()-2*noise, selected_fitted.flatten()+2*noise, \
                 color='red',alpha=0.1)
plt.plot(index, selected_original, c='black', alpha=0.5)
plt.xlabel('Time into nuclear cycle 14 (min)')
plt.ylabel('Fluorescence Intensity (au)')
plt.text(0.05, 19, 'x$10^{4}$', size=18, color='black')
#time_ticks = np
plt.xticks(np.arange(0,35,5))
plt.savefig('long_synthetic_fit.pdf', dpi=300)
plt.show()

original_promoters = genfromtxt('synthetic_promoter_traces_w13.csv', delimiter=',', skip_header=0)
#original_promoters = original_promoters[:, 1:]
selected_original_promoter = original_promoters[selector, 1:]

#selected_original_promoter[0, ] = 0
plt.figure(2)
plt.step(index, selected_promoter.flatten(), color='red', lw=1.5)
plt.step(index, selected_original_promoter.flatten(), c='black', alpha=0.25, lw=5)
plt.xlabel('Time into nuclear cycle 14 (min)')
plt.ylabel('Inferred Promoter State')
#time_ticks = np
plt.xticks(np.arange(0,35,5))
plt.yticks(np.arange(0, 2), labels=['OFF', 'ON'])
plt.savefig('long_synthetic_promoter.pdf', dpi=300)
plt.show()

#%%
MSE = np.square(np.subtract(original_promoters[0,1:],promoters[0,:])).mean()
#MSE = np.square(np.subtract(Y_true,Y_pred)).mean()

MSE_list = []
for i in np.arange(0, len(promoters)):
    MSE = np.square(np.subtract(original_promoters[i, 1:], promoters[i, :])).mean()
    MSE_list.append(MSE)


plt.figure(3)
n, bins, patches = plt.hist(MSE_list, 10, density=False, facecolor='b', alpha=0.75)

plt.ylabel('Number of promoter traces')
plt.xlabel('MSE')
plt.title('LONG GENE')
plt.show()