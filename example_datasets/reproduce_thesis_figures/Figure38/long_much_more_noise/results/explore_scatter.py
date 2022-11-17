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

# YOU MUST CHANGE THIS EVERY TIME!!!!!!!!!!!!!!!!!!!!
true_mu = 13889.8
true_noise = 32828
true_p_off_on = 0.0754
true_p_on_off = 0.1915

data_array = np.array(results_dataframe)
lls_array = data_array[:,10]
mu_array = data_array[:,8]
noise_array = data_array[:,9]
off_on_array = data_array[:,2]
on_off_array = data_array[:,3]

# import matplotlib.pyplot as plt
# import numpy as np
# mu_MSE_list = []
# for i in np.arange(0, len(data_array)):
#     MSE = np.square(np.subtract(true_mu, mu_array[i,])).mean()
#     mu_MSE_list.append(MSE)
#
# plt.figure(0)
# plt.scatter(mu_MSE_list, lls_array)
# plt.show()

#mse2 = ((true_mu - mu_array)**2).mean(axis=0)

# import matplotlib.pyplot as plt
# import numpy as np
# pon_MSE_list = []
# for i in np.arange(0, len(data_array)):
#     MSE = np.square(np.subtract(true_p_off_on, off_on_array[i,])).mean()
#     pon_MSE_list.append(MSE)
#
# plt.figure(1)
# plt.scatter(pon_MSE_list, lls_array)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
mu_rel_error_list = []
for i in np.arange(0, len(data_array)):
    #MSE = np.square(np.subtract(true_mu, mu_array[i,])).mean()
    rel_error = 100 * np.abs((mu_array[i,] - true_mu) / true_mu)
    mu_rel_error_list.append(rel_error)

import matplotlib.pyplot as plt
#plt.close('all')
import numpy as np
on_off_rel_error_list = []
for i in np.arange(0, len(data_array)):
    #MSE = np.square(np.subtract(true_mu, mu_array[i,])).mean()
    rel_error = 100 * np.abs((on_off_array[i,] - true_p_on_off) / true_p_on_off)
    on_off_rel_error_list.append(rel_error)

off_on_rel_error_list = []
for i in np.arange(0, len(data_array)):
    #MSE = np.square(np.subtract(true_mu, mu_array[i,])).mean()
    rel_error = 100 * np.abs((off_on_array[i,] - true_p_off_on) / true_p_off_on)
    off_on_rel_error_list.append(rel_error)

noise_error_list = []
for i in np.arange(0, len(data_array)):
    # MSE = np.square(np.subtract(true_mu, mu_array[i,])).mean()
    rel_error = 100 * np.abs((noise_array[i,] - true_noise) / true_noise)
    noise_error_list.append(rel_error)


# VERY IMPORTANT!!!!!!!!!!!!!!!
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 6)
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'arial'
#lls_array = lls_array / 1000

plt.figure(0)
plt.scatter(mu_rel_error_list, lls_array, color='black', s=50)
plt.scatter(mu_rel_error_list[46], lls_array[46,], c='red', s=50)
plt.ylabel('log likelihood')
plt.xlabel('Relative error (%)')
#plt.title('Emission')
#plt.text(5, -121, 'x$10^{3}$', size=18, color='black')
plt.tight_layout()
plt.savefig('emission_relative_error_final.pdf', dpi=300)
plt.show()

plt.figure(1)
plt.scatter(on_off_rel_error_list, lls_array, color='black', s=50)
plt.scatter(on_off_rel_error_list[46], lls_array[46,], c='red', s=50)
plt.ylabel('log likelihood')
plt.xlabel('Relative error (%)')
#plt.title('p_on_off')
plt.tight_layout()
plt.savefig('on_off_relative_error_final.pdf', dpi=300)
plt.show()

plt.figure(2)
plt.scatter(off_on_rel_error_list, lls_array, color='black', s=50)
plt.scatter(off_on_rel_error_list[46], lls_array[46,], c='red', s=50)
plt.ylabel('log likelihood')
plt.xlabel('Relative error (%)')
#plt.title('p_off_on')
plt.tight_layout()
plt.savefig('off_on_relative_error_final.pdf', dpi=300)
plt.show()

plt.figure(3)
plt.scatter(noise_error_list, lls_array, color='black', s=50)
plt.scatter(noise_error_list[46], lls_array[46,], c='red', s=50)
plt.ylabel('log likelihood')
plt.xlabel('Relative error (%)')
#plt.title('noise')
plt.tight_layout()
plt.savefig('noise_relative_error_final.pdf', dpi=300)
plt.show()


# plt.figure(4)
# plt.scatter(mu_rel_error_list, lls_array)
# plt.ylabel('log likelihood')
# plt.xlabel('Relative error (%)')
# plt.title('Emission Zoomed In (x axis zoom)')
# plt.ylim(-119549, -119547)
# plt.xlim(2, 5)
# plt.savefig('emission_x_axis_zoomed.pdf', dpi=300)
# plt.show()
#
# plt.figure(5)
# plt.scatter(mu_rel_error_list, lls_array)
# plt.ylabel('log likelihood')
# plt.xlabel('Relative error (%)')
# plt.title('Emission Zoomed In (no x axis zoom)')
# plt.ylim(-119549, -119547)
# #plt.xlim(2, 5)
# plt.savefig('emission_x_axis_no_zoom.pdf', dpi=300)
# plt.show()
#
#
#
# plt.figure(6)
# plt.scatter(off_on_rel_error_list, lls_array)
# plt.ylabel('log likelihood')
# plt.xlabel('Relative error (%)')
# plt.title('p_off_on')
# plt.savefig('p_off_on_relative_error.pdf', dpi=300)
# plt.show()
#
# plt.figure(7)
# plt.scatter(off_on_rel_error_list, lls_array)
# plt.ylabel('log likelihood')
# plt.xlabel('Relative error (%)')
# plt.title('p_off_on Zoomed In (no x axis zoom)')
# plt.ylim(-119549, -119547)
# #plt.xlim(2, 5)
# plt.savefig('p_off_on_no_x_axis_zoom.pdf', dpi=300)
# plt.show()
#
# plt.figure(8)
# plt.scatter(off_on_rel_error_list, lls_array)
# plt.ylabel('log likelihood')
# plt.xlabel('Relative error (%)')
# plt.title('p_off_on Zoomed In (x axis zoom)')
# plt.ylim(-119549, -119547)
# plt.xlim(25.5, 26.5)
# plt.savefig('p_off_on_x_axis_zoom.pdf', dpi=300)
# plt.show()

#%%
from scipy.linalg import logm
test_transitions = np.array([[0.0754, 0.1915],[1-0.0754, 1-0.1915]])
test_rates = (scipy.linalg.logm(test_transitions) / 20) * 60

test_transitions2 = np.array([[0.1, 0.2],[0.9, 1-0.2]])
print((scipy.linalg.logm(test_transitions2) / 20) * 60)