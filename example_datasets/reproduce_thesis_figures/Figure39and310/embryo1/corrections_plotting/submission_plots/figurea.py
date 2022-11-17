import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams["font.family"] = "Times New Roman"

# Matlab result
true_kon = 0.098
true_koff = 0.5110
true_emission = 32016
true_noise = 14897

kon_256 = 0.0978372111986
koff_256 = 0.510171958011
emission_256 = 32009.6322095
noise_256 = 14894.2632181

kon_128 = 0.0975202438045
koff_128 = 0.508535684955
emission_128 = 32017.3212252
noise_128 = 14887.6203776

kon_64 = 0.0967570446761
koff_64 = 0.504337544901
emission_64 = 32004.0308249
noise_64 = 14890.4530856

kon_32 = 0.0958484780609
koff_32 = 0.499890281757
emission_32 = 32018.5764011
noise_32 = 14952.3220362

kon_16 = 0.0936073881773
koff_16 = 0.487107511617
emission_16 = 31999.4745345
noise_16 = 15137.7409946

kon_8 = 0.090097999385
koff_8 = 0.467703633481
emission_8 = 31863.9013399
noise_8 = 15624.1269118

index = np.array([8, 16, 32, 64, 128, 256])

emissions = np.array([emission_8, emission_16, emission_32, emission_64, emission_128,
                      emission_256])

noise = np.array([noise_8, noise_16, noise_32, noise_64, noise_128, noise_256])

prob_on = np.array([kon_8, kon_16, kon_32, kon_64, kon_128, kon_256])

prob_off = np.array([koff_8, koff_16, koff_32, koff_64, koff_128, koff_256])


def rel_error(true_value, inferred_value):
    rel_calc = abs((inferred_value - true_value) / true_value)
    rel_calc2 = rel_calc * 100
    return rel_calc2


noise_error_holder = np.zeros((len(index),))

for i in np.arange(0, len(index)):
    noise_error_holder[i,] = rel_error(true_noise, noise[i])

prob_on_error_holder = np.zeros((len(index),))

for j in np.arange(0, len(index)):
    prob_on_error_holder[j,] = rel_error(true_kon, prob_on[j])

prob_off_error_holder = np.zeros((len(index),))

for k in np.arange(0, len(index)):
    prob_off_error_holder[k,] = rel_error(true_koff, prob_off[k])

emissions_error_holder = np.zeros((len(index),))

for l in np.arange(0, len(index)):
    emissions_error_holder[l,] = rel_error(true_emission, emissions[l])

# %%
#params = {'legend.fontsize': 20}
#plt.rcParams.update(params)
#plt.rcParams.update({'font.size': 20})
params = {'legend.fontsize': 20}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'arial'

noise_dict = {'8': rel_error(true_noise, noise_8), '16': rel_error(true_noise, noise_16),
              '32': rel_error(true_noise, noise_32), '64': rel_error(true_noise, noise_64),
              '128': rel_error(true_noise, noise_128), '256': rel_error(true_noise, noise_256)}
noise_names = list(noise_dict.keys())
noise_values = list(noise_dict.values())

prob_off_dict = {'8': rel_error(true_koff, koff_8), '16': rel_error(true_koff, koff_16),
                 '32': rel_error(true_koff, koff_32),
                 '64': rel_error(true_koff, koff_64), '128': rel_error(true_koff, koff_128),
                 '256': rel_error(true_koff, koff_256)}
prob_off_names = list(prob_off_dict.keys())
prob_off_values = list(prob_off_dict.values())

prob_on_dict = {'8': rel_error(true_kon, kon_8), '16': rel_error(true_kon, kon_16), '32': rel_error(true_kon, kon_32),
                '64': rel_error(true_kon, kon_64), '128': rel_error(true_kon, kon_128),
                '256': rel_error(true_kon, kon_256)}
prob_on_names = list(prob_on_dict.keys())
prob_on_values = list(prob_on_dict.values())

emissions_dict = {'8': rel_error(true_emission, emission_8), '16': rel_error(true_emission, emission_16),
                  '32': rel_error(true_emission, emission_32), '64': rel_error(true_emission, emission_64),
                  '128': rel_error(true_emission, emission_128), '256': rel_error(true_emission, emission_256)}
emissions_names = list(emissions_dict.keys())
emissions_values = list(emissions_dict.values())

plt.figure(1, figsize=(8, 6))
plt.plot(noise_names, noise_values, label='Noise', marker='s', linewidth=3)
plt.plot(prob_off_names, prob_off_values, label='$k_{off}$', marker='.', linewidth=3)
plt.plot(prob_on_names, prob_on_values, label='$k_{on}$', marker='o', linewidth=3)
plt.plot(emissions_names, emissions_values, label='Emission', marker='v', linewidth=3)
plt.xlabel('Number of Allowed States (M)')
#plt.ylabel('Relative Error (%)')
plt.ylim((0, 15))
# plt.xlim((0, 300))
# plt.title('Plot of Convergence of Truncated Model Parameters to Full Model Parameters')
# plt.title('Plot of Convergence of Truncated and Full Model Parameters')
plt.legend()
plt.savefig('figurea.pdf', dpi=300, transparent=True)
plt.show()