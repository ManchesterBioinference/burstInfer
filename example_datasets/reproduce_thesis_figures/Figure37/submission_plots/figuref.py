import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
#plt.style.use('dark_background')

kon_2048 = 0.06578
koff_2048 = 0.11615
emission_2048 = 6959.4984
noise_2048 = 9993.6278

kon_1024 = 0.06435
koff_1024 = 0.11338
emission_1024 = 6952.25697
noise_1024 = 9996.83918

kon_512 = 0.06287
koff_512 = 0.11060
emission_512 = 6947.39160
noise_512 = 9997.36251

kon_256 = 0.06115
koff_256 = 0.10723
emission_256 = 6938.26403
noise_256 = 9999.57792

kon_128 = 0.05923
koff_128 = 0.10357
emission_128 = 6931.94229
noise_128 = 10005.01434

kon_64 = 0.05695
koff_64 = 0.09936
emission_64 = 6922.36126
noise_64 = 10030.64179

kon_32 = 0.05567
koff_32 = 0.09731
emission_32 = 6930.70667
noise_32 = 10042.29043

kon_16 = 0.05465
koff_16 = 0.09575
emission_16 = 6949.67078
noise_16 = 10102.00224

kon_8 = 0.05479
koff_8 = 0.09586
emission_8 = 6999.33359
noise_8 = 10343.13084

index = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
#index = np.array([8, 16, 32, 64, 128, 256])

emissions = np.array([emission_8, emission_16, emission_32, emission_64, emission_128,
                      emission_256, emission_512, emission_1024, emission_2048])

noise = np.array([noise_8, noise_16, noise_32, noise_64, noise_128, noise_256,
                  noise_512, noise_1024, noise_2048])

prob_on = np.array([kon_8, kon_16, kon_32, kon_64, kon_128, kon_256,
                    kon_512, kon_1024, kon_2048])

prob_off = np.array([koff_8, koff_16, koff_32, koff_64, koff_128, koff_256,
                     koff_512, koff_1024, koff_2048])

# %%
params = {'legend.fontsize': 12}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 12})

"""
plt.figure(0)
emissions_normalised = emissions / emissions[7,]
noise_normalised = noise / noise[7,]
prob_on_normalised = prob_on / prob_on[7,]
prob_off_normalised = prob_off / prob_off[7,]
plt.plot(index, emissions_normalised, c='r')
plt.plot(index, noise_normalised, c='b')
plt.plot(index, prob_on_normalised, c='w')
plt.plot(index, prob_off_normalised, c='y')
plt.show()
plt.savefig('normalised.pdf', dpi=300, transparent=True)
plt.savefig('normalised.png', dpi=300)
"""

emissions_final = emissions[-1]
noise_final = noise[-1]
prob_on_final = prob_on[-1]
prob_off_final = prob_off[-1]

emissions_divided = emissions / emissions_final
noise_divided = noise / noise_final
prob_on_divided = prob_on / prob_on_final
prob_off_divided = prob_off / prob_off_final

#true_noise = noise_final
#true_emission = emissions_final
#true_kon = prob_on_final
#true_koff = prob_off_final

true_emission = 7000
true_noise = 10000
true_kon = 0.07
true_koff = 0.12

#%% Make a fancy plot
params = {'legend.fontsize': 18}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'arial'

def rel_error(true_value, inferred_value):
    rel_calc = abs((inferred_value - true_value) / true_value)
    rel_calc2 = rel_calc * 100
    return rel_calc2

noise_dict = {'8': rel_error(true_noise, noise_8), '16': rel_error(true_noise, noise_16),
              '32': rel_error(true_noise, noise_32), '64': rel_error(true_noise, noise_64),
              '128': rel_error(true_noise, noise_128), '256': rel_error(true_noise, noise_256),
              '512': rel_error(true_noise, noise_512), '1024': rel_error(true_noise, noise_1024),
              '2048': rel_error(true_noise, noise_2048)}
noise_names = list(noise_dict.keys())
noise_values = list(noise_dict.values())

prob_off_dict = {'8': rel_error(true_koff, koff_8), '16': rel_error(true_koff, koff_16),
                 '32': rel_error(true_koff, koff_32),
                 '64': rel_error(true_koff, koff_64), '128': rel_error(true_koff, koff_128),
                 '256': rel_error(true_koff, koff_256), '512': rel_error(true_koff, koff_512),
                 '1024': rel_error(true_koff, koff_1024), '2048': rel_error(true_koff, koff_2048)}
prob_off_names = list(prob_off_dict.keys())
prob_off_values = list(prob_off_dict.values())

prob_on_dict = {'8': rel_error(true_kon, kon_8), '16': rel_error(true_kon, kon_16), '32': rel_error(true_kon, kon_32),
                '64': rel_error(true_kon, kon_64), '128': rel_error(true_kon, kon_128),
                '256': rel_error(true_kon, kon_256), '512': rel_error(true_kon, kon_512),
                '1024': rel_error(true_kon, kon_1024), '2048': rel_error(true_kon, kon_2048)}
prob_on_names = list(prob_on_dict.keys())
prob_on_values = list(prob_on_dict.values())

emissions_dict = {'8': rel_error(true_emission, emission_8), '16': rel_error(true_emission, emission_16),
                  '32': rel_error(true_emission, emission_32), '64': rel_error(true_emission, emission_64),
                  '128': rel_error(true_emission, emission_128), '256': rel_error(true_emission, emission_256),
                  '512': rel_error(true_emission, emission_512), '1024': rel_error(true_emission, emission_1024),
                  '2048': rel_error(true_emission, emission_2048)}
emissions_names = list(emissions_dict.keys())
emissions_values = list(emissions_dict.values())

plt.figure(1, figsize = (8,6), dpi=300)
plt.plot(noise_names, noise_values, label = 'Noise', marker = 's', linewidth=3)
plt.plot(prob_off_names, prob_off_values, label = '$k_{off}$', marker = '.', linewidth=3)
plt.plot(prob_on_names, prob_on_values, label = '$k_{on}$', marker = 'o', linewidth=3)
plt.plot(emissions_names, emissions_values, label = 'Emission', marker = 'v', linewidth=3)
plt.xlabel('Number of Allowed States (M)')
#plt.ylabel('QUERY THIS')
#plt.ylim((0.7,1.15))
#plt.xlim((0, 300))
#plt.title('Plot of Convergence of Truncated Model Parameters to Full Model Parameters')
#plt.title('Plot of Convergence of Truncated and Full Model Parameters')
plt.legend()
plt.savefig('figuref.pdf', dpi=300, transparent=True)
plt.show()
