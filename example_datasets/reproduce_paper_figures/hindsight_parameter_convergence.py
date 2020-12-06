# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 00:59:12 2020

@author: Jon
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

true_kon = 0.070353336309323 # This is to Matlab
true_koff = 0.464397161485740
true_emission = 41354.89875953371
true_noise = 15364.95018917365

kon_256 = 0.0703975370151
koff_256 = 0.464528461567
emission_256 = 41355.4421665
noise_256 = 15383.4553458

kon_128 = 0.0702521547606
koff_128 = 0.462781716998
emission_128 = 41349.5580493
noise_128 = 15425.418704

kon_64 = 0.0693153852427
koff_64 = 0.467115326453
emission_64 = 42188.8476558
noise_64 = 15603.8805624

kon_32 = 0.0670500953392
koff_32 = 0.451529655946
emission_32 = 42357.8369306
noise_32 = 15950.0835477

kon_16 = 0.0629044588372
koff_16 = 0.419355892181
emission_16 = 42031.6615316
noise_16 = 16549.2031647

kon_8 = 0.0620334755086
koff_8 = 0.434113981492
emission_8 = 44399.6417218
noise_8 = 17317.3710659


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
    
    
#%%
params = {'legend.fontsize': 18}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'arial'

noise_dict = {'8': rel_error(true_noise, noise_8), '16': rel_error(true_noise, noise_16),
              '32': rel_error(true_noise, noise_32), '64': rel_error(true_noise, noise_64),
              '128': rel_error(true_noise, noise_128), '256': rel_error(true_noise, noise_256)}
noise_names = list(noise_dict.keys())
noise_values = list(noise_dict.values())

prob_off_dict = {'8': rel_error(true_koff, koff_8), '16': rel_error(true_koff, koff_16), '32': rel_error(true_koff, koff_32),
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

plt.figure(1, figsize = (8,6), dpi=300)
plt.plot(noise_names, noise_values, label = 'Noise', marker = 's', linewidth=3)
plt.plot(prob_off_names, prob_off_values, label = '$k_{off}$', marker = '.', linewidth=3)
plt.plot(prob_on_names, prob_on_values, label = '$k_{on}$', marker = 'o', linewidth=3)
plt.plot(emissions_names, emissions_values, label = 'Emission', marker = 'v', linewidth=3)
plt.xlabel('Number of Allowed States (M)')
plt.ylabel('Relative Error (%)')
plt.ylim((0,35))
#plt.xlim((0, 300))
#plt.title('Plot of Convergence of Truncated Model Parameters to Full Model Parameters')
#plt.title('Plot of Convergence of Truncated and Full Model Parameters')
plt.legend()

plt.savefig('pebwt_reborn_convergence.pdf', dpi=300, transparent=True)
plt.savefig('pebwt_reborn_convergence.svg')