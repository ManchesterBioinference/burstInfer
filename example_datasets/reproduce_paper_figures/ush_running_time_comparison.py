# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 05:49:58 2020

@author: Jon
"""
import numpy as np
import matplotlib.pyplot as plt

index = np.array([7,8,9,10,11,12,13,14,15,16,17,18,19])

# This is just a placeholder for now - it shouldn't change much with
# different window sizes. 
python = np.array([np.mean([91,102,97]), np.mean([91,92,96]), np.mean([98,85,80]), np.mean([75,74,85]), \
                   np.mean([75,72,85]), np.mean([71,81,71]), np.mean([84,68,71]), np.mean([79,82,74]), \
                   np.mean([87,76,72]), np.mean([91,92,75]), \
                   np.mean([72,71,74]), np.mean([75,82,82]), np.mean([71,84,82])])

#matlab = np.array([0.84,1.03,2.08,3.60,6.25,13.92,23.9,44.6,89,210,451,928,2507])

matlab = np.array([0.565,0.816,1.448,2.317,5.36,11.351,21.249,40.817,85.904,134.437,308.95,733.952,2205.506])

params = {'legend.fontsize': 18}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'arial'

#plt.figure(1, figsize = (14,10), dpi = 300)
plt.figure(1, figsize = (8,6), dpi=300)
#plt.figure(figsize=(12,7))
plt.scatter(index, python, marker = 'x', c = 'r', s = 80)
plt.scatter(index, matlab, c = 'C0', s = 80)
plt.xlabel('Window Size')
plt.ylabel('Running Time (seconds)')
plt.xticks(np.arange(7,20,1))
plt.yticks(np.arange(0,2700,400))
plt.xlim(6,20)
plt.gca().legend(('Truncated Model (Python)','Full Model (Matlab)'))
#plt.title('Plot of Running Time for Full and Truncated Models as a Function of Window Size')

plt.savefig('ush_running_time_completely_redone_plot_v2.pdf', dpi=300, transparent=True)  
#plt.savefig("ush_running_time_completely_redone_plot_v2.svg")