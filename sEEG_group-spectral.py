#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 23:12:25 2022

@author: bartek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob


#%% reading csv/xls files (150,140), path to normalized data

data_dir = ('insert path to normalized xls files')  # ... insert path here


hz = 150  # select the highest Hz band you are interested in
count = 0
sxx_z = []

for sxx_norm in glob.glob(data_dir):
    df = pd.read_excel(glob.glob(data_dir)[count])
    df = df[0:hz]
    s = df.to_numpy()
    sxx_z.append(s)
    count += 1
    
sxx_z_m = np.sum(sxx_z[0:count-1], axis=0) / count #mean z-score

t = np.linspace(0.5,14.48144531,140) # ... generate time axis 
t = t - 5  # ... zero stimulus time, 5 s in this case
f = np.linspace(0,149,150) # Hz scale

#%% plot!


plt.figure(figsize=(12,8))
plt.pcolormesh(t, f, np.log10(sxx_z_m), cmap='jet',shading='gouraud')  # Plot the result
plt.colorbar()                # ... with a color bar,
plt.ylim([4,50])             # ... set the frequency range,
plt.xlabel('Time (s)')        # ... and label the axes
plt.ylabel('Frequency (Hz)')
plt.axvline(x=0 , color='k', linestyle='--')   # ... denotes stimulus onset
plt.axvline(x=2 , color='k', linestyle='--')   
plt.clim([.75,-.75])  		# ... power scale, should be balanced (+y,-y)
plt.show()

#%%
# check to ensure all files are the same size

num = 0
for i in sxx_z:
    print(glob.glob(data_dir)[num])
    print(len(i))
    print(sxx_z[num].shape)
    num+=1