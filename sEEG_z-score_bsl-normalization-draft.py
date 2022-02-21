#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 21:22:21 2022

@author: bartek
"""

import pyedflib
from scipy.fftpack import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hanning, butter, lfilter, resample, iirfilter, spectrogram
import glob
import pandas as pd


def get_fft_values(y_values, T, N, f_s):

    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def butter_bandpass(lowcut, highcut, fs, order):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a
	   
def butterBandpass(d, lowcut, highcut, fs, order):
	b, a = butter_bandpass(lowcut, highcut, fs, order)
	y = lfilter(b, a, d)
	return y

def notchFilter(data, fs, band, frq, order, filter_type):
	nyq = fs/2.0
	low = frq - band/2.0
	high = frq + band/2.0
	low = low/nyq
	high = high/nyq
	b, a = iirfilter(order, [low, high], btype='bandstop', analog=False, ftype=filter_type)
	filtered_data = lfilter(b, a, data)
	return filtered_data

def get_bsl_Mean(bsl_sxx_values):
    sxx_len = len(bsl_sxx_values)
    bsl_mean = []
    for hz_band in bsl_sxx_values:
        mean = (np.sum(hz_band))/sxx_len
        bsl_mean.append(mean)
    return bsl_mean

def get_bsl_SD(bsl_sxx_values):
    ind = 0
    bsl_SD = []

    for hz_band in bsl_sxx_values:
        SD = np.std(bsl_sxx_values[ind])
        bsl_SD.append(SD)
        ind += 1
    return bsl_SD

#%% path to folder

files = glob.glob('') # insert path to folder and wildcard condition


files_r = ['r\''+files for files in files] #use for external drives (mac)
files_r

#%% read and writes edf file

iedf = files[0]

f = pyedflib.EdfReader(iedf)    
annots = f.readAnnotations()
num_chans = f.signals_in_file
n_samps = f.getNSamples()[0]

downsamplefactor = 1
sf_all = f.getSampleFrequencies()/downsamplefactor
f.close()

# init empty channels list
chan_labels = []

for i in np.arange(num_chans):
    chan_labels.append(str(i)+'_' + (f.signal_label(i).decode('latin-1').strip()))
chan_labels

#%% inspect signal and apply filtering

#data = np.zeros((num_chans, int(n_samps/downsamplefactor)))

epochs = []

for iepoch in files:
    
    f = pyedflib.EdfReader(iepoch)
    data = np.zeros((num_chans, int(n_samps/downsamplefactor)))
    count = 0 
    
    for i in np.arange(num_chans):

        data[count, :] = f.readSignal(count)
        count += 1
        
    epochs.append(data)
        
    f.close()
    
epochs[0].shape

plt.figure(figsize=(10,5))
plt.plot(np.arange(n_samps), epochs[0][1])
plt.show()

#sf assuming all channel are sampling at same frequency
sf = sf_all[0]

# bandpass filter
lowCut = 4
highCut = 150

data_all = []

for iepoch in epochs:
    
    epochs_f = butterBandpass(iepoch, lowCut, highCut, sf, 3)
    epoch_f = notchFilter(epochs_f, sf, 1, 60, 1, 'butter')
    
    data_all.append(epoch_f)
    
plt.figure(figsize=(10,5))
plt.plot(np.arange(n_samps), data_all[0][1])
plt.show()

#%% plots averaged trials

# interval size and overlap intervals
interval = int(sf)
overlap = int(sf * .90) 

# specify the contact number
contact =   # refer to chan_labels
count = 0
sxx_values =[]

while count < len(files):
    f, t, Sxx = spectrogram(data_all[count][contact], fs=sf, window=('tukey', 0.6), nperseg=interval, noverlap=overlap)
    sxx_values.append(Sxx)
    count += 1
        
sxx_m = np.sum(sxx_values[0:count-1], axis=0) / (count)

plt.figure(figsize=(12,8))
plt.pcolormesh(t, f, np.log10(sxx_m), cmap='jet',shading='gouraud')
plt.colorbar()                # ... with a color bar,
plt.ylim([4,150])             # ... frequency range... large for data inspection
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(chan_labels[contact])
#plt.axvline(x=5 , color='k', linestyle='--')   # ... stimulus start
#plt.axvline(x=7 , color='k', linestyle='--')   # ... end
plt.clim([2,-2])  		# ... power scale, should be balanced (+y,-y)
plt.show()

#%%  baseline z-score

# be cautious of edge effect when selecting baseline
sxx_bsl = sxx_m[:,5:40] # using first 40 fft windows in this case

bsl_m = get_bsl_Mean(sxx_bsl)
bsl_SD = get_bsl_SD(sxx_bsl)

sxx_normalized = []
hz_num = 0

for hz in sxx_m:
    z_scored = (hz-bsl_m[hz_num])/bsl_SD[hz_num]
    hz_num += 1
    sxx_normalized.append(z_scored)

# baseline normalized spectrogram
plt.figure(figsize=(12,8))
plt.pcolormesh(t, f, np.log10(sxx_normalized), cmap='jet',shading='gouraud')  # Plot the result
plt.colorbar()                # ... with a color bar,
plt.xlim([-4,8]) 
plt.xlabel('Time (s)',fontweight='bold',fontsize='17')
plt.xticks(color='k')
plt.xticks(fontsize= '14')
plt.ylabel('Frequency (Hz)',fontweight='bold',fontsize='17')
plt.ylim([4,50])             # ... set the frequency range,
plt.yticks(fontsize= '14')
#plt.axvline(x=0 , color='k', linestyle='--')   # ... stimulus start
#plt.axvline(x=7 , color='k', linestyle='--')   # ... end
plt.clim([1,-1])  		# ... power scale, should be balanced (+y,-y)
plt.show()

#%% load to spreadsheet

pt_num = '' # insert subject identifier prefix

df = pd.DataFrame(sxx_normalized)
writer = pd.ExcelWriter(pt_num+'_'+str(chan_labels[contact])+'_sxx_norm.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name=pt_num+chan_labels[contact], index=False)
writer.save()


#%% Plots the individual trials

file_num = 0

for sxx_i in sxx_values:

    plt.figure(figsize=(8,6))
    plt.pcolormesh(t, f, 16 * np.log10(sxx_i), cmap='jet', shading='gouraud') 
    plt.colorbar()               
    plt.ylim([4, 150])           
    plt.xlabel('Time (s)')    
    plt.ylabel('Frequency (Hz)')
    plt.title(files[file_num]) # file name ... useful if you include desc or timestamp in name
    plt.axvline(x=5 , color='k', linestyle='--')   
    plt.axvline(x=7 , color='k', linestyle='--')   
    plt.clim([40,-40])  		
    plt.show()
    
    file_num += 1
