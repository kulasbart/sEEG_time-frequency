#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 22:19:26 2021
@author: bartek

"""

import glob
import pyedflib
from scipy.fftpack import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirfilter, spectrogram


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

#%% collects epoched files from folder, replace 'path' with path to folder ... files must be same length

data_dir = ('insert path to epoched files')   # ... insert path here

#%%
#reading edf file - assumes all channels have the same sampling frequency

f = pyedflib.EdfReader(glob.glob(data_dir)[0])  
annots = f.readAnnotations()
num_chans = f.signals_in_file
n_samps = f.getNSamples()[0]

downsamplefactor = 1
sf_all = f.getSampleFrequencies()/downsamplefactor
f.close()

# list of channels imported from edf
chan_labels = []

for i in np.arange(num_chans):
    chan_labels.append(str(i)+'_' + (f.signal_label(i).decode('latin-1').strip()))

#%% import channels from edf

chan_labels = []

for i in np.arange(num_chans):
    chan_labels.append(str(i)+'_' + (f.signal_label(i).decode('latin-1').strip()))
epochs = []

# read epochs

for iepoch in glob.glob(data_dir):
    
    f = pyedflib.EdfReader(iepoch)
    data = np.zeros((num_chans, int(n_samps/downsamplefactor)))
    count = 0 
    
    for i in np.arange(num_chans):

        data[count, :] = f.readSignal(count)
        count += 1
        
    epochs.append(data)
        
    f.close()
    
print(epochs[0].shape)

#plots raw ieeg signal for inspection
plt.figure(figsize=(10,5))
plt.plot(np.arange(n_samps), epochs[0][1])
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.show()

#%% set parameters and apply filtering
sf = sf_all[0]

lowCut = 4
highCut = 250

data_all = []

for iepoch in epochs:
    
    epochs_f = butterBandpass(iepoch, lowCut, highCut, sf, 3)
    epoch_f = notchFilter(epochs_f, sf, 1, 60, 1, 'butter')
    
    data_all.append(epoch_f)

#plots the filtered signal for inspection
plt.figure(figsize=(10,5))
plt.plot(np.arange(n_samps), data_all[0][1])
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.show()

#%% set contact number, interval size and overlap intervals

contact =       # ... set contact number - refer to indexing chan_labels
interval = int(sf)
overlap = int(sf * .90)
count = 0
sxx_values =[]

for epoch_num in data_all:
    f, t, Sxx = spectrogram(epoch_num[contact], fs=sf, window=('tukey', 0.6), nperseg=interval, noverlap=overlap)
    sxx_values.append(Sxx)
    count += 1
        
sxx_m = np.sum(sxx_values[0:count-1], axis=0) / (count)

# plots spectrogram
plt.figure(figsize=(12,8))
plt.pcolormesh(t, f, np.log10(sxx_m), cmap='jet', shading='gouraud')  # Plot the result
plt.colorbar()               
plt.ylim([4, 150])             # ... set the frequency range,
plt.xlabel('Time (s)')        # ... and label the axes
plt.ylabel('Frequency (Hz)')
plt.axvline(x=5 , color='k', linestyle='--')   # ... mark stimulus
plt.axvline(x=7 , color='k', linestyle='--')   
plt.clim([1,-1])  		# ... power scale, can play around with scaling but it should be balanced (+y,-y)
plt.show()
