#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 22:19:26 2021
@author: bartek
"""

import pyedflib
from scipy.fftpack import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch, hanning, butter, lfilter, resample, iirfilter, spectrogram

#%%
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


#%% path and reading EDF file

# path to edf+ file
iedf = ''

f = pyedflib.EdfReader(iedf)

annots = f.readAnnotations()

# provides the number of signals aka channels in file
num_chans = f.signals_in_file

# number of samples, sf * file size (second)
n_samps = f.getNSamples()[0]

# grabs the sf of all channels
downsamplefactor = 1
sf_all = f.getSampleFrequencies()/downsamplefactor

#%% shaping an empty array: n_channels * n_samples

data = np.zeros((num_chans, int(n_samps/downsamplefactor)))

# init empty channels list

chan_labels = []

# fill arrays
for i in np.arange(num_chans):
    chan_labels.append(f.signal_label(i).decode('latin-1').strip())
    data[i, :] = f.readSignal(i)

channel_number= 0

for label in chan_labels:
    print(channel_number, label)
    channel_number += 1

f.close()

#%% channels of interest/in GM

# init channels of interest from chan_labels
# below is an example of selected contacts 
mpfc = 1
lateral_ofc = 5
cingulate = 11
insula = 81
amygdala = 23
hippocampus = 43

chan_num = [mpfc, lateral_ofc, cingulate, insula, amygdala, hippocampus]

#grabs the sampling frequency, should be the same for all channels - make sure

sf=sf_all[mpfc]

print("sample frequency:", sf)

#%% apply filtering

# bandpass filter
lowCut = 4
highCut = 100
data_all = butterBandpass(data, lowCut, highCut, sf, 3)

# notch filter
data_all = notchFilter(data_all, sf, 1, 60, 1, 'butter')

data_ch = data_all[chan_num,:]

#%% Compute spectrograms

# ... the interval size,
interval = int(sf)        

# ... and the overlap intervals
overlap = int(sf * .90) 
regions = {0:'Medial Prefrontal Cortex', 1:'Lateral Orbitofrontal Cortex', 2:'Anterior Cingulate Cortex',\
           3:'Insula', 4:'Amygdala',5:'Hippocampus'}


    
for ind, ichannel in enumerate(data_ch):
    f, t, Sxx = spectrogram(ichannel, fs=sf, window=('tukey', 0.6), nperseg=interval, noverlap=overlap)
    
    
    plt.pcolormesh(t, f, 14 * np.log10(Sxx), cmap='jet', shading='gouraud')  # Plot the result
    plt.colorbar()                # ... with a color bar,
    plt.ylim([4, 150])             # ... set the frequency range,
    plt.xlabel('Time (s)')        # ... and label the axes
    plt.ylabel('Frequency (Hz)')
    plt.title(label = regions[ind])
    plt.clim([40,-40])  		# ... power scale, should be balanced (+y,-y)
    plt.show()
    
    
# %% raw signal and power spectral density
# check for artefacts / noise in signal

data_channel = data_all[1,:] # specify channel here

# define sampling frequency and time vector
time = np.arange(data_channel.size) / sf

# plots the raw eeg signal
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.plot(time, data_ch, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.xlim([time.min(), time.max()])
plt.title('EEG data')


# define window length for welch's (4 seconds)
win = 4 * sf
freqs, psd = signal.welch(data_channel, sf, nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 8))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, highCut])
