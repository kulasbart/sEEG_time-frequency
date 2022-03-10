# sEEG time-frequency domain workflow

Includes minimal preprocessing steps to maintain integrity of the raw data, computation of spectrograms for time and frequency domain analyses of multichannel intracranial EEG data (EDF+), z-score power change across traditional frequency bands (theta, alpha, beta, gamma, and tools for inspecting multichannel eeg timeseries data

## Data type

Input: file.edf neural time series data, multichannel intracranial EEG data (EDF+) must be epoched into equal length epochs

Sample data provided: set of 15 second epochs with stimulus at 5 seconds, set of 15 second epochs with a high frequency (50 Hz) pulsating biphasic current conducted through the prefrontal brain region at 5 seconds - hope to create a script to denoise these files

## Analysis and plots

- For comparison of trials across different recorded contacts: the averaging script plots the mean spectral power changes of clipped edf files from the same individual... recommended to plot individual trials to inspect trials for artifact. Allows flexibility to switch between contacts, based on `chan_num` indexing of recording channels

- z-score baseline normalization script allows for group analyses across subjects. For group analyses of subjects with similar electrode coverage: normalizes to baseline activity (pre-stimuli). The script writes a new csv file which are merged using sEEG_inter_individual.py

- band-power averaging script allows you to define frequency ranges and plot power changes, % or voltage z-scored (voltage), averaged across subjects