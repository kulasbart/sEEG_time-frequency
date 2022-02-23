# sEEG time-frequency domain workflow

Input: file.edf neural time series data, organized into sets of epoched edf files (same duration)

Includes preprocessing steps, computation of spectrogram for time and frequency domain analyses of multichannel intracranial EEG data (EDF+), and tools to inspect multichannel eeg timeseries data. 

- For comparison of trials across different recorded contacts: the averaging script plots the mean spectral power changes of clipped edf files from the same individual... recommended to plot individual trials to inspect trials for artifact. Allows flexibility to switch between contacts, based on `chan_num` indexing of recording channels

- z-score baseline normalization script allows for group analyses across subjects. For group analyses of subjects with similar electrode coverage: normalizes to baseline activity (pre-stimuli). The script writes a new csv file which are merged using sEEG_inter_individual.py

- band-power averaging script allows you to define frequency ranges and plot power changes, % or voltage z-scored (voltage), averaged across subjects