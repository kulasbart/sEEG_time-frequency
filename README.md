# sEEG time-frequency domain workflow

Input: file.edf neural time series data, set of epoched edf files of the same duration

Includes preprocessing steps, computation of spectrogram for time and frequency domain analyses of multichannel intracranial EEG data (EDF+), and tools to inspect multichannel eeg timeseries data. 

For comparison of trials across different recorded contacts: the averaging script plots the mean spectral power over time of clipped edf files from the same individual... ensure trial are free of artifact

---IN PROGRESS---

Future additions: z-score normalization for inter-individual and group analyses. Also planning to expand the workflow to inspect phase coupling across ieeg channels