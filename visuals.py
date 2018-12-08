# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:54:09 2018

@author: Tyler
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import spect

#Specify global variables
filename = "./data/raw/subjects.mat"
sample_rate = 1000
start_seconds = 0
end_seconds = 1000
subject_num = 3

spectrogram_time_resolution_ms = 1
spectrogram_freq_resolution_hz = 4
spectrogram_freq_cutoff_hz = 160 #Will be half of what you specify

spectrogram_time_resolution = spectrogram_time_resolution_ms / sample_rate

#Load the dataset
mat = sio.loadmat(filename)
matobj = "subject"
mdata = mat[matobj][subject_num][0]
matx = np.array(mdata).ravel()
dataset = matx[start_seconds*sample_rate:end_seconds*sample_rate]

#Generate the spectrogram
points, freqs, bins = spect.specgram(dataset, sample_rate, time_resolution=spectrogram_time_resolution,
                                     frequency_resolution=spectrogram_freq_resolution_hz, high_frequency_cutoff=160)
  

# Reshape a numpy array 'a' of shape (n, x) to form shape((n - window_size), window_size, x))
def rolling_window(a, window, step_size):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

plt.figure()
plt.plot(dataset)
  
#extent = (bins[0], bins[-1], freqs[0], freqs[-1])
#plt.figure()
#plt.imshow(points, aspect='auto', origin='lower', extent=extent, vmax=.01)
#plt.plot(points[:1000])

