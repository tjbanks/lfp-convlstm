# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:54:09 2018

@author: Tyler
"""
import warnings
import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import spect
import scipy

warnings.filterwarnings("ignore")

#Specify global variables
filename = "./data/raw/subjects.mat"
sample_rate = fs = 1000
start_seconds = 0
end_seconds = 1000
subject_num = 3

spectrogram_time_resolution_ms = 12
spectrogram_freq_resolution_hz = 4
spectrogram_freq_cutoff_hz = 160 #Will be half of what you specify

spectrogram_time_resolution = spectrogram_time_resolution_ms / sample_rate
print("spectrogram time resolution: ",spectrogram_time_resolution)

#Load the dataset
mat = sio.loadmat(filename)
matobj = "subject"
mdata = mat[matobj][subject_num][0]
matx = np.array(mdata).ravel()
dataset = matx[start_seconds*sample_rate:end_seconds*sample_rate]
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def play_movie(r,r1):
    #just for fun
    fig = plt.figure()
    pausetime = .0001
    img = None
    for i,(f,f1) in enumerate(zip(r,r1)):
        s1 = plt.subplot(212)
        im=f#plt.imread(f)
        if img is None:
            img = plt.imshow(im)
        else:
            img.set_data(im)
        if plt.fignum_exists(fig.number):
            plt.xticks(range(0,101,20),range(i,i+101,20))
            plt.yticks(range(0,33,8),range(0,81,20))
            
            # Figure is still opened
            plt.pause(pausetime)
            plt.draw()    
            s1.set_title('Spectrogram')
            plt.xlabel('time (ms)')
            plt.ylabel('frequency (Hz)')
            
            s2 = plt.subplot(211)
            axes = plt.gca()
            s2.clear()
            
            plt.xticks(range(0,101,20),range(i,i+101,20))
            s2.plot(r1[i:i+101,1])
            s2.margins(x=0,y=0)
            axes.set_ylim([-1500,1500])
            s2.set_title('LFP')
            plt.xlabel('time (ms)')
            plt.ylabel('power')
        else:
            break
        
        
        #plt.xticks(range(0,101,20),range(i,i+101,20))
        # Figure is still opened
        #plt.pause(pausetime)
        #s2.draw()  
            

# Reshape a numpy array 'a' of shape (n, x) to form shape((n - window_size), window_size, x))
def rolling_window(a, window, step_size):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def gen_data():
    #plt.plot(dataset)
    #plt.figure()
    #highcut = 100
    #dataset = butter_bandpass_filter(dataset, 1, highcut, fs, order=4)
    #plt.plot(dataset)
    #Generate the spectrogram
    plt.figure()
    points, freqs, bins = spect.specgram(dataset, sample_rate, time_resolution=spectrogram_time_resolution,
                                         frequency_resolution=spectrogram_freq_resolution_hz, high_frequency_cutoff=spectrogram_freq_cutoff_hz)
    
    #f, t, Sxx = scipy.signal.spectrogram(dataset, sample_rate,nperseg=12,nfft=32)
    #plt.pcolormesh(t, f, Sxx)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()  
    
    #print(np.shape(f))
    #print(np.shape(t))
    #print(np.shape(Sxx))
    
    #plt.figure()
    #plt.plot(dataset)
      
    #extent = (bins[0], bins[-1], freqs[0], freqs[-1])
    #print(np.shape(bins)," ",np.shape(freqs))
    import scipy.ndimage
    print(np.shape(points))
    s = 1000000/np.shape(points)[1]
    points = scipy.ndimage.zoom(points, (1,s), order=3)
    print(np.shape(points))
    
    #plt.figure()
    #plt.pcolormesh(points)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show() 
    
    #print(bins)
    
    #print(freqs)
    #print("bins: ", bins)
    x = rolling_window(points, 100, 1)
    print("X Original Shape: ",np.shape(x))
    x = np.rollaxis(x,1)
    print("X New Shape after axis roll: ",np.shape(x))
    y = rolling_window(dataset, 100, 1)#100 for movie
    print("Y Shape: ", np.shape(y))
    
    #play_movie(x,y)
    
    y = rolling_window(dataset, 110, 1)
    yy = []
    for temp in y:
        yy.append(temp[-1])
    
    print("Y Shape 10 out: ",np.shape(yy))
    x = x[:len(x)-10]
    print("X New Shape: ",np.shape(x))
    
    return x,yy

#Scale
#min/max scaler


#Steps so far:
    #Load matlab subject
    #Take first 1000 seconds
    #Perform spectrogram analysis
    #Interpolate the dataset back to original size
    #
    