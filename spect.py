# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 23:53:34 2018

@author: Tyler
"""

import numpy as np
import scipy.signal

import matplotlib.pyplot as mlab

# from http://code.google.com/p/python-neural-analysis-scripts/source/browse/trunk/scripts/Filtering/Fir.py

def spectral_inversion(kernel):
    kernel = -kernel
    kernel[len(kernel)/2] += 1.0
    return kernel

def make_fir_filter(sampling_freq, critical_freq, kernel_window, taps, kind, **kwargs):
    nyquist_freq = sampling_freq/2
    critical_freq = np.array(critical_freq, dtype = np.float64)
    normalized_critical_freq = critical_freq/nyquist_freq

    if not taps % 2: #The order must be even for high and bandpass
        taps += 1

    if kind.lower() in ['low','low pass', 'low_pass']:
        kernel = scipy.signal.firwin(taps, normalized_critical_freq,
                               window=kernel_window, **kwargs)

    elif kind.lower() in ['high','high pass', 'high_pass']:
        lp_kernel = scipy.signal.firwin(taps, normalized_critical_freq,
                                  window = kernel_window, **kwargs)
        kernel = spectral_inversion(lp_kernel)
          
    elif kind.lower() in ['band','band pass', 'band_pass']:
        lp_kernel = scipy.signal.firwin(taps, normalized_critical_freq[0],
                                  window = kernel_window, **kwargs)
        hp_kernel = scipy.signal.firwin(taps, normalized_critical_freq[1],
                                  window = kernel_window, **kwargs)
        hp_kernel = spectral_inversion(hp_kernel)
        
        bp_kernel = spectral_inversion(lp_kernel + hp_kernel)
        kernel = bp_kernel
    
    return kernel

def fir_filter(sig, sampling_freq, critical_freq, kernel_window = 'hamming', taps = 101, kind = 'band', **kwargs):
    """
    Build a filter kernel of type <kind> and apply it to the signal
    Returns the filtered signal.

    Inputs:
        sig          : an n element sequence
        sampling_freq   : rate of data collection (Hz)
        critical_freq   : high and low cutoffs for filtering 
                        -- for bandpass this is a 2 element seq.
        kernel_window   : a string from the list - boxcar, triang, blackman,
                             hamming, bartlett, parzen, bohman, blackmanharris,                              nuttall, barthann
        taps            : the number of taps in the kernel (integer)
        kind            : the kind of filtering to be performed (high,low,band)
        **kwargs        : keywords passed onto scipy.firwin
    Returns:
        filtered signal : an n element seq
    """

    kernel = make_fir_filter(sampling_freq, critical_freq, kernel_window, taps, kind, **kwargs) 


    return np.roll(scipy.signal.lfilter(kernel, [1], sig), -taps/2+1)


# from http://code.google.com/p/python-neural-analysis-scripts/source/browse/trunk/LFP/signal_utils.py


def find_NFFT(frequency_resolution, sampling_frequency, 
              force_power_of_two=False):
    #This function returns the NFFT
    NFFT = (sampling_frequency*1.0)/frequency_resolution-2
    if force_power_of_two:
        pow_of_two = 1
        pot_nfft = 2**pow_of_two
        while pot_nfft < NFFT:
            pow_of_two += 1
            pot_nfft = 2**pow_of_two
        return pot_nfft
    else:
        return NFFT
        
def find_frequency_resolution(NFFT, sampling_frequency):
    return (sampling_frequency*1.0)/(NFFT + 2)

def find_NFFT_and_noverlap(frequency_resolution, sampling_frequency,
                           time_resolution, num_data_samples):
    NFFT =  find_NFFT(frequency_resolution, sampling_frequency)
    
    # finds the power of two which is just greater than NFFT
    pow_of_two = 1
    pot_nfft = 2**pow_of_two
    noverlap = pot_nfft-sampling_frequency*time_resolution
    while pot_nfft < NFFT or noverlap < 0:
        pow_of_two += 1
        pot_nfft = 2**pow_of_two
        noverlap = pot_nfft-sampling_frequency*time_resolution

    pot_frequency_resolution = find_frequency_resolution(pot_nfft, 
                                                         sampling_frequency)
    
    return {'NFFT':int(NFFT), 'power_of_two_NFFT':int(pot_nfft), 
            'noverlap':int(noverlap), 
            'power_of_two_frequency_resolution':pot_frequency_resolution} 

def resample_signal(signal, prev_sample_rate, new_sample_rate):
    rate_factor = new_sample_rate/float(prev_sample_rate)
    return scipy.signal.resample(signal, int(len(signal)*rate_factor))    

def psd(signal, sampling_frequency, frequency_resolution,
        high_frequency_cutoff=None,  axes=None, **kwargs):
    """
    This function wraps matplotlib.mlab.psd to provide a more intuitive 
        interface.
    Inputs:
        signal                  : the input signal (a one dimensional array)
        sampling_frequency      : the sampling frequency of signal
        frequency_resolution    : the desired frequency resolution of the 
                                    specgram.  this is the guaranteed worst
                                    frequency resolution.
        --keyword arguments--
        axes=None               : If an Axes instance is passed then it will
                                  plot to that.
        **kwargs                : Arguments passed on to 
                                   matplotlib.mlab.specgram
    Returns:
        Pxx
        freqs
    """
    if (high_frequency_cutoff is not None 
        and high_frequency_cutoff < sampling_frequency):
        resampled_signal = resample_signal(signal, sampling_frequency, 
                                                    high_frequency_cutoff)
    else:
        high_frequency_cutoff = sampling_frequency
        resampled_signal = signal
    num_data_samples = len(resampled_signal)
    NFFT= find_NFFT(frequency_resolution, high_frequency_cutoff, 
                    force_power_of_two=True) 
    if axes is not None:
        return axes.psd(resampled_signal, NFFT=NFFT, 
                             Fs=high_frequency_cutoff, 
                             noverlap=0, **kwargs)
    else:
        return mlab.psd(resampled_signal, NFFT=NFFT, 
                                        Fs=high_frequency_cutoff, 
                                        noverlap=0, **kwargs)

def plot_specgram(Pxx, freqs, bins, axes, logscale=True):
    if logscale:
        plotted_Pxx = 10*np.log10(Pxx)
    else:
        plotted_Pxx = Pxx
    extent = (bins[0], bins[-1], freqs[0], freqs[-1])
    im = axes.imshow(plotted_Pxx, aspect='auto', origin='lower', extent=extent)
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Frequency (Hz)')
    return im

def specgram(signal, sampling_frequency, time_resolution, 
             frequency_resolution, bath_signals=[], 
             high_frequency_cutoff=None,  axes=None, logscale=True, **kwargs):
    """
    This function wraps matplotlib.mlab.specgram to provide a more intuitive 
        interface.
    Inputs:
        signal                  : the input signal (a one dimensional array)
        sampling_frequency      : the sampling frequency of signal
        time_resolution         : the desired time resolution of the specgram
                                    this is the guaranteed worst time resolution
        frequency_resolution    : the desired frequency resolution of the 
                                    specgram.  this is the guaranteed worst
                                    frequency resolution.
        --keyword arguments--
        bath_signals            : Subtracts a bath signal from the spectrogram
        axes=None               : If an Axes instance is passed then it will
                                  plot to that.
        **kwargs                : Arguments passed on to 
                                   matplotlib.mlab.specgram
    Returns:
        If axes is None:
            Pxx
            freqs
            bins
        if axes is an Axes instance:
            Pxx, freqs, bins, and im
    """
    if (high_frequency_cutoff is not None 
        and high_frequency_cutoff < sampling_frequency):
        resampled_signal = resample_signal(signal, sampling_frequency, 
                                                    high_frequency_cutoff)
    else:
        high_frequency_cutoff = sampling_frequency
        resampled_signal = signal
    num_data_samples = len(resampled_signal)
    specgram_settings = find_NFFT_and_noverlap(frequency_resolution, 
                                               high_frequency_cutoff, 
                                               time_resolution, 
                                               num_data_samples)
    NFFT     = specgram_settings['power_of_two_NFFT']
    noverlap = specgram_settings['noverlap']
    Pxx, freqs, bins, im = mlab.specgram(resampled_signal, 
                                                NFFT=NFFT, 
                                                Fs=high_frequency_cutoff, 
                                                noverlap=noverlap, **kwargs)
    plotted_Pxx = Pxx
    if bath_signals:
        bath_signal = np.hstack(bath_signals)
        psd_Pxx, psd_freqs = psd(bath_signal, sampling_frequency, 
                                 frequency_resolution,
                                 high_frequency_cutoff=high_frequency_cutoff ) 
        plotted_Pxx = (Pxx.T/psd_Pxx).T

    if axes is not None:
        im = plot_specgram(plotted_Pxx, freqs, bins, axes, logscale=logscale)
        return plotted_Pxx, freqs, bins, im
    return plotted_Pxx, freqs, bins

#import acq.scanimage as sa

#data = sa.parseXSG(f)
#points, freqs, bins = specgram(data['ephys']['chan0'][:890000], 10000, time_resolution=0.1, frequency_resolution=0.5, high_frequency_cutoff=80)
#dataset = pd.read_csv("./data/raw/rawr_sub4.csv",header=None,dtype='float64',keep_default_na=False).values
#data = np.transpose(data)[0]

#mlab.plot(datasetss)
#for i in np.arange(0,num_lines):
#    row = next(reader)  # gets the first line
#    arr = np.array(list(map(float, row)))
#    dataset[i,0:len(arr)] = arr

