# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:54:09 2018

@author: Tyler
"""
import pandas as pd
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def load_csv(path):
    data = pd.read_csv(path)
    return data

def plot(data):
   plt.plot(data)

def get_spectrogram(x):
    return
    
data = load_csv("subject1_seg.csv")
print(data)
#plot(data)