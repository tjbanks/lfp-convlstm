# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:23:14 2018

@author: Tyler
"""

import gen_data as lfp

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


x,y = lfp.gen_data()

#Need to reshape first
x_scaler = MinMaxScaler(feature_range=(0,1))
x_scaled = x_scaler.fit_transform(x)

y_scaler = MinMaxScaler(feature_range=(-1,1))
y_scaled = x_scaler.fit_transform(y)

print(x)
print(x_scaled)