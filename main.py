# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:23:14 2018

@author: Tyler
"""

import gen_data as lfp

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import numpy as np
import logging as log

from keras.wrappers.scikit_learn import KerasClassifier
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,TimeDistributed,Flatten,LSTM,Dense,Reshape

log.getLogger().setLevel(log.DEBUG)


def get_data():
    x_orig,y_orig = lfp.gen_data()
    lookahead = 10
    window = 100
    window_step = 1
    
    
    #
    # SCALING SECTION
    #
    
    log.debug("X Original Shape: {}".format(np.shape(x_orig)))
    log.debug("Y Original Shape: {}".format(np.shape(y_orig)))
    
    x = x_orig.reshape(-1,1)
    x_scaler = MinMaxScaler(feature_range=(0,1))
    x_scaled = x_scaler.fit_transform(np.float32(x))
    x = x_scaled.reshape(np.shape(x_orig)[0],-1)
    
    y = y_orig.reshape(-1,1)
    y_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = y_scaler.fit_transform(y)
    y = y_scaled.reshape(np.shape(y_orig)[0])
    
    log.debug("X Transformed Shape: {}".format(np.shape(x)))
    log.debug("Y Transformed Shape: {}".format(np.shape(y)))
    
    
    #
    # ROLLING WINDOW SECTION
    #
    
    x = lfp.rolling_window(x, window, window_step)
    log.debug("X Rolling Window Shape: {}".format(np.shape(x)))
    x = np.rollaxis(x,1)
    log.debug("X New Shape after axis roll: {}".format(np.shape(x)))
    
    #y = lfp.rolling_window(y, 100, 1)#100 for movie
    #print("Y Shape: ", np.shape(y))
    #lfp.play_movie(x,y)
    
    y = lfp.rolling_window(y, window+lookahead, window_step)
    yy = []
    for temp in y:
        yy.append(temp[-1])
    y = yy
    
    log.debug("Y Shape 10 out: {}".format(np.shape(yy)))
    x = x[:len(x)-lookahead]
    log.debug("X New Shape: {}".format(np.shape(x)))

    return x,y


def conv_lstm():
    rand_state = 42
    epochs = 40
    batch = 5
    verbose = 1
    knsplits = 6
    shuffle = False # this is a sequential net
    
    def build_mod():
        cnn = Sequential()
        filters = 32
        filter_size = (2,2)
        cnn.add(Reshape((1, 33, 100), input_shape=(None,33, 100)))
        cnn.add(Conv2D(filters,filter_size, activation='relu', padding='same'))
        #cnn.add(MaxPooling2D(pool_size=(3,2)))
        cnn.add(Flatten())
        
        model = Sequential()
        model.add(TimeDistributed(cnn))
        model.add(LSTM(100))
        model.add(Dense(1,activation='relu'))
        model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
        return model
    
    x,y = get_data()
    estimator = KerasClassifier(build_fn=build_mod, epochs=epochs, batch_size=batch, verbose=verbose)
    kfold = KFold(n_splits=knsplits, shuffle=shuffle, random_state=rand_state)
    results = cross_val_score(estimator, x, y, cv=kfold)
    
    return str.format("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))#Nasty return

#look into grid search 
#https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

output = conv_lstm()
print(output)
