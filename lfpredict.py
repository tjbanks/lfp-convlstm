# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 19:26:16 2018

@author: TJBANKS
References:
    https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
    https://keras.io/layers/wrappers/
    https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
"""
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,TimeDistributed,Flatten,LSTM,Dense

def build_model():
    cnn = Sequential()
    cnn.add(Conv2D(1,(2,2), activation='relu', padding='same', input_shape=(10,10,1)))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Flatten())
    
    model = Sequential()
    model.add(TimeDistributed(cnn))
    model.add(LSTM())
    model.add(Dense())
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model


model = build_model()
batch_size=30
epochs=2

print('Training...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)