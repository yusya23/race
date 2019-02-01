# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 23:41:26 2019

@author: u5326
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def horse_racing_NN(X_train, X_test, y_train, y_test):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(4, activation=tf.nn.relu),
            tf.keras.layers.Dense(14, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  mtrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10)
    
    pred = model.predict(X_test).argmax(axis=1)
    print((pred == y_test).sum()/ len(pred))
    
def horse_racing_NN2(X_train, X_test, y_train, y_test):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(14, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  mtrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=100)
    
    pred = model.predict(X_test).argmax(axis=1)
    print((pred == y_test).sum()/ len(pred))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_class = 14
epochs = 100
y_train_one = keras.utils.to_categorical(y_train, num_class)
y_test_one = keras.utils.to_categorical(y_test, num_class)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))

model.summary()
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train_one,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test_one))
score = model.evaluate(X_test, y_test_one, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])