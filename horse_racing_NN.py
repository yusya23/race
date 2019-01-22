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
    
    model.fit(X_train, y_train, epochs=100)
    
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