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
            tf.keras.layers.Dense(8, activation=tf.nn.relu),
            
            tf.keras.layers.Dense(8, activation=tf.nn.relu),
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
from keras import regularizers
#one-hotエンコーディング
y_train_one = keras.utils.to_categorical(y_train, 14)
y_test_one = keras.utils.to_categorical(y_test, 14)
#データの正規化
X_train_np = np.array(X_train)
X_mean = np.mean(X_train_np, axis=0)
X_std = np.std(X_train_np, axis=0)
X_train = pd.DataFrame((X_train_np - X_mean) / X_std)
#データのシャッフル
rand = np.random.permutation(len(X_train))
X_train = X_train.iloc[rand]
y_train = y_train[rand]
#検証データの作成
partial_x_train = X_train[10000:]
x_val = X_train[:10000]
partial_y_train = y_train_one[10000:]
y_val = y_train_one[:10000]
#モデル構築
model = Sequential()
model.add(Dense(8, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(11,)))
model.add(Dropout(0.5))
model.add(Dense(8, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(8, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))    
model.summary()

epochs = 100
batch_size = 128
lr = 0.0003
model.compile(optimizer=RMSprop(lr=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val))
score = model.evaluate(X_test, y_test_one, verbose=0)
#スコア表示
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#訓練データと検証データの損失値をプロット
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#訓練データと検証データでの正答率をプロット
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()