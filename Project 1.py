# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:45:45 2022

Git Hub Project 1: Predicting Heart Disease

@author: mrob
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks
import pandas as pd
import numpy as np
#%%
#2. Data Loading and Preparation

#2.1 Data loading and check for missing values
data_path= r"C:\Users\captc\Desktop\AI_07\TensorFlow\Datasets_Online_Download\Heart Disease\heart.csv"
df= pd.read_csv(data_path)
#%%
#Check for missing values in the dataset
print(df.isna().sum())
#%%
#Name the features as hd(heart disease)_features and label as hd_label
#2.2 Split data into hd_features and hd_labels

heart_features = df.copy()
heart_label = heart_features.pop('target')
#%%
#Check the features and label
print("Heart Disease Features:")
print(heart_features.head())
print("Heart Disease Label:")
print(heart_label.head())
#%%
#One hot encode label
heart_disease_label_OH = pd.get_dummies(heart_label)
#Check the one-hot label
print("--------------------One-hot Label-----------------")
print(heart_disease_label_OH.head())
#%%
#2.3 Split data into train,test and validation dataset

#Import sklearn train_test split
from sklearn.model_selection import train_test_split

# Let's say we want to split the data in 80:10:10 for train:valid:test dataset

train_size=0.8
SEED=2468

# In the first step we will split the data in training and remaining dataset

X_train, X_rem, y_train, y_rem = train_test_split(heart_features,heart_label, train_size=0.8, random_state= SEED)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# We have to define valid_size=0.5 (that is 50% of remaining data which is X_rem and y_rem)

test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=SEED)
#%%
# Check the shape of each datasets that we have created in the above steps

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)
#%%

#2.4 Execute data normalization by applying standard scaler method.
    #Import StandarScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
#%%
#3. Create a Feed Forward Network Model

nIn= X_train.shape[-1]
nOut= y_train.shape[-1]
model= tf.keras.Sequential()

model.add(layers.InputLayer(input_shape= nIn))

#model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(nOut, activation= 'softmax'))
#%%
#3.1 Compile the model

optimizer='adam'
loss= losses.SparseCategoricalCrossentropy()

model.compile(optimizer, loss, metrics=['accuracy'])

model.summary()
#%%
#4. Define callback functions; earlystopping
from tensorflow.keras.callbacks import EarlyStopping

es_callback = tf.keras.callbacks.EarlyStopping(monitor= 'loss', patience= 1)
#%%
#5. Train the model

EPOCHS = 100
BATCH_SIZE = 64

history = model.fit(X_train,y_train,validation_data=(X_valid,y_valid), batch_size= BATCH_SIZE, epochs= EPOCHS, callbacks=[es_callback])
#%%
#6. Visualize the graph

import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis,training_loss,label='Training Loss')
plt.plot(epochs_x_axis,val_loss,label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis,training_acc,label='Training Accuracy')
plt.plot(epochs_x_axis,val_acc,label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()

plt.show()
#%%
test_result = model.evaluate(X_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")












