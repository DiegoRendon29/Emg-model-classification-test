import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import math
import scipy.stats
import random
from ETL import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam


def load_spectral_data(frame_size):
    file = obtain_spectral_data(frame_size)
    print(file)
    validation_data_list = []
    validation_label_list = []
    training_data_list = []
    training_label_list = []
    i = 0
    with h5py.File(file, "r") as f:
        for key, dataset in f.items():  # Directly iterating over datasets
            if i%1000 == 0:
                print(i)
            i = i+1
            attrs = dataset.attrs  # Store attributes once
            repetition = attrs["repetition"]
            movement = attrs["movement"]
            data = dataset[:]


            if repetition == 5:
                validation_data_list.append(data)
                validation_label_list.append(movement)
            elif repetition not in {3, 5}:
                training_data_list.append(data)
                training_label_list.append(movement)

            # Convert lists to numpy arrays efficiently

        validation_data = np.array(validation_data_list)
        validation_label = np.array(validation_label_list)

        training_data = np.array(training_data_list)
        training_label = np.array(training_label_list)



        # One-hot encoding
        train_labels = tf.keras.utils.to_categorical(training_label)

        val_labels = tf.keras.utils.to_categorical(validation_label)

        # Shuffle data efficiently
        if training_data.size > 0:
            idx = np.random.permutation(len(training_label))
            training_data = training_data[idx]
            train_labels = train_labels[idx]

        if validation_data.size > 0:
            idx = np.random.permutation(len(validation_label))
            validation_data = validation_data[idx]
            val_labels = val_labels[idx]

        print(training_data.shape)
        training_data = training_data.reshape(-1, 48, 55, 1)
        validation_data = validation_data.reshape(-1, 48, 55, 1)

        train_min = training_data.min()
        train_max = training_data.max()
        training_data = (training_data - train_min) / (train_max - train_min)
        validation_data = (validation_data - train_min) / (train_max - train_min)
        print(validation_data.shape)
        print(training_data.shape)
        return (training_data, train_labels), (validation_data, val_labels)

def save_model(model, output_dir,model_name):
    model.save(os.path.join(output_dir, model_name + '.h5'))


def creating_model():

    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),
                            activation='relu', input_shape=(48, 55, 1)))
    model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    #model.add(layers.Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(53, activation='softmax'))
    model.compile(optimizer="SGD",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model

def train_model(model, train_data, train_labels, val_data, val_labels,epoch=800,batch_size=64):
    model.fit(x=train_data, y=train_labels,
              epochs=epoch,
              batch_size=batch_size,
              validation_data=(val_data, val_labels))
def main(frame_size):
    output_dir = "saved_model/"
    model_name = f"Model_spectral_FrameSize{frame_size}"

    # Loading of data
    (train_data, train_labels), (val_data, val_labels) = load_spectral_data(frame_size)


if __name__ == "__main__":
    (train_data, train_labels), (val_data, val_labels) = load_spectral_data(31)

    print(train_data.shape)
    print(train_labels.shape)
    print(val_data.shape)
    print(val_labels.shape)
    model = creating_model()
    train_model(model, train_data, train_labels, val_data, val_labels)