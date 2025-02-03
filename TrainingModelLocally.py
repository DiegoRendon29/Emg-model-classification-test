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


def load_data(frame_size, input_dir=None, df=None):
    # Function to create and clean the data, it can direct extract from a dataframe or create one
    if df is None:
        file = f"statistical_data_frame{frame_size}.csv"
        try:
            # Read data
            df = pd.read_csv(os.path.join(input_dir, file))
            print(df)
        except:
            print(f"File does not exist, creating a new one.")
            df = obtain_complete_data(frame_size)

    print("Loaded data")
    print("Creating dataframe")
    df.drop(columns="subject", inplace=True)
    df.fillna(0, inplace=True)

    df_validation = df[df['repetition'] == 5]
    df_train = df[(df['repetition'] != 5) & (df['repetition'] != 3)]

    df_validation.drop(columns="repetition", inplace=True)
    df_train.drop(columns="repetition", inplace=True)

    train_label = df_train.pop("movement")
    validation_label = df_validation.pop("movement")

    scaler = MinMaxScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
    df_validation = pd.DataFrame(scaler.transform(df_validation), columns=df_validation.columns)

    train_data = df_train.values
    val_data = df_validation.values

    train_labels = tf.keras.utils.to_categorical(train_label)
    val_labels = tf.keras.utils.to_categorical(validation_label)

    idx = np.random.permutation(len(train_data))
    train_data, train_labels = train_data[idx], train_labels[idx]

    idx = np.random.permutation(len(val_data))
    val_data, val_labels = val_data[idx], val_labels[idx]

    return (train_data, train_labels), (val_data, val_labels)


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(53, activation='softmax'))
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def train_model(model, train_data, train_labels, val_data, val_labels):
    model.fit(x=train_data, y=train_labels,
              epochs=800,
              batch_size=64,
              validation_data=(val_data, val_labels))


def save_model(model, output_dir,model_name):
    model.save(os.path.join(output_dir, model_name + '.h5'))


def main(df=None, input_dir=None,frame_size=81):
    output_dir = "saved_model/"
    model_name = f"ModelFrameSize{frame_size}"

    # Loading of data
    (train_data, train_labels), (val_data, val_labels) = load_data(input_dir=input_dir, df=df,frame_size=frame_size)

    # Model creation
    model = create_model()
    train_model(model, train_data, train_labels, val_data, val_labels)
    save_model(model, output_dir,model_name)
    return model

if __name__ == "__main__":
    model = main(frame_size=101)


