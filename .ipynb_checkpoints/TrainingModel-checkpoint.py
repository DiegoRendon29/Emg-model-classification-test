import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
#import librosa
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import math
import scipy.stats
import random
#from ETL import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_data(input_dir):
    file = "statistical_data_frame31.csv"
    try:
        # Intentar leer el archivo CSV desde el directorio de entrada
        #df = obtain_complete_data(size_frame=31)
        #df = pd.read_csv(os.path.join(input_dir, file))
        print(input_dir)
        df = pd.read_csv(input_dir)
    except FileNotFoundError:
        print(f"Archivo {file} no encontrado. Obteniendo datos completos.")
        #df = obtain_complete_data(size_frame=31)
    except Exception as e:
        # Capturar cualquier otra excepción y mostrar el error
        print(f"Error al leer el archivo {file}: {e}")
        df = None  # O manejar el error de otra manera
    print("cargado")
    df.drop(columns="subject", inplace=True)
    df.fillna(0, inplace=True)
    print(df.shape)

    df_validation = df[df['repetition'] == 5]
    df_train = df[(df['repetition'] != 5) & (df['repetition'] != 3)]

    df_validation.drop(columns="repetition", inplace=True)
    df_train.drop(columns="repetition", inplace=True)

    train_label = df_train.pop("movement")
    validation_label = df_validation.pop("movement")
    print(validation_label.unique())

    scaler = MinMaxScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
    df_validation = pd.DataFrame(scaler.transform(df_validation), columns=df_validation.columns)

    train_data = df_train.values
    val_data = df_validation.values

    train_labels = tf.keras.utils.to_categorical(train_label)
    val_labels = tf.keras.utils.to_categorical(validation_label)
    print(train_labels.shape)
    return (train_data, train_labels), (val_data, val_labels)


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(2048, activation='tanh'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(1024, activation='tanh'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(512, activation='tanh'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(256, activation='tanh'))
    model.add(layers.Dense(53, activation='softmax'))
    model.compile(optimizer='SGD',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def train_model(model, train_data, train_labels, val_data, val_labels):
    model.fit(x=train_data, y=train_labels,
              epochs=2000,
              batch_size=64,
              validation_data=(val_data, val_labels))

def save_model(model, output_dir):
    model.save(os.path.join(output_dir, 'model.h5'))
    
def main():
    # Directorios proporcionados por SageMaker
    input_dir = "s3://emgninapro/data/complete_data/statistical_data_frame31.csv"  # Ruta a los datos de entrenamiento y validación
    output_dir = "s3://emgninapro/models/"  # Ruta donde se guardará el modelo entrenado

    # Cargar datos de entrenamiento y validación
    (train_data, train_labels), (val_data, val_labels) = load_data(input_dir)

    # Crear y entrenar el modelo
    model = create_model()
    train_model(model, train_data, train_labels, val_data, val_labels)

    # Guardar el modelo entrenado en el directorio de salida
    #save_model(model, output_dir)
    save_model(model, output_dir)



if __name__ == "__main__":
    input_dir = "s3://emgninapro/data/complete_data/"
    main()
