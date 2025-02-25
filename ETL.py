import time
import h5py

import boto3
import json
import math
from emgObject import Movement
import scipy.io
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
import os
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError
import os
from pyspark.conf import SparkConf
pd.DataFrame.iteritems = pd.DataFrame.items

def spark_sesion():
    os.environ['PYSPARK_PYTHON'] = "C:/Users/52669/anaconda3/envs/awsEmg/python.exe"
    os.environ['PYSPARK_DRIVER_PYTHON'] = "C:/Users/52669/anaconda3/envs/awsEmg/python.exe"
    spark = SparkSession.builder.appName("PipelineExample").getOrCreate()
    return spark


def obtain_complete_data(size_frame=31):
    s3 = boto3.client('s3')
    bucket = "emgninapro"
    #spark = spark_sesion()
    output_file = f'data/complete_data/statistical_data_frame{size_frame}.csv'

    try:
        s3.head_object(Bucket=bucket, Key=output_file)
        file_exists = True
    except ClientError:
        file_exists = False

    if file_exists:
        try:
            print("File exist")
            s3_object_existing = s3.get_object(Bucket=bucket, Key=output_file)
            existing_data = pd.read_csv(s3_object_existing['Body'])
            print(type(existing_data))
            #existing_data = spark.createDataFrame(existing_data)
            return existing_data
        except ClientError as e:
            print(f"Error al obtener el archivo: {e}")
            return None
    else:
        print(f"El archivo {output_file} no existe.")
        print("creating data...")
        create_data(size_frame=size_frame)
        return obtain_complete_data(size_frame)


def create_data(size_frame):
    s3 = boto3.client('s3')
    bucket = "emgninapro"
    prefix_raw = "data/RawData/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix_raw)
    output_file = f'data/complete_data/statistical_data_frame{size_frame}.csv'
    metrics = [
        "Census", "mean", "std", "kurtosis",
        "skewness", "entropy", "median",
        "percentile 25", "percentile 75"
    ]

    columns = [f"{metric} channel {i}" for i in range(10) for metric in metrics]

    columns.append("subject")
    columns.append("repetition")
    columns.append("movement")
    df = pd.DataFrame(columns=columns)  # Inicializamos el DataFrame vacío

    if 'Contents' in response:
        print(f'Files in folder "{prefix_raw}":')

        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.mat'):
                print(f'File : - {obj["Key"]} (Size: {obj["Size"]} bytes)')

                # Leer el archivo desde S3
                file_object = s3.get_object(Bucket=bucket, Key=key)
                file_data = file_object['Body'].read()
                file_stream = io.BytesIO(file_data)
                df_temp = scipy.io.loadmat(file_stream, appendmat=False)

                emg = df_temp['emg']
                stimulus = df_temp['restimulus']
                stimulus_index = np.where(stimulus != 0)[0]
                ends = np.where(np.diff(stimulus_index) != 1)[0]
                starts = np.insert(ends + 1, 0, 1)
                ends = np.append(ends, len(stimulus_index) - 1)


                temp_list = []  # Lista para almacenar datos temporalmente antes de convertir a DataFrame
                for i in range(len(starts)):
                    mov_data = Movement(
                        emg=emg[stimulus_index[starts[i]]:stimulus_index[ends[i]], :],
                        subject=df_temp['subject'],
                        exercise=df_temp['exercise'],
                        movement=stimulus[stimulus_index[starts[i]]]
                    )
                    data_extracted = mov_data.create_parameters(size=size_frame, hop=20)

                    temp_list.extend(
                        [window + [mov_data.subject, i % 10, mov_data.totalMov] for window in data_extracted]
                    )

                # Convertimos los datos acumulados a DataFrame y los agregamos a `df`
                if temp_list:
                    df = pd.concat([df, pd.DataFrame(temp_list, columns=columns)], ignore_index=True)

    # **Guardar el DataFrame completo una sola vez**
    try:
        s3.head_object(Bucket=bucket, Key=output_file)
        file_exists = True
    except s3.exceptions.ClientError:
        file_exists = False

    if file_exists:

        s3_object_existing = s3.get_object(Bucket=bucket, Key=output_file)
        existing_data = pd.read_csv(s3_object_existing['Body'])

        df = pd.concat([existing_data, df], ignore_index=True)


    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=output_file, Body=csv_buffer.getvalue())


def create_spectral_data(size_frame):
    s3 = boto3.client('s3')
    bucket = "emgninapro"
    prefix_raw = "data/RawData/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix_raw)
    output_file = f'data/complete_data/spectral_data_frame{size_frame}.h5'
    mtx = 0
    if 'Contents' in response:
        print(f'Files in folder "{prefix_raw}":')

        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.mat'):
                print(f'File : - {obj["Key"]} (Size: {obj["Size"]} bytes)')

                # Leer el archivo desde S3
                file_object = s3.get_object(Bucket=bucket, Key=key)
                file_data = file_object['Body'].read()
                file_stream = io.BytesIO(file_data)
                df_temp = scipy.io.loadmat(file_stream, appendmat=False)

                emg = df_temp['emg']
                stimulus = df_temp['restimulus']
                stimulus_index = np.where(stimulus != 0)[0]
                ends = np.where(np.diff(stimulus_index) != 1)[0]
                starts = np.insert(ends + 1, 0, 1)
                ends = np.append(ends, len(stimulus_index) - 1)

                for i in range(len(starts)):
                    mov_data = Movement(
                        emg=emg[stimulus_index[starts[i]]:stimulus_index[ends[i]], :],
                        subject=df_temp['subject'],
                        exercise=df_temp['exercise'],
                        movement=stimulus[stimulus_index[starts[i]]]
                    )
                    data_extracted = mov_data.create_windowsSpect(size=size_frame,hop=20)

                    with h5py.File(output_file, "a") as f:
                        for spect in data_extracted:

                            dt_spect = f.create_dataset(f"matriz{mtx}", data=spect)
                            dt_spect.attrs["subject"] = mov_data.subject
                            dt_spect.attrs["repetition"] = i % 10
                            dt_spect.attrs["movement"] = mov_data.totalMov
                            mtx = mtx + 1

    print(spect.shape)
    s3.upload_file(output_file, bucket, output_file)


def obtain_spectral_data(size_frame):
    s3 = boto3.client('s3')
    bucket = "emgninapro"

    output_file = f'data/complete_data/spectral_data_frame{size_frame}.h5'

    if os.path.exists(output_file):
        print("File Alredy locally, returning location.")
    else:
        print("File do not exist locally, trying to download.")
        try:

            s3.head_object(Bucket=bucket, Key=output_file)
            print("File exist in s3")
            file_exists = True
        except ClientError:
            file_exists = False
        if file_exists:
            try:
                #s3_object_existing = s3.get_object(Bucket=bucket, Key=output_file)
                #existing_data = pd.read_csv(s3_object_existing['Body'])
                print("downloading...")
                s3.download_file(bucket, output_file, output_file)
                print(f"File donwloaded in:{output_file}")
                return output_file
            except ClientError as e:
                print(f"Error: {e}")
                return None
        else:

            print("File do not exist in s3, creating it...")
            create_spectral_data(size_frame=size_frame)
            return obtain_complete_data(size_frame)
    return output_file



if __name__ == "__main__":
    #create_spark_sesion()
    #spark_sesion(
    create_spectral_data(size_frame=31)

