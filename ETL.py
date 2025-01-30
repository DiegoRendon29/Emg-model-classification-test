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

def create_spark_sesion():
    os.environ['PYSPARK_PYTHON'] = "C:/Users/52669/anaconda3/envs/awsEmg/python.exe"
    os.environ['PYSPARK_DRIVER_PYTHON'] = "C:/Users/52669/anaconda3/envs/awsEmg/python.exe"
    os.environ['HADOOP_HOME'] = 'C:/Users/52669/Documents/spark/spark-3.5.4-bin-hadoop3'
    os.environ['PATH'] += ';' + os.path.join(os.environ['HADOOP_HOME'], 'bin')

    # Archivos JAR con las rutas corregidas
    aws_sdk_jar = "file:///C:/Users/52669/Documents/spark/spark-3.5.4-bin-hadoop3/aws-java-sdk-1.12.780.jar"
    hadoop_aws_jar = "file:///C:/Users/52669/Documents/spark/spark-3.5.4-bin-hadoop3/hadoop-aws-3.3.0.jar"
    guava_jar = "file:///C:/Users/52669/Documents/spark/spark-3.5.4-bin-hadoop3/jars/guava-33.4.0-jre.jar"

    spark = SparkSession.builder \
        .appName("MiAplicacionSpark") \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.region", "us-west-2") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
        .getOrCreate()

    print("hasta aqui 2 ")
    data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    columns = ["name", "value"]
    df = spark.createDataFrame(data, columns)
    print("hasta aqui 3")
    # Escribe en S3
    df.write.parquet("s3a://emgninapro/")
    print("jasta")
    return spark


def obtain_complete_data(size_frame=31):
    s3 = boto3.client('s3')
    bucket = "emgninapro"
    spark = spark_sesion()
    output_file = f'data/complete_data/statistical_data_frame{size_frame}.csv'

    try:
        s3.head_object(Bucket=bucket, Key=output_file)
        file_exists = True
    except ClientError:
        file_exists = False

    if file_exists:
        try:
            print("Archivo si existe")
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
        # Leer el archivo CSV existente en S3
        s3_object_existing = s3.get_object(Bucket=bucket, Key=output_file)
        existing_data = pd.read_csv(s3_object_existing['Body'])

        # Concatenar los datos nuevos con los existentes
        df = pd.concat([existing_data, df], ignore_index=True)

    # **Subir el archivo actualizado a S3**
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=output_file, Body=csv_buffer.getvalue())





if __name__ == "__main__":
    #create_spark_sesion()
    #spark_sesion(
    df = obtain_complete_data(size_frame=31)

