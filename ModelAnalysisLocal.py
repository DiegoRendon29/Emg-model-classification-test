import tensorflow as tf
import os
import shap
from TrainingModelLocally import load_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
matplotlib.use("TkAgg")

def read_model(model_nm):
    directory = "saved_model/"
    model_location = os.path.join(directory, model_nm)
    model = tf.keras.models.load_model(model_location)
    return model


def analysis_shap(model, frame_size):
    print(model.input_shape)
    (train_data, train_labels), (val_data, val_labels) = load_data(frame_size=frame_size)
    K = 540  # Reduce el número de muestras, ajusta según necesidad
    sample_k = shap.sample(train_data, K)
    explaining_data = shap.sample(val_data,800)
    explainer = shap.KernelExplainer(model.predict, sample_k)
    shap_values = explainer.shap_values(explaining_data)
    print(shap_values.shape)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
        shap_values = np.mean(np.abs(shap_values), axis=0)
    else:
        shap_values = np.abs(shap_values)
    print(shap_values.shape)
    metrics = [
        "Census", "mean", "std", "kurtosis",
        "skewness", "entropy", "median",
        "percentile 25", "percentile 75"
    ]

    columns = [f"{metric} channel {i}" for i in range(10) for metric in metrics]
    mean_shap_values =shap_values.mean(axis=2).mean(axis=0).flatten()  # Calculamos el promedio por característica
    important_features = np.argsort(mean_shap_values)[::-1]
    print(mean_shap_values.shape)
    print(important_features)
    print(f"Show the {min(90, len(important_features))} more impactful features")
    for idx in important_features[:90]:
        print(f"Feature: {columns[idx]}, Mean SHAP: {mean_shap_values[idx]}")

    top_k = 10
    top_features = important_features[:top_k]
    top_shap_values = mean_shap_values[top_features]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[::-1], top_shap_values[::-1],
             color="skyblue")  # Invertimos para que la más importante esté arriba
    plt.xlabel("Mean SHAP Value")
    plt.ylabel("Feature Name")
    plt.title("Top Features by Mean SHAP Value")
    plt.show()

def creation_confusion(model, frame_size):
    (train_data, train_labels), (val_data, val_labels) = load_data(frame_size=frame_size)
    y_pred = model.predict(val_data)

    y_true = np.argmax(val_labels, axis=1)  # Convertir one-hot a índices de clase
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predictions",fontsize=14)
    plt.ylabel("True values",fontsize=14)

    plt.show()
if __name__ == "__main__":
    model_name = "ModelFrameSize61.h5"
    model = read_model(model_name)
    print(model.input_shape)
    #analysis_shap(model,61)
    creation_confusion(model,31)

