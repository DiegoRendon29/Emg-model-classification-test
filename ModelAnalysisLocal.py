import tensorflow as tf
import os
import shap
from TrainingModelLocally import load_data


def read_model(model_nm):
    directory = "saved_model/"
    model_location = os.path.join(directory, model_nm)
    model = tf.keras.models.load_model(model_location)
    return model


def analysis_shap(model, frame_size):
    (train_data, train_labels), (val_data, val_labels) = load_data(frame_size=frame_size)
    explainer = shap.KernelExplainer(model.predict, train_data)
    shap_values = explainer.shap_values(val_data)


    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
        shap_values = np.mean(np.abs(shap_values), axis=0)
    else:
        shap_values = np.abs(shap_values)
    metrics = [
        "Census", "mean", "std", "kurtosis",
        "skewness", "entropy", "median",
        "percentile 25", "percentile 75"
    ]

    columns = [f"{metric} channel {i}" for i in range(10) for metric in metrics]
    mean_shap_values = shap_values.mean(axis=1).flatten()  # Calculamos el promedio por característica
    important_features = np.argsort(mean_shap_values)[::-1]

    print(f"Show the {min(10, len(important_features))} more impactful features")
    for idx in important_features[:10]:
        print(f"Feature: {columns[idx]}, Mean SHAP: {mean_shap_values[idx]}")


if __name__ == "__main__":
    model_name = "ModelFrameSize31.h5"
    model = read_model(model_name)
    analysis_shap(model,31)