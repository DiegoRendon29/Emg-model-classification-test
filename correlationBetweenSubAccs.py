import math
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
sns.set(style="whitegrid")


def calculate_correlation_age(file_name):

    df = pd.read_excel(file_name)
    sub_age = np.array(df["age"])
    sub_accs = np.array(df["accuracy"])

    plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.scatterplot(x=sub_age, y=sub_accs, color='b', label='Accuracy')
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Accuracy", fontsize=13)
    plt.xlim(20, 42)
    plt.ylim(0, 1)
    plt.show()

    correlation = np.corrcoef(sub_age, sub_accs)[0, 1]  # Extraemos el valor de la matriz
    print("Pearson correlation:", correlation)


def calculate_correlation_bmi(file_name):

    df = pd.read_excel(file_name)
    sub_bmi = np.array(df["bmi"])
    sub_accs = np.array(df["accuracy"])

    plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.scatterplot(x=sub_bmi, y=sub_accs, color='b', label='Accuracy')
    plt.xlabel("bmi", fontsize=14)
    plt.ylabel("Accuracy", fontsize=13)
    plt.xlim(17.2, 30)
    plt.ylim(0, 1)
    plt.show()

    correlation = np.corrcoef(sub_bmi, sub_accs)[0, 1]  # Extraemos el valor de la matriz
    print("Pearson correlation bmi-accuracy:", correlation)


if __name__ == "__main__":
    file_n = "Parameters subjects.xlsx"
    calculate_correlation_age(file_n)
    calculate_correlation_bmi(file_n)