import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
sns.set(style="whitegrid")
# Datos
frame_size = [30, 50, 60, 80, 100, 120]
accuracy = [0.6649, 0.7269, 0.752, 0.7962, 0.8296,0.8469]



# Crear la figura y el eje
plt.figure(figsize=(8, 5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Graficar la línea
sns.lineplot(x=frame_size, y=accuracy, marker='o', markersize=13, color='b', label='Accuracy')

plt.fill_between(frame_size, accuracy, alpha=0.3, color='b')

plt.xlim(28, 122)
plt.ylim(0.0,1)

# Etiquetas y título
plt.xlabel("Frame Size",fontsize=18)
plt.ylabel("Accuracy",fontsize=18)
plt.legend(fontsize=16)

# Mostrar la gráfica
plt.show()




features = ["mean channel 6","mean channel 9","mean channel 1","mean channel 7", "percentile 25 channel 6",
            "mean channel 8", "mean channel 5", "mean channel 2","percentile 25 channel 1","mean channel 0"]

values = [0.003548273037187967,0.0031643484332732974,0.003013642835635103,0.002616443716732535,0.002567866617450359,
          0.002537756278611418, 0.0023908827152394965, 0.002358448773090569,0.0023035990881301416,0.0022394152991050513]

plt.figure(figsize=(10, 6))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.barh(features[::-1],values[::-1],
         color="skyblue")  # Invertimos para que la más importante esté arriba
plt.xlabel("Mean SHAP Value")
plt.ylabel("Feature Name")
plt.show()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
models_name = ["Neural network", "Random forest", "Support vector classification", "K-nn", "Decision tree"]
accs = [ 0.6649,0.2058,0.2058, 0.157,0.157]

plt.figure(figsize=(10, 6))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.bar(models_name,accs,
         color="powderblue")
addlabels(models_name, accs)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()
