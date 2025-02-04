import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Datos
frame_size = [30, 50, 60, 80, 100]
accuracy = [0.6649, 0.7269, 0.752, 0.7962, 0.8296]

# Configurar estilo de Seaborn
sns.set(style="whitegrid")

# Crear la figura y el eje
plt.figure(figsize=(8, 5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Graficar la línea
sns.lineplot(x=frame_size, y=accuracy, marker='o', markersize=13, color='b', label='Accuracy')

# Agregar el área sombreada con transparencia
plt.fill_between(frame_size, accuracy, alpha=0.3, color='b')

plt.xlim(28, 102)  # Límites del eje X
plt.ylim(0.0,1)  # Límites del eje Y

# Etiquetas y título
plt.xlabel("Frame Size",fontsize=18)
plt.ylabel("Accuracy",fontsize=18)
plt.legend(fontsize=16)

# Mostrar la gráfica
plt.show()
