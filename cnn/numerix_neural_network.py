# Importar librerías
# tensorflow: crear y entrenar la red
# layers: crear capas
# models: crear modelos
# numpy: manejar arreglos
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# cargar dataset MNIST con dígitos del 0-9
# x_train: imágenes de entrenamiento
# y_train: etiquetas (qué número es cada imagen)
# x_test y y_test: datos de prueba
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar los pixeles para que pasen de 0-255 a 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# agregar dimensión extra para que sean (28, 28, 1), no (28, 28)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Creación del modelo
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),     # Capa de entrada con 32 neuronas
    layers.MaxPooling2D((2,2)),     # Reducir dimesión de la imagen
    layers.Conv2D(64, (3,3), activation='relu'),    # Primer capa intermedia con 64 neuronas
    layers.Flatten(),
    layers.Dense(64, activation='relu'),            # Segunda capa intermedia con 64 neuronas
    layers.Dense(10, activation='softmax')          # Capa de salida con 10 neuronas (1 por cada dígito)
])

# compilar modelo de la red. 
# adam es el algoritmo que ajusta los pesos de la red
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# entrenar el modelo con 5 vueltas completas al dataset
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("Modelo entrenado!")

# Guardar modelo entrenado
model.save("cnn/trained_model.h5")