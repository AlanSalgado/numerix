# Importamos las librerías que vamos a usar
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Cargamos el dataset MNIST (imágenes de números escritos a mano del 0 al 9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pasamos los valores de los pixeles de 0-255 a 0-1 para que sea más fácil entrenar la red
x_train, x_test = x_train / 255.0, x_test / 255.0

# Añadimos una dimensión extra porque la red espera imágenes con "canales" (en este caso en blanco y negro)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# ---------------------------
# CREACIÓN DEL MODELO
# ---------------------------
model = models.Sequential([
    # Primera capa: detecta patrones pequeños en la imagen
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    # Reduce la información pero mantiene lo más importante
    layers.MaxPooling2D((2,2)),
    
    # Segunda capa: detecta patrones más complejos
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Prepara los datos en una sola lista para poder conectarlos a la parte final
    layers.Flatten(),

    # Capa intermedia con varias neuronas que ayudan a tomar decisiones
    layers.Dense(128, activation='relu'),
    # Apaga algunas neuronas al azar para evitar que el modelo "memorice"
    layers.Dropout(0.3),

    # Otra capa intermedia
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    # Capa de salida: 10 opciones (los números del 0 al 9)
    layers.Dense(10, activation='softmax')
])

# ---------------------------
# CONFIGURAMOS EL ENTRENAMIENTO
# ---------------------------
model.compile(
    optimizer='adam',                        # Método para aprender
    loss='sparse_categorical_crossentropy',  # Cómo calcula el error
    metrics=['accuracy']                     # Qué mide (qué tan bien acierta)
)

# ---------------------------
# ENTRENAMOS EL MODELO
# ---------------------------
model.fit(
    x_train,
    y_train,
    epochs=20,               # Número de repeticiones del entrenamiento
    validation_data=(x_test, y_test)  # Datos para comprobar cómo va aprendiendo
)
print("modelo entrenado")

# ---------------------------
# GUARDAMOS EL MODELO
# ---------------------------
model.save("trained_model.h5")
