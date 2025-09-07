import numpy as np
import tensorflow as tf
from PIL import Image
import io, base64

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Cargar el modelo cuando inicie el servidor
model = tf.keras.models.load_model("trained_model.h5")

class PredictDigit(APIView):
    def post(self, request):
        try:
            # Recibir la imagen del front
            data = request.data.get("image")
            if not data:
                return Response({"error": "No se envió la imagen"}, status=status.HTTP_400_BAD_REQUEST)

            # Obtener la imagen en base 64 y con escala de grises
            image_data = base64.b64decode(data.split(",")[1])
            image = Image.open(io.BytesIO(image_data)).convert("L")

            # Redimensionar y convertir a arreglo
            image = image.resize((28, 28))
            img_array = np.array(image).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=(0, -1))

            # Predicción
            pred = model.predict(img_array)
            digit = int(np.argmax(pred))

            return Response({"prediction": digit})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)