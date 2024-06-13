import tensorflow as tf
import os

# Obtener la ruta al archivo keras_model.h5
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "BDFKLUW.h5")

# Cargar el modelo Keras
model = tf.keras.models.load_model(model_path)

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TensorFlow Lite
output_path = os.path.join(current_dir, "BDFKLUW.tflite")
with open(output_path, "wb") as f:
    f.write(tflite_model)

print("Modelo convertido y guardado en:", output_path)
