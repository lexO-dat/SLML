import tensorflow as tf

# Cargar el modelo entrenado en TensorFlow
model = tf.keras.models.load_model('keras_model.h5')

# Convertir el modelo a formato TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Guardar el modelo cuantizado
with open('../Model/quantized_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Modelo cuantizado guardado exitosamente.")
