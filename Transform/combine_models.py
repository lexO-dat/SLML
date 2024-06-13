from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Concatenate, Input

# Cargar los modelos
modelo1 = load_model('../Model/model_co.h5')
modelo2 = load_model('../Model/model_hg.h5')

# Crear nuevas capas de entrada para cada modelo
input_modelo1 = Input(shape=(224, 224, 3), name='input_modelo1')
input_modelo2 = Input(shape=(224, 224, 3), name='input_modelo2')

# Obtener las capas de salida de cada modelo (las últimas capas)
output_modelo1 = modelo1.layers[-2].output  # Capa previa a la última capa de salida
output_modelo2 = modelo2.layers[-2].output  # Capa previa a la última capa de salida

# Fusionar las capas de salida de ambos modelos
fusion_output = Concatenate()([output_modelo1, output_modelo2])

# Crear un nuevo modelo combinado
modelo_combinado = Model(inputs=[input_modelo1, input_modelo2], outputs=fusion_output)

# Ver la arquitectura del modelo combinado
modelo_combinado.summary()
