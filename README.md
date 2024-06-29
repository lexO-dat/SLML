# SLML - Dispositivo Detector de Lengua de Señas

## Estructura del Proyecto

- **Model**: Contiene los modelos TensorFlow para la interpretación de lengua de señas.
- **Transform**: Scripts para convertir modelos de Keras (.h5) a TensorFlow Lite (.tflite), facilitando el despliegue en dispositivos de bajo consumo.
- **raspberry_app_lite**:
  - `lite_version.py`: Prueba modelos TensorFlow Lite individuales.
  - `multiple_models.py`: Utiliza un modelo general que redirige a modelos específicos basados en la seña detectada.

## Detalles del Script `multiple_models.py`

### Importación de Librerías
Utiliza OpenCV (`opencv-python`), MediaPipe (`mediapipe`), NumPy (`numpy`), y TensorFlow 15.1.0 (`tensorflow`) para procesar imágenes y manejar modelos de aprendizaje automático.

### Configuración y Procesamiento
- **MediaPipe**: Inicializa la detección de manos.
- **TensorFlow Lite**: Carga un modelo general para clasificar señas en categorías.

### Captura y Procesamiento de Video
- Captura video en tiempo real, detecta y analiza manos en cada frame.
- Convierte colores, prepara y normaliza imágenes para el modelo.

### Detección y Clasificación
- Clasifica las señas en categorías y luego determina la letra específica usando modelos dedicados.
- Para la raspberry debe estar comentado lo que es mostrar las imagenes y en vez de usar opencv-python usar opencv-python-headless

### Visualización y Control
- Muestra resultados en una ventana y permite cerrar el proceso con la tecla 'Esc'.
- 
### Instalar dependencias
``` bash
pip install opencv-python mediapipe tensorflow numpy requests
```
### Comando para Ejecutar
```bash
python multiple_models.py
```


## train_and_test

Incluye scripts para testear modelos Keras y capturar imágenes necesarias para entrenar modelos de manera eficiente.

## Propósito General

El proyecto SLML busca desarrollar un dispositivo eficiente y accesible para el reconocimiento de lengua de señas, aprovechando la capacidad de Raspberry Pi y ESP32-CAM, utilizando TensorFlow para asegurar un consumo eficiente de recursos.
