import cv2
import mediapipe as mp
import numpy as np
import os

#Se inicializa la libreria hands de mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Se configura la detección de manos con una complejidad de modelo baja y umbrales de confianza
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

#Se inicializa la captura de video (0 es la cámara predeterminada del PC)
cap = cv2.VideoCapture(0)

#Se crea una carpeta para guardar las imágenes
save_folder = "../data/I"
os.makedirs(save_folder, exist_ok=True)  # Crea la carpeta si no existe

frame_counter = 0 

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #convertidor de color de BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    #img en blanco
    image_white = np.ones(image.shape, dtype=np.uint8) * 255
    image_white = cv2.cvtColor(image_white, cv2.COLOR_RGB2BGR)

    #pega el rig de la imagen en blanco
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_white,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    #Guardar la imagen en la carpeta especificada
    image_path = os.path.join(save_folder, f'hand_landmarks_{frame_counter}.jpg')
    cv2.imwrite(image_path, image_white)
    print(f"Imagen guardada: {image_path}")
    frame_counter += 1

    #Muestra en vivo la imagen resultante
    cv2.imshow('MediaPipe Hands', image_white)
    if cv2.waitKey(5) & 0xFF == 27:
        break

#Cierra cosas y limpia buffer
hands.close()
cap.release()
cv2.destroyAllWindows()
