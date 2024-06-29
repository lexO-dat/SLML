import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

#Se inicializa mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Carga el modelo general a tensorflow
interpreter_groups = tf.lite.Interpreter(model_path="../Model/GENERAL_MODEL/GENERAL_MODEL.tflite")    
interpreter_groups.allocate_tensors()
input_details_groups = interpreter_groups.get_input_details()
output_details_groups = interpreter_groups.get_output_details()

#Setea los parametros de la deteccion de la mano
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

#Inicia la captura de vide (en la raspberry es la ip de la esp32-cam)
cap = cv2.VideoCapture(0)

#asigna los grupos para el modelo general
labels_groups = ["CO", "HG", "AET", "IJY", "BDKLU"]

#mapea las letras para usarlas con el modelo especifico
specific_labels_dict = {
    "CO": ["c", "o"],
    "HG": ["h", "g"],
    "AET": ["a", "e", "t"],
    "IJY": ["i", "j", "y"],
    "BDKLU": ["b", "d", "k", "l", "u"]
}

while cap.isOpened():

    #inicia la captura de imagenes
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #Convierte la imagen capturada de BGR a RGB ya que opencv la captura en BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    #Se genra una imagen con el fondo de color blanco (255,255,255) para asi poner el esqueleto de la mano encima
    image_white = np.ones(image.shape, dtype=np.uint8) * 255

    #procesa la imagen y genera el esqueleto de distintos colores
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #dibuja el esqueleto
            mp_drawing.draw_landmarks(
                image_white,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            #prepara la imagen para el modelo
            img_resize = cv2.resize(image_white, (224, 224))  #recortandola a 224x224
            input_data = np.expand_dims(img_resize.astype(np.float32) / 255.0, axis=0)  #normaliza la imagen
            interpreter_groups.set_tensor(input_details_groups[0]['index'], input_data)
            interpreter_groups.invoke()
            output_data = interpreter_groups.get_tensor(output_details_groups[0]['index'])
            group_index = np.argmax(output_data) #obtiene la prediccion viendo que prediccion tiene la confianza mas alta
            group_label = labels_groups[group_index] #busca ese grupo en el labels para derivarlo al modelo especifico
            #print("Grupo detectado:", group_label)
            
            #se llama al modelo especifico
            path_to_specific_model = f"../Model/{group_label}/{group_label}.tflite"
            interpreter_specific = tf.lite.Interpreter(model_path=path_to_specific_model)
            interpreter_specific.allocate_tensors()
            input_details_specific = interpreter_specific.get_input_details()
            output_details_specific = interpreter_specific.get_output_details()       
            interpreter_specific.set_tensor(input_details_specific[0]['index'], input_data)
            interpreter_specific.invoke()
            output_data_specific = interpreter_specific.get_tensor(output_details_specific[0]['index'])
            #se extraen los labels conseguidos con el modelo general y se usa la prediccion mas alta
            letter_index = np.argmax(output_data_specific)
            specific_labels = specific_labels_dict[group_label]
            #devuelve la letra detectada
            current_letter = specific_labels[letter_index]

            print("Letra predecida:", current_letter)

    # Display the result.
    cv2.imshow('MediaPipe Hands', image_white)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up.
hands.close()
cap.release()
cv2.destroyAllWindows()







                

