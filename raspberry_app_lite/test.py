import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import requests

def enviar_letra(letra):
    url = "https://tics-api.onrender.com/letra"
    payload = {'letra': letra}
    try:
        requests.put(url, json=payload)
    except requests.exceptions.RequestException as e:
        print("Error al enviar la letra:", e)

# Cargando el modelo para determinar el grupo de letras
interpreter_groups = tf.lite.Interpreter(model_path="../Model/GENERAL_MODEL.tflite")
interpreter_groups.allocate_tensors()
input_details_groups = interpreter_groups.get_input_details()
output_details_groups = interpreter_groups.get_output_details()

# Etiquetas para los grupos de letras
labels_groups = ["hg", "co"]

specific_labels_dict = {
    "co": ["c", "o"],
    "hg": ["g", "h"]
}

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Detector de manos
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224
word = ""
prev_letter = None
previous_letter = None
counter = 0

while True:
    success, img = cap.read()
    if success:
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                input_data = np.expand_dims(imgWhite.astype(np.float32) / 255.0, axis=0)
                interpreter_groups.set_tensor(input_details_groups[0]['index'], input_data)
                interpreter_groups.invoke()
                output_data = interpreter_groups.get_tensor(output_details_groups[0]['index'])
                group_index = np.argmax(output_data)
                group_label = labels_groups[group_index]
                #print("Grupo detectado:", group_label)

                path_to_specific_model = f"../Model/model_{group_label}.tflite"
                interpreter_specific = tf.lite.Interpreter(model_path=path_to_specific_model)
                interpreter_specific.allocate_tensors()
                input_details_specific = interpreter_specific.get_input_details()
                output_details_specific = interpreter_specific.get_output_details()       
                interpreter_specific.set_tensor(input_details_specific[0]['index'], input_data)
                interpreter_specific.invoke()
                output_data_specific = interpreter_specific.get_tensor(output_details_specific[0]['index'])
                letter_index = np.argmax(output_data_specific)
                specific_labels = specific_labels_dict[group_label]
                current_letter = specific_labels[letter_index]

                if current_letter == previous_letter:
                    counter += 1
                else:
                    previous_letter = current_letter
                    counter = 1

                if counter >= 5 and current_letter != prev_letter:
                    prev_letter = current_letter
                    print(f"LA LETRA ES: {current_letter}")

            else:
                print("imgCrop has invalid dimensions:", imgCrop.shape)
    else:
        print("Error al leer el fotograma de la cámara.")
        break

cap.release()
