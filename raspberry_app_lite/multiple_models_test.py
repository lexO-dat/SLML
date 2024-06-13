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

# Load the model from the repository
model = tf.keras.models.load_model("path/to/your/downloaded/model.h5")

esp32_url = "http://192.168.174.119/240x240.jpg"
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224
word = ""
prev_letter = None
previous_letter = None
labels = ["A", "B", "H", "O", "L"]  # Adjust based on the new model's output

while True:
    # Enable for the Raspberry Pi
    # cap.open(esp32_url)

    success, img = cap.read()

    if success:
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                input_data = np.expand_dims(imgWhite.astype(np.float32) / 255.0, axis=0)

                predictions = model.predict(input_data)
                index = np.argmax(predictions)
                current_letter = labels[index]
                
                # Confidence of the prediction
                confidence = np.max(predictions)

                if current_letter == previous_letter:
                    counter += 1
                else:
                    previous_letter = current_letter
                    counter = 1

                if counter >= 5 and current_letter != prev_letter:
                    prev_letter = current_letter
                    print("LA LETRA ES:", current_letter, "CON UNA CONFIANZA DE:", confidence)
                    if current_letter == "next":
                        word = ""
                    else:
                        word = word  # + current_letter

                    enviar_letra(current_letter)
                    letter_time = time.time()
                    counter = 0

            else:
                print("imgCrop has invalid dimensions:", imgCrop.shape)

        cv2.imshow("Image", imgOutput)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Error al leer el fotograma de la cámara.")
        break

cap.release()
cv2.destroyAllWindows()
