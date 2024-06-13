import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Cargando el modelo para determinar el grupo de letras
interpreter_groups = tf.lite.Interpreter(model_path="../Model/GENERAL_MODEL.tflite")
interpreter_groups.allocate_tensors()
input_details_groups = interpreter_groups.get_input_details()
output_details_groups = interpreter_groups.get_output_details()

#Setup for hand detection.
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

#Video capture initialization.
cap = cv2.VideoCapture(0)

#Etiquetas para los grupos de letras
labels_groups = ["CO", "HG", "AST", "IJY"]

specific_labels_dict = {
    "CO": ["c", "o"],
    "HG": ["h", "g"],
    "AST": ["a", "s", "t"],
    "IJY": ["i", "j", "y"]
}

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # Prepare a white image for drawing.
    image_white = np.ones(image.shape, dtype=np.uint8) * 255

    # Process each hand landmark.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks.
            mp_drawing.draw_landmarks(
                image_white,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Prepare the image for the model.
            img_resize = cv2.resize(image_white, (224, 224))  # Resize if necessary (already 224x224)
            input_data = np.expand_dims(img_resize.astype(np.float32) / 255.0, axis=0)  # Normalize the image
            interpreter_groups.set_tensor(input_details_groups[0]['index'], input_data)
            interpreter_groups.invoke()
            output_data = interpreter_groups.get_tensor(output_details_groups[0]['index'])
            group_index = np.argmax(output_data)
            group_label = labels_groups[group_index]
            #print("Grupo detectado:", group_label)
            
            
            path_to_specific_model = f"../Model/{group_label}.tflite"
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

            print("Predicted Letter:", current_letter)

    # Display the result.
    cv2.imshow('MediaPipe Hands', image_white)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up.
hands.close()
cap.release()
cv2.destroyAllWindows()







                

