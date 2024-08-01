import pickle  # Module for serializing and deserializing Python objects

import cv2  # OpenCV library for image and video processing
import mediapipe as mp  # MediaPipe library for computer vision tasks
import numpy as np  # Library for numerical operations

# Load the trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start capturing video from the first webcam
cap = cv2.VideoCapture(1)

# Initializing MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Setting up the MediaPipe Hands module in static image mode with a minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map predicted labels to characters
labels_dict = {0: 'A', 1: 'B'}

while True:
    data_list = []  # List to store normalized hand landmarks data
    x_data = []  # List to store x-coordinates of landmarks
    y_data= []  # List to store y-coordinates of landmarks

    # Read a frame from the webcam
    ret, frame = cap.read()

    H, W, _ = frame.shape  # Get the height and width of the frame

    # Convert the frame from BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw on
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the x and y coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_data.append(x)
                y_data.append(y)

            # Normalize the landmarks relative to the minimum x and y values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_list.append(x - min(x_data))
                data_list.append(y - min(y_data))

        # Calculate the bounding box for the detected hand
        x1 = int(min(x_data) * W) - 10
        y1 = int(min(y_data) * H) - 10
        x2 = int(max(x_data) * W) - 10
        y2 = int(max(y_data) * H) - 10

        # Predict the character for the detected hand gesture
        prediction = model.predict([np.asarray(data_list)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw the bounding box and the predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
