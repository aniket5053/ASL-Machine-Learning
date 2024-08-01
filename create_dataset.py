import os  # Module for interacting with the operating system
import pickle  # Module for serializing and deserializing Python objects

import mediapipe as mp  # MediaPipe library for computer vision tasks
import cv2  # OpenCV library for image and video processing
import matplotlib.pyplot as plt  # Matplotlib library for plotting

# Initializing MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Setting up the MediaPipe Hands module in static image mode with a minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the dataset
DATA_DIR = './data'

# Print the list of directories/files in the dataset directory
print(os.listdir(DATA_DIR))

data = []  # List to store hand landmarks data
labels = []  # List to store corresponding labels

# Loop through each directory (class) in the dataset directory
for directory in os.listdir(DATA_DIR):
    # Loop through each image file in the current class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, directory)):
        data_list = []  # List to store landmarks for a single image

        x_data = []  # List to store x-coordinates of landmarks
        y_data = []  # List to store y-coordinates of landmarks

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, directory, img_path))
        # Convert the image from BGR to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
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

            # Append the processed landmarks and label to the data and labels lists
            data.append(data_list)
            labels.append(directory)

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
