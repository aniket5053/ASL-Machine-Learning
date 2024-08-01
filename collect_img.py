import os  # Module for interacting with the operating system
import cv2  # OpenCV library for image and video processing
import mediapipe as mp  # MediaPipe library for computer vision tasks

# Initializing MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Setting up the MediaPipe Hands module in static image mode with a minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory to save the dataset
DATA_DIR = './data'

# Create the dataset directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3  # Number of gesture classes
dataset_size = 300  # Number of images per class

cap = cv2.VideoCapture(1)  # Start capturing video from the first webcam

# Loop through each gesture class
for j in range(number_of_classes):
    # Create a directory for the current gesture class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Display the frame with instruction to press 'Q' to start collecting data
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press Q', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collecting the dataset images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Save the current frame as an image file
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
