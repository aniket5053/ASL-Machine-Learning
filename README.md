# Hand Gesture Recognition using MediaPipe and RandomForestClassifier

This project implements a hand gesture recognition system using MediaPipe for hand tracking and a RandomForestClassifier for gesture classification. The system captures hand gestures through a webcam, processes the images to extract hand landmarks, and classifies the gestures into predefined categories.

## Project Overview

- **Hand Gesture Data Collection:** Collects hand gesture images from a webcam and saves them in a structured format.
- **Model Training:** Uses the collected images to train a RandomForestClassifier to recognize different hand gestures.
- **Real-time Gesture Recognition:** Utilizes the trained model to recognize and display hand gestures in real-time.

## Files Description

- `collect_data.py`: Script to collect hand gesture images and save them in the `./data` directory. The script captures frames from the webcam, processes them using MediaPipe to detect hand landmarks, and saves the frames as images.
- `train_model.py`: Script to train a RandomForestClassifier model using the collected dataset. The script loads the dataset, trains the model, evaluates its accuracy, and saves the trained model to a pickle file.
- `real_time_recognition.py`: Script to perform real-time hand gesture recognition using the trained model. The script captures frames from the webcam, processes them using MediaPipe to detect hand landmarks, predicts the gesture, and displays the result on the screen.
- `data.pickle`: Serialized dataset containing hand landmarks and labels.
- `model.p`: Serialized trained RandomForestClassifier model.

