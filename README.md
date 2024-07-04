# Indian Sign Language Recognition System 


This project focuses on real-time sign language detection using hand landmarks extracted from video input. It combines computer vision techniques with machine learning to recognize gestures corresponding to letters of the alphabet.

**Key Dependencies:**

Python: Programming language used throughout the project.
OpenCV: Library for computer vision tasks such as video capture, image processing, and visualization.
Mediapipe: Google's framework for building applied machine learning pipelines, utilized for detecting and tracking hand landmarks.
NumPy: Library for numerical operations, facilitating efficient array manipulation and data handling.
Scikit-learn: Python machine learning library used to train a Random Forest classifier for gesture recognition.
Pickle: Python module for object serialization, employed to save trained machine learning models.


**Scripts:**

img_collection.py: Collects training images from a webcam for each sign language gesture.
dataset.py: Prepares the collected images into a structured dataset, extracting hand landmarks using Mediapipe.
train.py: Trains a Random Forest classifier on the prepared dataset to recognize sign language gestures.
inference.py: Performs real-time inference using the trained model to predict sign language gestures from webcam input.

