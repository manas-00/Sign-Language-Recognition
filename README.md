# SignLanguageDetection

## CollectData.py
This Python script uses OpenCV to capture images from a webcam, processes them, and saves them into respective directories labeled from 'A' to 'Z' based on key presses. It continuously captures frames, displays the current frame and a defined region of interest (ROI), and saves the ROI when corresponding alphabet keys are pressed. The script maintains counts of images in each directory to name the saved files sequentially.


## DataTransformation
The script captures and processes video frames using MediaPipe to detect hand landmarks, saving the extracted keypoints for each frame as .npy files in organized directories. It initializes MediaPipe Hands, loops through predefined actions and sequences, reads frames from images, and performs hand landmark detection and annotation. The frames are displayed with collection status messages, and keypoints are exported to corresponding directories. A commented-out section indicates similar processing for special actions, and cleanup commands are included but commented out.


## AllFunction

The script sets up a system to capture and process images for hand landmark detection using MediaPipe. It imports necessary libraries, initializes MediaPipe's drawing and hand solutions, and defines functions for detecting hands, drawing landmarks, and extracting keypoints. The mediapipe_detection function converts images to RGB, makes predictions, and converts them back to BGR. The draw_styled_landmarks function draws landmarks on the detected hands, while the extract_keypoints function flattens and concatenates landmark coordinates. The script sets paths for saving data and defines actions (A-Z) and special actions ('thankyou', 'iloveyou', 'hello'), along with parameters for the number of sequences and their length.


## TrainModel
The script trains a deep learning model to recognize hand gestures based on keypoints extracted from images using MediaPipe. It imports necessary functions and libraries, defines special actions, and maps action labels to numerical values. It loads sequences of keypoint data from .npy files, creates training and testing datasets, and one-hot encodes the labels. The script defines an LSTM-based neural network model, and compiles it with categorical cross-entropy loss and accuracy metrics. The model is trained for 200 epochs, and its architecture is saved to a JSON file, while the trained weights are saved to an H5 file.


## FinalPrediction
This script uses a trained model to perform real-time hand gesture recognition from a webcam feed. It imports necessary functions and libraries, including the previously trained model architecture and weights. The script initializes detection variables and captures video from the webcam. It uses MediaPipe to detect hand landmarks in each frame, extracts keypoints, and feeds sequences of keypoints into the model for prediction. The predicted gesture is displayed on the screen, and a sentence is formed based on consecutive gestures. The accuracy of each prediction is also displayed. The script continuously loops through frames until the user presses 'q' to exit.