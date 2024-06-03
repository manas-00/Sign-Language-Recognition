from AllFunction import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from gtts import gTTS
import pygame
import time
import os
import numpy as np

# Initialize pygame audio mixer
pygame.mixer.init()

actions = np.concatenate([actions, ['thankyou', 'iloveyou', 'hello']])

# Ensure colors list is the same length as actions
colors = [(245,117,16)] * len(actions)

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Function to convert text to speech and play the audio
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    filename = 'output.mp3'
    # Ensure pygame mixer has stopped
    pygame.mixer.music.stop()
    # Wait a short time to ensure file release
    time.sleep(0.5)
    # Check if the file exists and delete it
    if os.path.exists(filename):
        os.remove(filename)
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        if keypoints is not None:
            sequence.append(keypoints)
            sequence = sequence[-20:]

        if len(sequence) == 20 and all(kp is not None for kp in sequence):
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_alphabet = actions[np.argmax(res)]
            print(predicted_alphabet)

            # Convert predicted alphabet to voice and play
            text_to_speech(predicted_alphabet)
                
            # Update sentence and predictions lists
            if res[np.argmax(res)] > threshold: 
                if len(predictions) == 0 or (len(predictions) > 0 and predictions[-1] != np.argmax(res)):
                    predictions.append(np.argmax(res))
                    sentence.append(predicted_alphabet)

            # Truncate lists to keep only the last element
            if len(sentence) > 1:
                sentence = sentence[-1:]

            # Viz probabilities
            for num, prob in enumerate(res):
                cv2.rectangle(frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
                cv2.putText(frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + ' '.join(sentence), (3, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
