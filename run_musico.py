import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from statistics import mode
from musico.emotions.utils.datasets import get_labels
from musico.emotions.utils.inference import detect_faces
from musico.emotions.utils.inference import draw_text
from musico.emotions.utils.inference import draw_bounding_box
from musico.emotions.utils.inference import apply_offsets
from musico.emotions.utils.inference import load_detection_model
from musico.emotions.utils.preprocessor import preprocess_input
import pkg_resources
import os

USE_WEBCAM = False # If false, loads video file source

# hyper-parameters for bounding boxes shape
frame_window = 20
emotion_offsets = (20, 40)

# loading models
haarcascade_path = pkg_resources.resource_filename('musico.emotions.models', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haarcascade_path)
emotion_model_path = pkg_resources.resource_filename('musico.emotions.models', 'emotion_model.hdf5')
emotion_model_path = './musico/emotions/models/emotion_model.hdf5'
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Variables for collecting data from video
emotion_labels = pd.Series(get_labels('fer2013'))
emotion_history = pd.DataFrame(columns=emotion_labels)

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    demo_file = pkg_resources.resource_filename('musico.emotions.demo', 'dinner.mp4')
    cap = cv2.VideoCapture(demo_file) # Video file source

# Loop over frames of video feed
iframe = 0
while cap.isOpened(): # True:
    print(f'iframe = {iframe}')
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5,
		minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    ## Loop over faces in frame
    emotion_faces = pd.DataFrame(columns=emotion_labels, index=range(len(faces)))
    for iface, face_coordinates in enumerate(faces):

        ### Preprocess face
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        ### Recognize emotions
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_prediction = pd.Series(emotion_prediction[0], index=emotion_labels)
        emotion_faces.loc[iface, :] = emotion_prediction
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    ## Collect emotion probabilities (averaged over all faces in frame)
    emotion_face_average = emotion_faces.mean(axis=0)
    emotion_history.loc[iframe, :] = emotion_face_average
    print(emotion_history)

    ## Show frame with added emotion boxes
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    ## Check end event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ## Increase frame counter
    iframe += 1

cap.release()
cv2.destroyAllWindows()
