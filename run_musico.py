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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------------------------------------------------------------------------------------
# Main function for running as script
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Schemas and tables for the stats-packages')
    parser.add_argument(
        '--use_webcam', required=False, default=False, action='store_true',
        help='Whether to use the Webcam or a demo video.'
    )
    parser.add_argument(
        '--rolling_window', required=False, default=5, type=int,
        help='Number of frames used for the rolling window for averaging emotion probabilities.'
    )
    parser.add_argument(
        '--nth_frame', required=False, default=1, type=int,
        help='Use only every n-th frame for processing. Set a higher number here if CPU power is limiting.'
    )
    
    args = parser.parse_args()


    # Assign arguments
    USE_WEBCAM = args.use_webcam # If false, loads video file source
    rolling_window = args.rolling_window # number of frames for emotion mode
    nth_frame = args.nth_frame # process only every n-th frame

    # Hyper-parameters for bounding boxes shape
    emotion_offsets = (20, 40)

    # Loading models
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
    ## Colors for emotions
    emotion_colors_name = pd.Series({
        'angry': 'red',
        'disgust': 'brown',
        'fear': 'black',
        'happy': 'yellow',
        'sad': 'darkblue',
        'surprise': 'lightblue',
        'neutral': 'green',
    })
    emotion_colors = emotion_colors_name.apply(lambda color: mcolors.to_rgb(color))
    emotion_colors = emotion_colors.apply(lambda tup: tuple(map(lambda x: x*255, tup)))

    # Prepare plot for showing results
    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    plotlines = pd.Series(index=emotion_labels, dtype='object')
    for emotion in emotion_history.columns:
        line, = ax.plot(0, 0, color=emotion_colors_name[emotion], label=emotion)
        plotlines[emotion] = line
    ax.legend(loc='upper left')
    plt.show(block=False)

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
        if (iframe % nth_frame) != 0:
            # Process only every n-th frame
            iframe += 1
            continue

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
            emotion_max_probability = emotion_prediction.max()
            emotion_text = emotion_prediction.idxmax()
            emotion_color = emotion_colors[emotion_text]

            ### Add bounding box to frame
            draw_bounding_box(face_coordinates, rgb_image, emotion_color)
            draw_text(
                face_coordinates, rgb_image, emotion_text,
                emotion_color, 0, -45, 1, 1
            )

        ## Collect emotion probabilities (averaged over all faces in frame)
        emotion_face_average = emotion_faces.mean(axis=0)
        emotion_history.loc[iframe, :] = emotion_face_average
        emotion_history_rolling = emotion_history.rolling(window=rolling_window).mean()
        # emotion_history_rolling = emotion_history.copy()
        # for emotion in emotion_history.columns:
        #     emotion_history_rolling[emotion] = emotion_history.rolling(window=rolling_window).mean()
        print(emotion_history)
        print(emotion_history_rolling)
        for emotion in emotion_history.columns:
            line = plotlines[emotion]
            line.set_ydata(emotion_history_rolling.loc[:, emotion])
            line.set_xdata(emotion_history_rolling.index)
        ax.set_xlim(0,max(emotion_history_rolling.index))
        fig.canvas.draw()
        fig.canvas.flush_events()

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
