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
import datetime
import time

import yaml
from configparser import ConfigParser
from pprint import pprint

from musico.instructions import instruct

# ----------------------------------------------------------------------------------------------------------
# Main function for running as script
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Import setttings
    with open("settings.yaml", "r") as yamlfile:
        settings = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Imported settings: ")
    pprint(settings)

    ## Assign Program parameters
    USE_WEBCAM = int(settings['Program']['use_webcam'])
    ROLLING_WINDOW = int(settings['Program']['rolling_window'])
    NTH_FRAME = int(settings['Program']['nth_frame'])
    SAVE_PATH = settings['Program']['save_path']


    ## Assign Emotion parameters
    EMOTIONS_USED = pd.Series(settings['Emotions']['Used'])
    EMOTIONS_COLORS = pd.Series(settings['Emotions']['Colors'])

    ## Assign Instructions
    INSTRUCTIONS_USED = pd.Series(settings['Instructions']['Used'])
    INSTRUCTIONS_END = pd.Series(settings['Instructions']['End'])
    INSTRUCTIONS_RULES = settings['Instructions']['Rules']

    ## Assign Events
    EVENTS = settings['Events']

    # Prepare data storage: data, videos and plots
    output_file_stem = 'ml-musico'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    dt = datetime.datetime.now().strftime('%Y-%m-%dT%H%M')
    output_file_video = os.path.join(
        SAVE_PATH, 
        f'{output_file_stem}_video_{dt}.avi' # save video feed to file
    )
    output_file_video_annotated = os.path.join(
        SAVE_PATH, 
        f'{output_file_stem}_video_annotated_{dt}.avi' # save video feed to file
    )
    output_file_emotionplot = os.path.join(
        SAVE_PATH,
        f'{output_file_stem}_emotionplot_{dt}.png' # save emotion plot to file
    )
    output_file_emotiondata = os.path.join(
        SAVE_PATH,
        f'{output_file_stem}_emotiondata_{dt}.xlsx' # save emotion plot to file
    )


    # Prepare the emotion-face-classifier

    ## Hyper-parameters for bounding boxes shape
    emotion_offsets = (20, 40)

    ## Loading models
    haarcascade_path = pkg_resources.resource_filename('musico.emotions.models', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    emotion_model_path = pkg_resources.resource_filename('musico.emotions.models', 'emotion_model.hdf5')
    emotion_classifier = load_model(emotion_model_path)

    ## getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    ## Variables for collecting data from video
    emotion_labels = pd.Series(get_labels('fer2013'))
    emotion_history = pd.DataFrame(columns=['frame','time'] + list(EMOTIONS_USED))
    ## Colors for emotions
    emotion_colors = EMOTIONS_COLORS.apply(lambda color: mcolors.to_rgb(color))
    emotion_colors = emotion_colors.apply(lambda tup: tuple(map(lambda x: x*255, tup)))
    ## Initialize with random instruction
    new_instruction = instruct.give_random_instruction(
        instructions=INSTRUCTIONS_USED, 
        rules=INSTRUCTIONS_RULES,
        history=None,
    )
    instruction_history = instruct.update_history(
        history = None, frame = 0, time = 0, key = new_instruction['key'],
    )

    # Prepare plots
    ## For following emotions
    fig_em, ax_em = plt.subplots()
    ax_em.set_ylim(0,1)
    plotlines = pd.Series(index=emotion_labels, dtype='object')
    for emotion in EMOTIONS_USED:
        line, = ax_em.plot(0, 0, color=EMOTIONS_COLORS[emotion], label=emotion)
        plotlines[emotion] = line
    ax_em.legend(loc='upper left')
    ax_em.set_xlabel('Frame')
    ax_em.set_ylabel('Emotion probability (average)')
    fig_em.show()
    ## For giving instructions
    fig_in, ax_in = plt.subplots()
    ax_in.set_xlim(0, 1)
    ax_in.set_ylim(0, 1)
    ax_in.set_axis_off()
    ax_in = instruct.draw_instruction(ax=ax_in, instruction=new_instruction)
    fig_in.show()

    # Starting video streaming
    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)
    ## Select video or webcam feed
    cap = None
    if (USE_WEBCAM >= 0):
        cap = cv2.VideoCapture(USE_WEBCAM) # Webcam source
    else:
        demo_file = pkg_resources.resource_filename('musico.emotions.demo', 'dinner.mp4')
        cap = cv2.VideoCapture(demo_file) # Video file source

    ## Prepare video save to file
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    output_fwriter = cv2.VideoWriter(
        filename=output_file_video,
        fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), # AVI
        # fourcc=cv2.VideoWriter_fourcc('X','2','6','4'), # MP4
        fps=24, 
        frameSize=frame_size,
    )
    output_fwriter_annotated = cv2.VideoWriter(
        filename=output_file_video_annotated,
        fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), # AVI
        # fourcc=cv2.VideoWriter_fourcc('X','2','6','4'), # MP4
        fps=24, 
        frameSize=frame_size,
    )

    # Loop over frames of video feed
    time0 = time.perf_counter()
    iframe = -1
    while cap.isOpened(): # True:
        ## Increase frame counter, set time
        iframe += 1
        itime = round(time.perf_counter() - time0, 2)
        print(f'iframe = {iframe}, itime = {itime}s')

        ## Grab, but not process, the next frame
        retval = cap.grab()

        ## Process only every n-th frame
        if (iframe % NTH_FRAME) != 0:
            continue
        else:
            retval, bgr_image = cap.retrieve()

        ## Check if video is empty
        if retval is False:
            break

        ## Save to file
        output_fwriter.write(bgr_image)

        ## Start face recognition
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        ## Loop over faces in frame
        emotion_faces = pd.DataFrame(columns=EMOTIONS_USED, index=range(len(faces)))
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
            emotion_prediction = emotion_prediction[EMOTIONS_USED]
            emotion_faces.loc[iface, :] = emotion_prediction
            emotion_max_probability = emotion_prediction.max()
            emotion_text = emotion_prediction.idxmax()
            emotion_color = emotion_colors[emotion_text]

            ### Add bounding box to face in image
            draw_bounding_box(face_coordinates, rgb_image, emotion_color)
            draw_text(
                coordinates=face_coordinates, 
                image_array=rgb_image, 
                text=emotion_text,
                color=emotion_color, 
                x_offset=0, y_offset=-45, font_scale=1, thickness=1,
            )

        ## Add frame and time info to image
        draw_text(
            coordinates=[0, frame_size[1], 0, 0], 
            image_array=rgb_image, 
            text=f'frame = {iframe}, time = {itime}s',
            color=(0,0,0), # color
            x_offset=0, y_offset=-5, font_scale=0.5, thickness=1, 
        )

        ## Collect emotion probabilities (averaged over all faces in frame)
        emotion_face_average = emotion_faces.mean(axis=0)
        emotion_history.loc[iframe, ['frame','time']] = [iframe, itime]
        emotion_history.loc[iframe, EMOTIONS_USED] = emotion_face_average
        emotion_history_rolling = emotion_history.rolling(window=ROLLING_WINDOW).mean()
        for emotion in EMOTIONS_USED:
            line = plotlines[emotion]
            line.set_ydata(emotion_history_rolling.loc[:, emotion])
            line.set_xdata(emotion_history_rolling.index)
        ax_em.set_xlim(0,max(1, max(emotion_history_rolling.index)))
        fig_em.canvas.draw()
        fig_em.canvas.flush_events()

        ## Show frame with added emotion boxes
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        output_fwriter_annotated.write(bgr_image)
        ## Wait for "q" event
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        # Evaluate instructions
        if itime > INSTRUCTIONS_RULES['max_minutes_song']*60:
            ## Song is over
            new_instruction = INSTRUCTIONS_END['END']
            new_instruction['key'] = 'END'
            instruct.update_history(
                history = instruction_history, frame=iframe, time=itime, key=new_instruction['key']
            )
        else:
            ## Song is still on

            ### Check time of last instruction
            time_since_last_instruction = itime - instruction_history.iloc[-1, :]['time']
            if time_since_last_instruction > INSTRUCTIONS_RULES['max_seconds_between_instructions']:
                ### Give new instruction because none has been given for too long
                new_instruction = instruct.give_random_instruction(
                    instructions=INSTRUCTIONS_USED, 
                    rules=INSTRUCTIONS_RULES,
                    history=instruction_history
                )
                instruct.update_history(
                    history = instruction_history, frame=iframe, time=itime, key=new_instruction['key']
                )
            else:
                ### Last instruction was quite recent

                if time_since_last_instruction <= INSTRUCTIONS_RULES['min_seconds_between_instructions']:
                    #### Stay with current instruction
                    pass
                else:
                    #### Evaluate events
                    pass

        del ax_in.texts[-1]
        ax_in = instruct.draw_instruction(ax=ax_in, instruction=new_instruction)
        fig_in.canvas.draw()
        fig_in.canvas.flush_events()


    # Close video feed
    cap.release()
    cv2.destroyAllWindows()

    # Save emotion plot
    fig_em.savefig(output_file_emotionplot)

    # Save data
    history = pd.merge(left=instruction_history, right=emotion_history, how='outer', on=['frame','time']).sort_values(by=['frame','time'])
    history.to_excel(output_file_emotiondata, index=False)