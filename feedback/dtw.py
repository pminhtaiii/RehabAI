"""
Auto-converted from dtw.ipynb on 2026-04-25 19:04:47
"""

# %%
# Installing necessary packages
# Notebook-only setup commands kept as comments for reference:
# !git clone https://github.com/tensorflow/docs.git
# !pip install -q ./docs
# !pip install -q tensorflow-hub
# !pip install -q opencv-python
# !pip install -q tslearn

# %%
# Importing necessary packages
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

import pandas as pd
import os
import subprocess

from tslearn.metrics import dtw as ts_dtw

# Running a Jupyter notebook file named 'movenet.ipynb'
from movenet import *  # noqa: F401,F403

# %%
# Function to process a single frame
def process_frame(frame, crop_region, frame_height, frame_width):

    keypoints_with_scores = run_inference(
        movenet, frame,
        crop_region,
        crop_size=[input_size, input_size])
    crop_region = determine_crop_region(keypoints_with_scores, frame_height, frame_width)
    frame_record = get_video_frame_record(keypoints_with_scores)
    
    return frame_record, crop_region

# %%
# Function to create a DataFrame from video records
def dataframe(video_records):
    video_df_cols = get_dataframe_cols()
    video_df = pd.DataFrame(data=video_records, columns=video_df_cols)
    video_df.to_csv("real_time_output.csv", index=False)

# %% [markdown]
# ## Feedback -1

# %%
# List of body joints for analysis
body_joints = ['right_shoulder_x','right_shoulder_y',
              'left_shoulder_x', 'left_shoulder_y', 
              'right_elbow_x', 'right_elbow_y',
              'left_elbow_x', 'left_elbow_y',
              'right_wrist_x', 'right_wrist_y',
              'left_wrist_x', 'left_wrist_y',
              'left_hip_x', 'left_hip_y',
              'right_hip_x', 'right_hip_y',
              'left_knee_x', 'left_knee_y',
              'right_knee_x', 'right_knee_y']

# %%
# Reading ground truth data from a CSV file
ground_truth = pd.read_csv('csvs/E_ID15_Es1.csv')[body_joints]

# Function to provide feedback based on Dynamic Time Warping (DTW) comparisons
def feedback(counter):
    video = pd.read_csv('real_time_output.csv')
    for joint in body_joints:
        dtw = ts_dtw(video[joint][:counter], ground_truth[joint][:counter])
        if dtw > 2.5:
            axis = 'vertically' if joint[-1:] == 'y' else 'horizontally'
            print(f'Adjust your {joint.replace("_", " ")[:-2]} {axis}')

# %%
# Function to record a video of an exercise in real-time
def film_exercise():
    
    limit = 500
        
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_region = init_crop_region(frame_height, frame_width)

    video_records = []
    counter = 0
    start = False
    
    while True:
        ret, frame = cap.read()
        
        if counter == 0: 
            cv2.waitKey(10)
            print('START ... ')
            start = True
            
        counter+=1
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
        if counter == limit:
            break
            
        if start == True:
            frame_record, crop_region = process_frame(frame, crop_region, frame_height, frame_width)
            video_records.append(frame_record)
            dataframe(video_records)
            if counter % 30 == 0:
                print(f' {counter} \t\t {feedback(counter)}')
        else:
            continue

    cap.release()
    cv2.destroyAllWindows()

    print(counter)

# %%
# Calling the function to record the exercise in real-time
film_exercise()

# %%
