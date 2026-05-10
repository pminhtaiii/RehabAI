"""DTW-based feedback system for exercise comparison."""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
import pandas as pd
import os
import subprocess

from tslearn.metrics import dtw as ts_dtw

from movenet import *


def process_frame(frame, crop_region, frame_height, frame_width):
    """Process a single frame for pose detection."""
    keypoints_with_scores = run_inference(
        movenet, frame,
        crop_region,
        crop_size=[input_size, input_size])
    crop_region = determine_crop_region(keypoints_with_scores, frame_height, frame_width)
    frame_record = get_video_frame_record(keypoints_with_scores)
    return frame_record, crop_region


def dataframe(video_records):
    """Create a DataFrame from video records and save to CSV."""
    video_df_cols = get_dataframe_cols()
    video_df = pd.DataFrame(data=video_records, columns=video_df_cols)
    video_df.to_csv("real_time_output.csv", index=False)


body_joints = ['right_shoulder_x', 'right_shoulder_y',
               'left_shoulder_x', 'left_shoulder_y',
               'right_elbow_x', 'right_elbow_y',
               'left_elbow_x', 'left_elbow_y',
               'right_wrist_x', 'right_wrist_y',
               'left_wrist_x', 'left_wrist_y',
               'left_hip_x', 'left_hip_y',
               'right_hip_x', 'right_hip_y',
               'left_knee_x', 'left_knee_y',
               'right_knee_x', 'right_knee_y']

ground_truth = pd.read_csv('csvs/E_ID15_Es1.csv')[body_joints]


def feedback(counter):
    """Provide feedback based on DTW comparisons."""
    video = pd.read_csv('real_time_output.csv')
    for joint in body_joints:
        dtw = ts_dtw(video[joint][:counter], ground_truth[joint][:counter])
        if dtw > 2.5:
            axis = 'vertically' if joint[-1:] == 'y' else 'horizontally'
            print(f'Adjust your {joint.replace("_", " ")[:-2]} {axis}')


def film_exercise():
    """Record a video of an exercise in real-time."""
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

        counter += 1
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
        if counter == limit:
            break

        if start:
            frame_record, crop_region = process_frame(frame, crop_region, frame_height, frame_width)
            video_records.append(frame_record)
            dataframe(video_records)
            if counter % 30 == 0:
                print(f' {counter} \t\t {feedback(counter)}')

    cap.release()
    cv2.destroyAllWindows()
    print(counter)


if __name__ == '__main__':
    film_exercise()
