"""Extract 3D body landmarks from exercise videos using MediaPipe PoseLandmarker.

Uses the new Tasks API (mediapipe >= 0.10.x). Output CSV has columns per keypoint:
{name}_x, {name}_y, {name}_z, {name}_v (12 keypoints × 4 = 48 columns per frame).
"""

import os
import cv2
import glob
import numpy as np
import pandas as pd
import urllib.request
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


BASE_DIR = 'C:/RehabAI'

INPUT_BASE_DIR = f'{BASE_DIR}/01_raw_data'
OUTPUT_BASE_DIR = f'{BASE_DIR}/03_raw_joints'

# Model: lite / full / heavy (heavy = best accuracy, slower)
MODEL_VARIANT = 'heavy'
MODEL_URLS = {
    'lite':  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'full':  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'heavy': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}

# ── MediaPipe body-only keypoints ─────────────────────────────────────────────
# Exclude face, ears, eyes, mouth, hands, feet details — keep 12 body joints
BODY_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def get_dataframe_cols():
    """Return column names: 12 joints × 4 channels = 48 cols."""
    df_cols = []
    for name in BODY_LANDMARKS:
        df_cols.append(f"{name}_x")
        df_cols.append(f"{name}_y")
        df_cols.append(f"{name}_z")
        df_cols.append(f"{name}_v")
    return df_cols


def download_model(variant='heavy'):
    """Download the PoseLandmarker .task model file if not present."""
    url = MODEL_URLS[variant]
    filename = f'pose_landmarker_{variant}.task'
    if not os.path.exists(filename):
        print(f"Downloading {variant} model...")
        urllib.request.urlretrieve(url, filename)
        print(f"  ✓ Saved to {filename}")
    return filename


def extract_frame_record(landmarks):
    """Extract body keypoint values from PoseLandmarker result.

    Args:
        landmarks: list of NormalizedLandmark (33 total from MediaPipe)

    Returns:
        list of 48 floats: [x, y, z, visibility] for each of 12 body joints
    """
    record = []
    for name, idx in BODY_LANDMARKS.items():
        lm = landmarks[idx]
        record.append(lm.x)          # normalized x (0-1)
        record.append(lm.y)          # normalized y (0-1)
        record.append(lm.z)          # depth relative to hip midpoint
        record.append(lm.visibility)  # confidence score
    return record


def main():
    video_files = glob.glob(os.path.join(INPUT_BASE_DIR, '**', '*.mp4'), recursive=True)
    print(f"Found {len(video_files)} videos to process in {INPUT_BASE_DIR}")

    if len(video_files) == 0:
        print(f"No .mp4 files found. Check INPUT_BASE_DIR: {INPUT_BASE_DIR}")
        return

    # Download model file
    model_path = download_model(MODEL_VARIANT)

    # Initialize PoseLandmarker (Tasks API)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    df_cols = get_dataframe_cols()
    success_count = 0
    skip_count = 0
    # Global timestamp that NEVER resets — MediaPipe VIDEO mode requires
    # monotonically increasing timestamps across the entire session
    global_timestamp_ms = 0

    for i, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        video_dir = os.path.dirname(video_path)
        exercise_dir = os.path.dirname(video_dir)
        relative_path = os.path.relpath(exercise_dir, INPUT_BASE_DIR)

        save_dir = os.path.join(OUTPUT_BASE_DIR, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        # Add '_mediapipe' suffix to avoid overwriting MoveNet data
        csv_filename = video_filename.replace('.mp4', '_mediapipe.csv')
        csv_filepath = os.path.join(save_dir, csv_filename)

        if os.path.exists(csv_filepath):
            skip_count += 1
            continue

        print(f"[{i+1}/{len(video_files)}] Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ✗ Failed to open {video_path}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        video_records = []
        detected_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe Tasks API expects RGB + mp.Image wrapper
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Use global timestamp — always increasing across all videos
            global_timestamp_ms += int(1000 / fps)

            result = landmarker.detect_for_video(mp_image, global_timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]  # First detected pose
                record = extract_frame_record(landmarks)
                video_records.append(record)
                detected_count += 1
            else:
                # No pose detected — append zeros
                video_records.append([0.0] * len(df_cols))

        cap.release()

        if video_records:
            video_df = pd.DataFrame(data=video_records, columns=df_cols)
            video_df.to_csv(csv_filepath, index=False)
            success_count += 1
            detection_rate = (detected_count / len(video_records)) * 100
            print(f"  ✓ Saved: {csv_filepath}")
            print(f"    {len(video_records)} frames, {detected_count} detected ({detection_rate:.0f}%), {fps}fps")
        else:
            print(f"  ✗ Warning: No frames extracted for {video_path}")

    landmarker.close()

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {success_count} videos")
    print(f"  Skipped (already exists): {skip_count} videos")
    print(f"  Output format: 12 body joints × (x, y, z, visibility) = 48 columns")
    print(f"  Output suffix: _mediapipe.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
