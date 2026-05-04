"""
joint_features.py — Real-time feature extraction from MediaPipe landmarks
===================================================================================
Extracts 2D features with aspect ratio correction from incoming MediaPipe landmarks.
These functions mirror 03_extract_joint_features.py EXACTLY so that
inference matches training.

MediaPipe landmark mapping:
0: nose
11, 12: left_shoulder, right_shoulder
13, 14: left_elbow, right_elbow
15, 16: left_wrist, right_wrist
23, 24: left_hip, right_hip
25, 26: left_knee, right_knee
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# 2D Geometry Functions (Aspect Ratio Corrected)
# ══════════════════════════════════════════════════════════════════════════════

def get_joint_2d(df, joint_name):
    """Return (N, 2) array of [x, y] with aspect ratio corrected (x * 1.7778)."""
    return np.column_stack((
        df[f"{joint_name}_x"] * 1.7778,
        df[f"{joint_name}_y"]
    ))

def calculate_angle_2d(first, middle, end):
    """Angle (degrees) at vertex *middle* in 2D. Uses atan2(cross, dot)."""
    v1 = np.array(first) - np.array(middle)
    v2 = np.array(end) - np.array(middle)
    
    # Cross product in 2D: x1*y2 - x2*y1
    cross_prod = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot_prod = np.sum(v1 * v2, axis=1)
    
    # Use abs to ensure angle is between 0 and 180
    return pd.Series(np.degrees(np.abs(np.arctan2(cross_prod, dot_prod))))

def calculate_distance_2d(pair1, pair2):
    """Euclidean distance in 2D between two (N, 2) joint arrays."""
    return pd.Series(np.linalg.norm(pair1 - pair2, axis=1))

def get_torso_vector_2d(df):
    """Torso vector from shoulder midpoint to hip midpoint (N, 2)."""
    shoulder_mid = (get_joint_2d(df, "left_shoulder") + get_joint_2d(df, "right_shoulder")) / 2
    hip_mid = (get_joint_2d(df, "left_hip") + get_joint_2d(df, "right_hip")) / 2
    return hip_mid - shoulder_mid

def calculate_vector_angle_2d(vec1, vec2):
    """Angle (degrees) between two (N, 2) vector arrays."""
    dot = np.sum(vec1 * vec2, axis=1)
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    cos_angle = dot / (norm1 * norm2 + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return pd.Series(np.degrees(np.arccos(cos_angle)))


# ══════════════════════════════════════════════════════════════════════════════
# Per-Exercise Feature Extractors
# ══════════════════════════════════════════════════════════════════════════════

def get_es1_features(df):
    """Es1 — Lifting of arms (6 features)."""
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)
    hand_vec = get_joint_2d(df, "right_wrist") - get_joint_2d(df, "left_wrist")
    horizontal = np.tile([1, 0], (len(df), 1))
    features_df["hand_tilted_angle"] = calculate_vector_angle_2d(hand_vec, horizontal)
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)
    return features_df

def get_es2_features(df):
    """Es2 — Lateral tilt of trunk (6 features)."""
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)
    features_df["left_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    return features_df

def get_es3_features(df):
    """Es3 — Trunk rotation (9 features)."""
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)
    features_df["left_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    left_arm_vec = get_joint_2d(df, "left_elbow") - get_joint_2d(df, "left_shoulder")
    features_df["left_arm_torso_angle"] = calculate_vector_angle_2d(torso_vec, left_arm_vec)
    right_arm_vec = get_joint_2d(df, "right_elbow") - get_joint_2d(df, "right_shoulder")
    features_df["right_arm_torso_angle"] = calculate_vector_angle_2d(torso_vec, right_arm_vec)
    return features_df

def get_es4_features(df):
    """Es4 — Pelvis rotation (2 features)."""
    features_df = pd.DataFrame()
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)
    knee_dist = calculate_distance_2d(get_joint_2d(df, "left_knee"), get_joint_2d(df, "right_knee"))
    hip_dist = calculate_distance_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "right_hip"))
    features_df["knee_hip_ratio"] = knee_dist / (hip_dist + 1e-8)
    return features_df

def get_es5_features(df):
    """Es5 — Squatting (7 features)."""
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)
    features_df["left_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    return features_df

FEATURE_EXTRACTORS = {
    'Es1': get_es1_features,
    'Es2': get_es2_features,
    'Es3': get_es3_features,
    'Es4': get_es4_features,
    'Es5': get_es5_features,
}

def extract_features(df, exercise):
    extractor = FEATURE_EXTRACTORS.get(exercise)
    if not extractor:
        return pd.DataFrame()
    return extractor(df)