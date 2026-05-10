"""Feature extraction from MediaPipe landmarks.

Extracts 2D features with aspect ratio correction from incoming MediaPipe landmarks.
Must match 03_extract_joint_features.py exactly for training/inference consistency.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter as scipy_median_filter

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


def apply_median_filter(joint_df, window=5):
    filtered = joint_df.copy()
    coord_cols = [c for c in filtered.columns if c.endswith('_x') or c.endswith('_y')]
    for col in coord_cols:
        filtered[col] = scipy_median_filter(filtered[col].values, size=window, mode='nearest')
    return filtered


def calculate_angular_velocity(angle_series):
    vel = np.diff(angle_series, prepend=angle_series.iloc[0] if hasattr(angle_series, 'iloc') else angle_series[0])
    return pd.Series(vel)


# ══════════════════════════════════════════════════════════════════════════════
# Per-Exercise Feature Extractors
# ══════════════════════════════════════════════════════════════════════════════

def get_es1_features(df):
    """Es1 — Lifting of arms (10 features).
    Guo&Khan (2021) baseline (6) + shoulder_angle L/R (Capecci 2019 PO α_l/r) +
    wrist_height_ratio (Proietti 2022) + wrist_elevation_velocity (Liao 2020).
    """
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
    # ── PRIMARY OUTCOME: Shoulder angle (Capecci 2019 α_l/r) ──
    features_df["left_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    # ── Wrist height ratio (Proietti 2022) ──
    shoulder_mid = (get_joint_2d(df, "left_shoulder") + get_joint_2d(df, "right_shoulder")) / 2
    wrist_mid = (get_joint_2d(df, "left_wrist") + get_joint_2d(df, "right_wrist")) / 2
    hip_mid = (get_joint_2d(df, "left_hip") + get_joint_2d(df, "right_hip")) / 2
    torso_length = hip_mid[:, 1] - shoulder_mid[:, 1]
    wrist_elevation = shoulder_mid[:, 1] - wrist_mid[:, 1]
    features_df["wrist_height_ratio"] = wrist_elevation / (torso_length + 1e-8)
    # ── Wrist elevation velocity (Liao 2020) ──
    features_df["wrist_elevation_velocity"] = calculate_angular_velocity(features_df["wrist_height_ratio"])
    return features_df

def get_es2_features(df):
    """Es2 — Lateral tilt of trunk (8 features).
    Guo&Khan baseline (6) + lateral_displacement (Capecci 2019) +
    torso_tilt_velocity.

    lateral_displacement: horizontal displacement of shoulder midpoint relative
    to hip midpoint, normalized by hip width. Complements torso_tilted_angle
    by capturing the DISTANCE component (angle alone doesn't distinguish
    tall vs short subjects with same tilt angle).
    Source: Capecci et al. (2019), CF feature δ (trunk displacement).
    Expected Spearman +0.04.

    torso_tilt_velocity: temporal derivative of torso tilt angle.
    Captures movement dynamics — stroke patients show irregular velocity
    profiles during lateral tilt.
    Source: Capecci et al. (2019), velocity as motion descriptor.
    """
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    torso_angle = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["torso_tilted_angle"] = torso_angle
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)
    features_df["left_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    # ── NEW: Lateral displacement (Capecci 2019 CF δ) ──
    # Horizontal displacement of shoulder midpoint vs hip midpoint, normalized.
    shoulder_mid = (get_joint_2d(df, "left_shoulder") + get_joint_2d(df, "right_shoulder")) / 2
    hip_mid = (get_joint_2d(df, "left_hip") + get_joint_2d(df, "right_hip")) / 2
    hip_dist = calculate_distance_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "right_hip"))
    features_df["lateral_displacement"] = np.abs(shoulder_mid[:, 0] - hip_mid[:, 0]) / (hip_dist + 1e-8)
    # ── NEW: Torso tilt velocity ──
    features_df["torso_tilt_velocity"] = calculate_angular_velocity(torso_angle)
    return features_df

def get_es3_features(df):
    """Es3 — Trunk rotation (9 features).
    Guo&Khan baseline (7) + shoulder_width_ratio (Chen 2021) +
    shoulder_width_velocity.

    shoulder_width_ratio: shoulder_dist / hip_dist. Captures 2D perspective
    distortion from trunk rotation — the ONLY 2D feature that provides
    rotation information in frontal view.

    Physics: When trunk rotates ±30° around vertical axis:
      - projected_shoulder_width = actual_width × cos(θ)
      - cos(30°) = 0.866 → ~13% width change
      - MediaPipe noise: ±3-5% per frame → per-frame SNR ≈ 2.6
      - LSTM processes ~100-150 frames → effective SNR ≈ 26 (temporal smoothing)
    Normalization: hip_dist is relatively stable during rotation (hips don't
    rotate as much as shoulders), making this camera-distance invariant.

    Trade-off:
      PRO: Only way to capture rotation in 2D. Es3 currently has NO
           rotation-specific feature — all existing features are either
           arm-related (elbow) or tilt-related (torso_angle).
      CON: Low per-frame SNR, confounded by shoulder shrugging.
      VERDICT: 1 feature, low overfitting risk. Even noisy rotation info
               > no rotation info for LSTM temporal processing.

    Source: Chen et al. (2021), IEEE Access — perspective distortion model.
    Expected Spearman +0.05-0.10 for Es3.

    shoulder_width_velocity: temporal derivative of shoulder_width_ratio.
    Rotation speed pattern — healthy rotation is smooth, stroke is jerky.
    """
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
    # ── NEW: Shoulder width ratio (Chen 2021 perspective distortion) ──
    # shoulder_dist / hip_dist — captures rotation via 2D perspective change.
    # Hips are more stable than shoulders during trunk rotation → good normalizer.
    hip_dist = calculate_distance_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "right_hip"))
    sw_ratio = shoulder_dist / (hip_dist + 1e-8)
    features_df["shoulder_width_ratio"] = sw_ratio
    # ── NEW: Shoulder width velocity — rotation smoothness ──
    features_df["shoulder_width_velocity"] = calculate_angular_velocity(sw_ratio)
    return features_df

def get_es4_features(df):
    """Es4 — Pelvis rotation (6 features).
    Original: 2 features. NEW: +hip_angle L/R (Capecci 2019 CF ψ_l/r),
    +hip_angles_diff (bilateral symmetry), +torso_tilted_velocity.
    """
    features_df = pd.DataFrame()
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    torso_angle = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["torso_tilted_angle"] = torso_angle
    knee_dist = calculate_distance_2d(get_joint_2d(df, "left_knee"), get_joint_2d(df, "right_knee"))
    hip_dist = calculate_distance_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "right_hip"))
    features_df["knee_hip_ratio"] = knee_dist / (hip_dist + 1e-8)
    left_hip = calculate_angle_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_knee"))
    features_df["left_hip_angle"] = left_hip
    right_hip = calculate_angle_2d(get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_knee"))
    features_df["right_hip_angle"] = right_hip
    features_df["hip_angles_diff"] = np.abs(left_hip - right_hip)
    features_df["torso_tilted_velocity"] = calculate_angular_velocity(torso_angle)
    return features_df

def get_es5_features(df):
    """Es5 — Squatting (8 features).
    Reduced from 11→8 to fix overfitting (67 samples / 11 features = 6:1 → too low).
    REMOVED: left/right_elbow_angle, elbow_angles_diff (static during squat per Guo&Khan 2021),
             knee_angle_velocity (unilateral → biased; flip±1° aug covers temporal smoothness).
    """
    features_df = pd.DataFrame()
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["left_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    # Primary Outcomes: knee flexion angles (sagittal plane) — Capecci 2019 θ_l/r
    left_knee = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_knee"), get_joint_2d(df, "left_ankle"))
    right_knee = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_knee"), get_joint_2d(df, "right_ankle"))
    features_df["left_knee_angle"] = left_knee
    features_df["right_knee_angle"] = right_knee
    # Frontal plane alignment: knee valgus detection (Proietti 2022)
    knee_dist = calculate_distance_2d(get_joint_2d(df, "left_knee"), get_joint_2d(df, "right_knee"))
    ankle_dist = calculate_distance_2d(get_joint_2d(df, "left_ankle"), get_joint_2d(df, "right_ankle"))
    features_df["knee_ankle_ratio"] = knee_dist / (ankle_dist + 1e-8)
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
    df = apply_median_filter(df, window=5)
    return extractor(df)