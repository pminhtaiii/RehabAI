import pandas as pd
import numpy as np
import math


def get_joint_pair(df, joint_name):
  return [df[f"{joint_name}_x"], df[f"{joint_name}_y"]]


def calculate_angle(first, middle, end):
    first = np.array(first)
    middle = np.array(middle)
    end = np.array(end)

    radians = np.arctan2(end[1]-middle[1], end[0]-middle[0]) - np.arctan2(first[1]-middle[1], first[0]-middle[0])
    angle = np.abs(radians*180.0/np.pi)

    angles = [360-a if a > 180 else a for a in angle]
    return angles


def calculate_distance(pair1, pair2):
  pair1_x, pair1_y = pair1
  pair2_x, pair2_y = pair2
  return pd.Series([math.dist([x1, y1], [x2, y2]) for x1, y1, x2, y2 in zip(pair1_x, pair1_y, pair2_x, pair2_y)])


def get_es1_features(df):
  features_df = pd.DataFrame()

  features_df["left_arm_torso_angle"] = calculate_angle(get_joint_pair(df, "left_elbow"),
                                                   get_joint_pair(df, "left_shoulder"),
                                                   get_joint_pair(df, "left_hip"))

  features_df["right_arm_torso_angle"] = calculate_angle(get_joint_pair(df, "right_elbow"),
                                                   get_joint_pair(df, "right_shoulder"),
                                                   get_joint_pair(df, "right_hip"))

  features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"),
                                                     get_joint_pair(df, "left_elbow"),
                                                     get_joint_pair(df, "left_wrist"))

  features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"),
                                                     get_joint_pair(df, "right_elbow"),
                                                     get_joint_pair(df, "right_wrist"))

  features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                                    get_joint_pair(df, "left_knee"),
                                                    get_joint_pair(df, "left_ankle"))

  features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"),
                                                    get_joint_pair(df, "right_knee"),
                                                    get_joint_pair(df, "right_ankle"))

  mid_hip_point = [(get_joint_pair(df, "left_hip")[0] + get_joint_pair(df, "right_hip")[0])/2, (get_joint_pair(df, "left_hip")[1] + get_joint_pair(df, "right_hip")[1])/2]
  features_df["hip_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                        mid_hip_point,
                                        get_joint_pair(df, "right_hip"))

  features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))

  features_df["ankle_dist"] = calculate_distance(get_joint_pair(df, "left_ankle"), get_joint_pair(df, "right_ankle"))

  return features_df


def get_es2_features(df):
  features_df = pd.DataFrame()

  features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"),
                                                     get_joint_pair(df, "left_elbow"),
                                                     get_joint_pair(df, "left_wrist"))

  features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"),
                                                     get_joint_pair(df, "right_elbow"),
                                                     get_joint_pair(df, "right_wrist"))

  features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                                    get_joint_pair(df, "left_knee"),
                                                    get_joint_pair(df, "left_ankle"))

  features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"),
                                                    get_joint_pair(df, "right_knee"),
                                                    get_joint_pair(df, "right_ankle"))

  mid_hip_point = [(get_joint_pair(df, "left_hip")[0] + get_joint_pair(df, "right_hip")[0])/2, (get_joint_pair(df, "left_hip")[1] + get_joint_pair(df, "right_hip")[1])/2]
  features_df["hip_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                        mid_hip_point,
                                        get_joint_pair(df, "right_hip"))

  features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))

  features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))

  features_df["left_shoulder_wrist_vert_dist"] = np.abs(df["left_shoulder_y"] - df["left_wrist_y"])

  features_df["right_shoulder_wrist_vert_dist"] = np.abs(df["right_shoulder_y"] - df["right_wrist_y"])

  return features_df


def get_es3_features(df):
  features_df = pd.DataFrame()

  features_df["elbows_horiz_dist"] = np.abs(df["left_elbow_x"] - df["right_elbow_x"])

  features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"),
                                                     get_joint_pair(df, "left_elbow"),
                                                     get_joint_pair(df, "left_wrist"))

  features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"),
                                                     get_joint_pair(df, "right_elbow"),
                                                     get_joint_pair(df, "right_wrist"))

  features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                                    get_joint_pair(df, "left_knee"),
                                                    get_joint_pair(df, "left_ankle"))

  features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"),
                                                    get_joint_pair(df, "right_knee"),
                                                    get_joint_pair(df, "right_ankle"))

  features_df["left_shoulder_extension_angle"] = calculate_angle(get_joint_pair(df, "left_elbow"),
                                                     get_joint_pair(df, "left_shoulder"),
                                                     get_joint_pair(df, "right_shoulder"))

  features_df["right_shoulder_extension_angle"] = calculate_angle(get_joint_pair(df, "right_elbow"),
                                                     get_joint_pair(df, "right_shoulder"),
                                                     get_joint_pair(df, "left_shoulder"))

  mid_hip_point = [(get_joint_pair(df, "left_hip")[0] + get_joint_pair(df, "right_hip")[0])/2, (get_joint_pair(df, "left_hip")[1] + get_joint_pair(df, "right_hip")[1])/2]
  features_df["hip_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                        mid_hip_point,
                                        get_joint_pair(df, "right_hip"))

  features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))

  features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))

  features_df["hip_dist"] = calculate_distance(get_joint_pair(df, "left_hip"), get_joint_pair(df, "right_hip"))

  features_df["left_shoulder_wrist_vert_dist"] = np.abs(df["left_shoulder_y"] - df["left_wrist_y"])

  features_df["right_shoulder_wrist_vert_dist"] = np.abs(df["right_shoulder_y"] - df["right_wrist_y"])

  return features_df


def get_es4_features(df):
  features_df = pd.DataFrame()

  features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"),
                                                     get_joint_pair(df, "left_elbow"),
                                                     get_joint_pair(df, "left_wrist"))

  features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"),
                                                     get_joint_pair(df, "right_elbow"),
                                                     get_joint_pair(df, "right_wrist"))

  features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                                    get_joint_pair(df, "left_knee"),
                                                    get_joint_pair(df, "left_ankle"))

  features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"),
                                                    get_joint_pair(df, "right_knee"),
                                                    get_joint_pair(df, "right_ankle"))

  features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))

  features_df["hip_dist"] = calculate_distance(get_joint_pair(df, "left_hip"), get_joint_pair(df, "right_hip"))

  return features_df


def get_es5_features(df):
  features_df = pd.DataFrame()

  features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"),
                                                    get_joint_pair(df, "left_knee"),
                                                    get_joint_pair(df, "left_ankle"))

  features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"),
                                                    get_joint_pair(df, "right_knee"),
                                                    get_joint_pair(df, "right_ankle"))

  features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))

  features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))

  features_df["hip_dist"] = calculate_distance(get_joint_pair(df, "left_hip"), get_joint_pair(df, "right_hip"))

  features_df["knee_dist"] = calculate_distance(get_joint_pair(df, "left_knee"), get_joint_pair(df, "right_knee"))

  features_df["ankle_dist"] = calculate_distance(get_joint_pair(df, "left_ankle"), get_joint_pair(df, "right_ankle"))

  features_df["left_shoulder_wrist_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "left_wrist"))

  features_df["right_shoulder_wrist_dist"] = calculate_distance(get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "right_wrist"))

  return features_df


# ── Temporal Statistics (appended as constant features per sample) ────────────

TEMPORAL_STAT_NAMES = [
    'overall_variance',
    'overall_rom',
    'total_displacement',
    'movement_smoothness',
    'active_ratio',
]


def compute_temporal_statistics(features_array):
    """Compute per-sample temporal statistics from a (T, F) feature array.

    Returns a 1D array of 5 scalar values that summarize the temporal
    dynamics of the movement.  These are appended as constant columns
    to every frame so the LSTM has explicit access to motion quality
    information.

    Designed for small-N regression: only 5 extra features to avoid
    curse of dimensionality with ~72 samples.
    """
    features_array = np.asarray(features_array, dtype=np.float64)
    T, F = features_array.shape

    if T < 2:
        return np.zeros(5, dtype=np.float64)

    # 1. Average variance across all feature channels
    per_feature_var = np.var(features_array, axis=0)
    overall_variance = float(np.mean(per_feature_var))

    # 2. Average range of motion (max - min per feature)
    per_feature_rom = np.ptp(features_array, axis=0)
    overall_rom = float(np.mean(per_feature_rom))

    # 3. Total displacement (mean frame-to-frame absolute difference)
    diffs = np.abs(np.diff(features_array, axis=0))
    total_displacement = float(np.mean(diffs))

    # 4. Movement smoothness (mean autocorrelation at lag-1)
    autocorrs = []
    for f in range(F):
        signal = features_array[:, f]
        if np.std(signal) > 1e-8:
            corr = np.corrcoef(signal[:-1], signal[1:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append(corr)
    smoothness = float(np.mean(autocorrs)) if autocorrs else 0.0

    # 5. Active ratio (fraction of frames with above-threshold movement)
    frame_displacements = np.mean(diffs, axis=1)
    threshold = np.median(frame_displacements) * 0.5
    active_ratio = float(np.mean(frame_displacements > threshold))

    return np.array([
        overall_variance,
        overall_rom,
        total_displacement,
        smoothness,
        active_ratio,
    ], dtype=np.float64)


def append_temporal_statistics(features_array):
    """Compute temporal stats and append as constant columns to every frame.

    Input:  (T, F_orig) raw feature array
    Output: (T, F_orig + 5) expanded feature array
    """
    features_array = np.asarray(features_array, dtype=np.float64)
    T = features_array.shape[0]
    stats = compute_temporal_statistics(features_array)
    # Broadcast stats to every frame: (T, 5)
    stats_broadcast = np.tile(stats, (T, 1))
    return np.hstack([features_array, stats_broadcast])