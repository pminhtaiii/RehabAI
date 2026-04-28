"""
Export StandardScaler artifacts WITH temporal statistics (v2).
==============================================================
Run on Colab after mounting Google Drive, or locally.

The scalers now expect F_orig + 5 features (5 temporal stats appended).

Usage:
  Colab:  !python export_scalers_colab.py
  Local:  python export_scalers_colab.py --env local --base-dir C:/RehabAI
"""

import os
import argparse
import pandas as pd
import numpy as np
import math
import joblib
from sklearn.preprocessing import StandardScaler


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Export scalers with temporal stats')
    p.add_argument('--env', choices=['colab', 'local'], default='colab')
    p.add_argument('--base-dir', default=None)
    return p.parse_args()


# ── Feature Engineering (must match backend/joint_features.py) ────────────────

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
    features_df["left_arm_torso_angle"] = calculate_angle(get_joint_pair(df, "left_elbow"), get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "left_hip"))
    features_df["right_arm_torso_angle"] = calculate_angle(get_joint_pair(df, "right_elbow"), get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "right_hip"))
    features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "left_elbow"), get_joint_pair(df, "left_wrist"))
    features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "right_elbow"), get_joint_pair(df, "right_wrist"))
    features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), get_joint_pair(df, "left_knee"), get_joint_pair(df, "left_ankle"))
    features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"), get_joint_pair(df, "right_knee"), get_joint_pair(df, "right_ankle"))
    mid_hip_point = [(get_joint_pair(df, "left_hip")[0] + get_joint_pair(df, "right_hip")[0])/2, (get_joint_pair(df, "left_hip")[1] + get_joint_pair(df, "right_hip")[1])/2]
    features_df["hip_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), mid_hip_point, get_joint_pair(df, "right_hip"))
    features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))
    features_df["ankle_dist"] = calculate_distance(get_joint_pair(df, "left_ankle"), get_joint_pair(df, "right_ankle"))
    return features_df

def get_es2_features(df):
    features_df = pd.DataFrame()
    features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "left_elbow"), get_joint_pair(df, "left_wrist"))
    features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "right_elbow"), get_joint_pair(df, "right_wrist"))
    features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), get_joint_pair(df, "left_knee"), get_joint_pair(df, "left_ankle"))
    features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"), get_joint_pair(df, "right_knee"), get_joint_pair(df, "right_ankle"))
    mid_hip_point = [(get_joint_pair(df, "left_hip")[0] + get_joint_pair(df, "right_hip")[0])/2, (get_joint_pair(df, "left_hip")[1] + get_joint_pair(df, "right_hip")[1])/2]
    features_df["hip_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), mid_hip_point, get_joint_pair(df, "right_hip"))
    features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))
    features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))
    features_df["left_shoulder_wrist_vert_dist"] = np.abs(df["left_shoulder_y"] - df["left_wrist_y"])
    features_df["right_shoulder_wrist_vert_dist"] = np.abs(df["right_shoulder_y"] - df["right_wrist_y"])
    return features_df

def get_es3_features(df):
    features_df = pd.DataFrame()
    features_df["elbows_horiz_dist"] = np.abs(df["left_elbow_x"] - df["right_elbow_x"])
    features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "left_elbow"), get_joint_pair(df, "left_wrist"))
    features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "right_elbow"), get_joint_pair(df, "right_wrist"))
    features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), get_joint_pair(df, "left_knee"), get_joint_pair(df, "left_ankle"))
    features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"), get_joint_pair(df, "right_knee"), get_joint_pair(df, "right_ankle"))
    features_df["left_shoulder_extension_angle"] = calculate_angle(get_joint_pair(df, "left_elbow"), get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))
    features_df["right_shoulder_extension_angle"] = calculate_angle(get_joint_pair(df, "right_elbow"), get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "left_shoulder"))
    mid_hip_point = [(get_joint_pair(df, "left_hip")[0] + get_joint_pair(df, "right_hip")[0])/2, (get_joint_pair(df, "left_hip")[1] + get_joint_pair(df, "right_hip")[1])/2]
    features_df["hip_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), mid_hip_point, get_joint_pair(df, "right_hip"))
    features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))
    features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))
    features_df["hip_dist"] = calculate_distance(get_joint_pair(df, "left_hip"), get_joint_pair(df, "right_hip"))
    features_df["left_shoulder_wrist_vert_dist"] = np.abs(df["left_shoulder_y"] - df["left_wrist_y"])
    features_df["right_shoulder_wrist_vert_dist"] = np.abs(df["right_shoulder_y"] - df["right_wrist_y"])
    return features_df

def get_es4_features(df):
    features_df = pd.DataFrame()
    features_df["left_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "left_elbow"), get_joint_pair(df, "left_wrist"))
    features_df["right_elbow_extension_angle"] = calculate_angle(get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "right_elbow"), get_joint_pair(df, "right_wrist"))
    features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), get_joint_pair(df, "left_knee"), get_joint_pair(df, "left_ankle"))
    features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"), get_joint_pair(df, "right_knee"), get_joint_pair(df, "right_ankle"))
    features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))
    features_df["hip_dist"] = calculate_distance(get_joint_pair(df, "left_hip"), get_joint_pair(df, "right_hip"))
    return features_df

def get_es5_features(df):
    features_df = pd.DataFrame()
    features_df["left_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "left_hip"), get_joint_pair(df, "left_knee"), get_joint_pair(df, "left_ankle"))
    features_df["right_knee_extension_angle"] = calculate_angle(get_joint_pair(df, "right_hip"), get_joint_pair(df, "right_knee"), get_joint_pair(df, "right_ankle"))
    features_df["hands_dist"] = calculate_distance(get_joint_pair(df, "left_wrist"), get_joint_pair(df, "right_wrist"))
    features_df["shoulder_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "right_shoulder"))
    features_df["hip_dist"] = calculate_distance(get_joint_pair(df, "left_hip"), get_joint_pair(df, "right_hip"))
    features_df["knee_dist"] = calculate_distance(get_joint_pair(df, "left_knee"), get_joint_pair(df, "right_knee"))
    features_df["ankle_dist"] = calculate_distance(get_joint_pair(df, "left_ankle"), get_joint_pair(df, "right_ankle"))
    features_df["left_shoulder_wrist_dist"] = calculate_distance(get_joint_pair(df, "left_shoulder"), get_joint_pair(df, "left_wrist"))
    features_df["right_shoulder_wrist_dist"] = calculate_distance(get_joint_pair(df, "right_shoulder"), get_joint_pair(df, "right_wrist"))
    return features_df


FEATURE_EXTRACTORS = {
    "Es1": get_es1_features, "Es2": get_es2_features, "Es3": get_es3_features,
    "Es4": get_es4_features, "Es5": get_es5_features,
}

TEMPORAL_STAT_NAMES = [
    'overall_variance', 'overall_rom', 'total_displacement',
    'movement_smoothness', 'active_ratio',
]


# ── Temporal Statistics (must match backend/joint_features.py) ────────────────

def compute_temporal_statistics(features_array):
    arr = np.asarray(features_array, dtype=np.float64)
    T, F = arr.shape
    if T < 2:
        return np.zeros(5, dtype=np.float64)

    overall_variance = float(np.mean(np.var(arr, axis=0)))
    overall_rom = float(np.mean(np.ptp(arr, axis=0)))
    diffs = np.abs(np.diff(arr, axis=0))
    total_displacement = float(np.mean(diffs))

    autocorrs = []
    for f in range(F):
        sig = arr[:, f]
        if np.std(sig) > 1e-8:
            c = np.corrcoef(sig[:-1], sig[1:])[0, 1]
            if not np.isnan(c):
                autocorrs.append(c)
    smoothness = float(np.mean(autocorrs)) if autocorrs else 0.0

    frame_disp = np.mean(diffs, axis=1)
    active_ratio = float(np.mean(frame_disp > np.median(frame_disp) * 0.5))

    return np.array([overall_variance, overall_rom, total_displacement,
                     smoothness, active_ratio], dtype=np.float64)


def append_temporal_stats(features_array):
    arr = np.asarray(features_array, dtype=np.float64)
    T = arr.shape[0]
    stats = compute_temporal_statistics(arr)
    return np.hstack([arr, np.tile(stats, (T, 1))])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.base_dir:
        base = args.base_dir
    elif args.env == 'colab':
        base = '/content/drive/MyDrive/RehabAI'
    else:
        base = 'C:/RehabAI'

    csv_path = f'{base}/05_final_datasets/KiMoRe_final.csv'
    save_dir = f'{base}/models/best_models'

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Group raw features by exercise
    exercise_groups = {}
    for _, row in df.iterrows():
        exercise = row['exercise']
        jp_path = row['joint_positions']
        if pd.isna(jp_path) or not os.path.exists(jp_path):
            continue
        extractor = FEATURE_EXTRACTORS.get(exercise)
        if not extractor:
            continue
        joint_data = pd.read_csv(jp_path)
        raw_features = extractor(joint_data)
        # Append temporal statistics
        raw_arr = raw_features.to_numpy(dtype=np.float64)
        expanded = append_temporal_stats(raw_arr)
        col_names = list(raw_features.columns) + TEMPORAL_STAT_NAMES
        expanded_df = pd.DataFrame(expanded, columns=col_names)

        if exercise not in exercise_groups:
            exercise_groups[exercise] = []
        exercise_groups[exercise].append(expanded_df)

    # Fit and save scalers
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nFitting scalers (with temporal stats) → {save_dir}")

    for exercise, df_list in exercise_groups.items():
        all_frames = pd.concat(df_list, ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(all_frames)

        scaler_path = os.path.join(save_dir, f"scaler_{exercise}.joblib")
        joblib.dump(scaler, scaler_path)

        n_orig = all_frames.shape[1] - len(TEMPORAL_STAT_NAMES)
        print(f"  ✓ {exercise}: {len(df_list)} videos, "
              f"{n_orig} orig + {len(TEMPORAL_STAT_NAMES)} temporal = "
              f"{all_frames.shape[1]} features")
        print(f"    mean range: [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")
        print(f"    std range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
        print(f"    saved to: {scaler_path}")

    print(f"\n✅ Done! Copy scaler_EsX.joblib files to your project's models/ directory.")


if __name__ == '__main__':
    main()
