import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import joblib

FINAL_DATASET_DIR = 'C:/RehabAI/05_final_datasets'

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


# ── Temporal Statistics (must match backend/joint_features.py) ────────────────

TEMPORAL_STAT_NAMES = [
    'overall_variance', 'overall_rom', 'total_displacement',
    'movement_smoothness', 'active_ratio',
]


def compute_temporal_statistics(features_array):
    """5 scalar stats summarizing temporal dynamics of movement."""
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


def append_temporal_stats_to_df(features_df):
    """Compute temporal stats and append as constant columns to DataFrame."""
    arr = features_df.to_numpy(dtype=np.float64)
    T = arr.shape[0]
    stats = compute_temporal_statistics(arr)
    for name, val in zip(TEMPORAL_STAT_NAMES, stats):
        features_df[name] = val
    return features_df


def main():
    input_path = os.path.join(FINAL_DATASET_DIR, 'KiMoRe_final.csv')
    if not os.path.exists(input_path):
        print(f"Dataset not found: {input_path}")
        return
        
    df = pd.read_csv(input_path)

    # 1. First Pass: Extract all raw features and group them by exercise
    print("Extracting raw features (with temporal statistics)...")
    raw_features = {}       # Dictionary to hold dataframes by row index
    exercise_groups = {}    # To group rows by exercise to fit scalers

    for index, row in df.iterrows():
        exercise = row['exercise']
        joint_positions_path = row['joint_positions']

        if pd.isna(joint_positions_path) or not os.path.exists(joint_positions_path):
            continue

        joint_positions_data = pd.read_csv(joint_positions_path)

        if exercise == 'Es1':
            joint_features_data = get_es1_features(joint_positions_data)
        elif exercise == 'Es2':
            joint_features_data = get_es2_features(joint_positions_data)
        elif exercise == 'Es3':
            joint_features_data = get_es3_features(joint_positions_data)
        elif exercise == 'Es4':
            joint_features_data = get_es4_features(joint_positions_data)
        elif exercise == 'Es5':
            joint_features_data = get_es5_features(joint_positions_data)
        else:
            continue

        if joint_features_data is not None:
            # Append temporal statistics as constant columns
            joint_features_data = append_temporal_stats_to_df(joint_features_data)
            raw_features[index] = joint_features_data
            if exercise not in exercise_groups:
                exercise_groups[exercise] = []
            exercise_groups[exercise].append(joint_features_data)

    # 2. Fit Global Scalers for each exercise
    print("Fitting global scalers (including temporal stats)...")
    scalers = {}
    for exercise, df_list in exercise_groups.items():
        # Combine all frames for all videos of this exercise
        all_frames_df = pd.concat(df_list, ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(all_frames_df)
        scalers[exercise] = scaler
        scaler_path = os.path.join(FINAL_DATASET_DIR, f'scaler_{exercise}.joblib')
        joblib.dump(scaler, scaler_path)
        n_orig = all_frames_df.shape[1] - len(TEMPORAL_STAT_NAMES)
        print(f"  - Fitted scaler for {exercise} ({n_orig} orig + "
              f"{len(TEMPORAL_STAT_NAMES)} temporal = {all_frames_df.shape[1]} features) "
              f"→ saved to {scaler_path}")

    # 3. Second Pass: Apply scalers, save to CSV, update master dataframe
    print("Applying scalers and saving features...")
    for index, raw_df in raw_features.items():
        exercise = df.loc[index, 'exercise']
        scaler = scalers[exercise]
        
        # Transform the data and recreate DataFrame with columns
        scaled_data = scaler.transform(raw_df)
        scaled_df = pd.DataFrame(scaled_data, columns=raw_df.columns)
        
        # Save to disk
        joint_positions_path = df.loc[index, 'joint_positions']
        joint_positions_dir = os.path.dirname(joint_positions_path)
        joint_positions_file_name = os.path.basename(joint_positions_path).replace('.csv', '')
        
        joint_features_path = f'{joint_positions_dir}/{joint_positions_file_name}_features.csv'
        scaled_df.to_csv(joint_features_path, index=False)
        
        df.loc[index, 'joint_features'] = joint_features_path

    # Save final metadata
    df = df[['ID', 'exercise', 'video', 'joint_positions', 'joint_features', 'clinical_score', '#frames']]
    output_path = os.path.join(FINAL_DATASET_DIR, 'KiMoRe_data_movenet_features.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved completed dataset to {output_path}")

if __name__ == "__main__":
    main()

