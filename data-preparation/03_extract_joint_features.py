"""
03_extract_joint_features.py — Paper-Defined Exercise-Specific Feature Extraction
===================================================================================
Implements exercise-specific features from Guo & Khan (2021) "Exercise-Specific
Feature Extraction Approach for Assessing Physical Rehabilitation" (Paper 2),
as referenced by arXiv 2306.09546 (Paper 1).

Features are selected per-exercise using Table 2 (optimized subsets) from Paper 2:
  Es1: 6 features (IDs 0,1,2,3,4,5)
  Es2: 6 features (IDs 0,1,3,5,6,7)
  Es3: 9 features (IDs 0,1,2,3,5,6,7,8,9)
  Es4: 2 features (IDs 3,10)
  Es5: 7 features (IDs 0,1,2,3,5,6,7)

CRITICAL: This file's feature extraction functions MUST produce identical features
to backend/joint_features.py so that training and inference see the same data.

Usage:
  Colab:  Change BASE_DIR to '/content/drive/MyDrive/RehabAI' and run
  Local:  python 03_extract_joint_features.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = 'C:/RehabAI'
FINAL_DATASET_DIR = f'{BASE_DIR}/05_final_datasets'


# ══════════════════════════════════════════════════════════════════════════════
# 2D Geometry Functions (Aspect Ratio Corrected)
# MUST MATCH backend/joint_features.py exactly
# ══════════════════════════════════════════════════════════════════════════════

def get_joint_2d(df, joint_name):
    """Return (N, 2) array of [x, y] with aspect ratio corrected (x * 1.7778).
    MediaPipe normalizes x and y to [0, 1]. For a 16:9 video, x is stretched.
    Multiplying x by 1920/1080 (1.7778) makes the space isotropic for angle math.
    """
    return np.column_stack((
        df[f"{joint_name}_x"] * 1.7778,
        df[f"{joint_name}_y"]
    ))


def calculate_angle_2d(first, middle, end):
    """Angle (degrees) at vertex *middle* in 2D. Uses atan2(cross, dot).
    Returns absolute angle [0, 180].
    """
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
    """Torso vector from shoulder midpoint to hip midpoint (N, 2).
    Points roughly downward (y increases downward).
    """
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
# Per-Exercise Feature Extractors — Paper 2, Table 2 (optimized subsets)
# MUST MATCH backend/joint_features.py exactly
# ══════════════════════════════════════════════════════════════════════════════

def get_es1_features(df):
    """Es1 — Lifting of arms. Table 2: features 0,1,2,3,4,5 (6 features)."""
    features_df = pd.DataFrame()

    # Feature 0: left_elbow_angle — elbow extension angle at left elbow
    left_elbow = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow

    # Feature 1: right_elbow_angle — elbow extension angle at right elbow
    right_elbow = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow

    # Feature 2: hand_shoulder_ratio — hands distance / shoulder width
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)

    # Feature 3: torso_tilted_angle — angle between torso vector and vertical
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)

    # Feature 4: hand_tilted_angle — angle between hand-to-hand vector and horizontal
    hand_vec = get_joint_2d(df, "right_wrist") - get_joint_2d(df, "left_wrist")
    horizontal = np.tile([1, 0], (len(df), 1))
    features_df["hand_tilted_angle"] = calculate_vector_angle_2d(hand_vec, horizontal)

    # Feature 5: elbow_angles_diff — |left_elbow_angle - right_elbow_angle|
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)

    return features_df


def get_es2_features(df):
    """Es2 — Lateral tilt of trunk. Table 2: features 0,1,3,5,6,7 (6 features)."""
    features_df = pd.DataFrame()

    # Feature 0: left_elbow_angle
    left_elbow = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow

    # Feature 1: right_elbow_angle
    right_elbow = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow

    # Feature 3: torso_tilted_angle
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)

    # Feature 5: elbow_angles_diff
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)

    # Feature 6: left_shoulder_angle — shoulder elevation (hip→shoulder→elbow)
    features_df["left_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))

    # Feature 7: right_shoulder_angle — shoulder elevation (hip→shoulder→elbow)
    features_df["right_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))

    return features_df


def get_es3_features(df):
    """Es3 — Trunk rotation. Table 2: features 0,1,2,3,5,6,7,8,9 (9 features)."""
    features_df = pd.DataFrame()

    # Feature 0: left_elbow_angle
    left_elbow = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow

    # Feature 1: right_elbow_angle
    right_elbow = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow

    # Feature 2: hand_shoulder_ratio
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)

    # Feature 3: torso_tilted_angle
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)

    # Feature 5: elbow_angles_diff
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)

    # Feature 6: left_shoulder_angle
    features_df["left_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))

    # Feature 7: right_shoulder_angle
    features_df["right_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))

    # Feature 8: left_arm_torso_angle — angle between torso vector and left upper arm
    left_arm_vec = get_joint_2d(df, "left_elbow") - get_joint_2d(df, "left_shoulder")
    features_df["left_arm_torso_angle"] = calculate_vector_angle_2d(torso_vec, left_arm_vec)

    # Feature 9: right_arm_torso_angle — angle between torso vector and right upper arm
    right_arm_vec = get_joint_2d(df, "right_elbow") - get_joint_2d(df, "right_shoulder")
    features_df["right_arm_torso_angle"] = calculate_vector_angle_2d(torso_vec, right_arm_vec)

    return features_df


def get_es4_features(df):
    """Es4 — Pelvis rotation. Table 2: features 3,10 (2 features)."""
    features_df = pd.DataFrame()

    # Feature 3: torso_tilted_angle
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)

    # Feature 10: knee_hip_ratio — knee distance / hip width
    knee_dist = calculate_distance_2d(get_joint_2d(df, "left_knee"), get_joint_2d(df, "right_knee"))
    hip_dist = calculate_distance_2d(get_joint_2d(df, "left_hip"), get_joint_2d(df, "right_hip"))
    features_df["knee_hip_ratio"] = knee_dist / (hip_dist + 1e-8)

    return features_df


def get_es5_features(df):
    """Es5 — Squatting. Table 2: features 0,1,2,3,5,6,7 (7 features)."""
    features_df = pd.DataFrame()

    # Feature 0: left_elbow_angle
    left_elbow = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow

    # Feature 1: right_elbow_angle
    right_elbow = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow

    # Feature 2: hand_shoulder_ratio
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)

    # Feature 3: torso_tilted_angle
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)

    # Feature 5: elbow_angles_diff
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)

    # Feature 6: left_shoulder_angle
    features_df["left_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))

    # Feature 7: right_shoulder_angle
    features_df["right_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))

    return features_df


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_EXTRACTORS = {
    'Es1': get_es1_features,
    'Es2': get_es2_features,
    'Es3': get_es3_features,
    'Es4': get_es4_features,
    'Es5': get_es5_features,
}


def main():
    input_path = os.path.join(FINAL_DATASET_DIR, 'KiMoRe_final_mediapipe.csv')
    if not os.path.exists(input_path):
        print(f"Dataset not found: {input_path}")
        print("  Run 02_prepare_dataset.py first.")
        return

    df = pd.read_csv(input_path)

    print("=" * 60)
    print("  03_extract_joint_features.py — Paper-Specific Features")
    print("  Source: Guo & Khan (2021), Table 2 optimized subsets")
    print("=" * 60)
    print(f"  Input: {input_path} ({len(df)} rows)")
    print(f"  Feature mode: MediaPipe 3D → 2D Aspect Ratio Corrected")
    print()

    # ── Pass 1: Extract all raw features, group by exercise ──────────────────
    print("Pass 1: Extracting paper-defined features...")
    raw_features = {}
    exercise_groups = {}
    skip_count = 0

    for index, row in df.iterrows():
        exercise = row['exercise']
        joint_positions_path = row['joint_positions']

        if pd.isna(joint_positions_path) or not os.path.exists(str(joint_positions_path)):
            skip_count += 1
            continue

        extractor = FEATURE_EXTRACTORS.get(exercise)
        if extractor is None:
            skip_count += 1
            continue

        try:
            joint_data = pd.read_csv(joint_positions_path)
        except Exception as e:
            print(f"  ⚠ Error reading {joint_positions_path}: {e}")
            skip_count += 1
            continue

        # Extract features
        features_df = extractor(joint_data)

        # Handle NaN/inf from edge cases (e.g., zero-length vectors)
        features_df = features_df.fillna(0.0)
        features_df = features_df.replace([np.inf, -np.inf], 0.0)

        raw_features[index] = features_df
        if exercise not in exercise_groups:
            exercise_groups[exercise] = []
        exercise_groups[exercise].append(features_df)

    print(f"  Extracted: {len(raw_features)} samples, skipped: {skip_count}")

    # ── Pass 2: Fit global scalers per exercise ──────────────────────────────
    print("\nPass 2: Fitting global StandardScalers (per exercise)...")
    scalers = {}
    for exercise in sorted(exercise_groups.keys()):
        df_list = exercise_groups[exercise]
        all_frames = pd.concat(df_list, ignore_index=True)

        # Replace any residual NaN/inf before fitting
        all_frames = all_frames.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        scaler = StandardScaler()
        scaler.fit(all_frames)
        scalers[exercise] = scaler

        scaler_path = os.path.join(FINAL_DATASET_DIR, f'scaler_{exercise}.joblib')
        joblib.dump(scaler, scaler_path)

        n_feat = all_frames.shape[1]
        print(f"  {exercise}: {len(df_list)} videos, "
              f"{n_feat} features → {scaler_path}")

    # ── Pass 3: Save RAW feature CSVs (unscaled) ────────────────────────────
    # Scaling is deferred to training time / inference time for consistency.
    # The scaler artifacts saved in Pass 2 are used by both pipelines.
    print("\nPass 3: Saving RAW (unscaled) feature CSVs...")
    df['joint_features'] = np.nan

    for index, raw_df in raw_features.items():
        # Save raw features (no scaler applied)
        clean_df = raw_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Save
        joint_positions_path = df.loc[index, 'joint_positions']
        base_dir = os.path.dirname(joint_positions_path)
        base_name = os.path.basename(joint_positions_path).replace('.csv', '')
        features_path = os.path.join(base_dir, f'{base_name}_features.csv')
        clean_df.to_csv(features_path, index=False)

        df.loc[index, 'joint_features'] = features_path

    # ── Save final metadata ──────────────────────────────────────────────────
    output_columns = [
        'ID', 'clinical_group', 'expertise', 'exercise',
        'video', 'joint_positions', 'joint_features',
        'clinical_score', '#frames',
    ]
    # Ensure all columns exist
    for col in output_columns:
        if col not in df.columns:
            df[col] = np.nan
    df_out = df[output_columns]

    output_path = os.path.join(FINAL_DATASET_DIR, 'KiMoRe_data_movenet_features.csv')
    df_out.to_csv(output_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = df_out['joint_features'].notna().sum()
    print(f"\n{'=' * 60}")
    print(f"  Summary:")
    print(f"    Samples with features: {valid}/{len(df_out)}")
    print(f"    Scaler artifacts: {len(scalers)} (one per exercise)")
    print(f"    Output: {output_path}")

    # Feature count per exercise
    for ex, scaler in sorted(scalers.items()):
        print(f"    {ex}: scaler.n_features_in_ = {scaler.n_features_in_}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
