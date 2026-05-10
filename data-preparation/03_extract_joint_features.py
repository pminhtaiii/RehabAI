"""Exercise-specific feature extraction from joint positions.

Implements features selected per-exercise for clinical scoring.
Must produce identical features to backend/joint_features.py.
"""

import os
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter
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


def apply_median_filter(joint_df, window=5):
    filtered = joint_df.copy()
    coord_cols = [c for c in filtered.columns if c.endswith('_x') or c.endswith('_y')]
    for col in coord_cols:
        filtered[col] = scipy_median_filter(filtered[col].values, size=window, mode='nearest')
    return filtered


def calculate_angular_velocity(angle_series):
    """First-order finite difference of an angle series.
    velocity[0] = 0, velocity[t] = angle[t] - angle[t-1].
    Provides temporal dynamics info to LSTM without it having to learn derivatives.
    Source: Capecci 2019 mentions velocity as a motion descriptor;
            Liao et al. 2020 uses joint displacements (temporal diffs).
    """
    vel = np.diff(angle_series, prepend=angle_series.iloc[0] if hasattr(angle_series, 'iloc') else angle_series[0])
    return pd.Series(vel)


# ══════════════════════════════════════════════════════════════════════════════
# Per-Exercise Feature Extractors — Paper 2, Table 2 (optimized subsets)
# MUST MATCH backend/joint_features.py exactly
# ══════════════════════════════════════════════════════════════════════════════

def get_es1_features(df):
    """Es1 — Lifting of arms (10 features).
    Guo&Khan (2021) baseline (6) + shoulder_angle L/R (Capecci 2019 PO α_l/r) +
    wrist_height_ratio (Proietti 2022) + wrist_elevation_velocity (Liao 2020).

    shoulder_angle (hip→shoulder→elbow): PRIMARY OUTCOME per Capecci et al. (2019).
    "angles between right/left arm and upper torso (α_l/r) represent the POs"
    — IEEE TNSRE 27(7) p.1440, Exercise 1. Measures arm-torso ROM directly.

    MUST MATCH backend/joint_features.py exactly.
    """
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
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
    # angle(hip, shoulder, elbow) — arm-to-torso ROM, the main goal of arm lifting.
    # Source: Capecci et al. (2019), IEEE TNSRE 27(7), Exercise 1 PO description.
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

    MUST MATCH backend/joint_features.py exactly.
    """
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    torso_angle = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["torso_tilted_angle"] = torso_angle
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)
    features_df["left_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    # ── NEW: Lateral displacement (Capecci 2019 CF δ) ──
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

    MUST MATCH backend/joint_features.py exactly.
    """
    features_df = pd.DataFrame()
    left_elbow = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"), get_joint_2d(df, "left_wrist"))
    features_df["left_elbow_angle"] = left_elbow
    right_elbow = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"), get_joint_2d(df, "right_wrist"))
    features_df["right_elbow_angle"] = right_elbow
    hands_dist = calculate_distance_2d(get_joint_2d(df, "left_wrist"), get_joint_2d(df, "right_wrist"))
    shoulder_dist = calculate_distance_2d(get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "right_shoulder"))
    features_df["hand_shoulder_ratio"] = hands_dist / (shoulder_dist + 1e-8)
    torso_vec = get_torso_vector_2d(df)
    vertical = np.tile([0, 1], (len(df), 1))
    features_df["torso_tilted_angle"] = calculate_vector_angle_2d(torso_vec, vertical)
    features_df["elbow_angles_diff"] = np.abs(left_elbow - right_elbow)
    features_df["left_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_elbow"))
    features_df["right_shoulder_angle"] = calculate_angle_2d(
        get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_elbow"))
    # ── NEW: Shoulder width ratio (Chen 2021 perspective distortion) ──
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

    left_hip = calculate_angle_2d(
        get_joint_2d(df, "left_shoulder"), get_joint_2d(df, "left_hip"), get_joint_2d(df, "left_knee"))
    features_df["left_hip_angle"] = left_hip

    right_hip = calculate_angle_2d(
        get_joint_2d(df, "right_shoulder"), get_joint_2d(df, "right_hip"), get_joint_2d(df, "right_knee"))
    features_df["right_hip_angle"] = right_hip
    features_df["hip_angles_diff"] = np.abs(left_hip - right_hip)
    features_df["torso_tilted_velocity"] = calculate_angular_velocity(torso_angle)

    return features_df


def get_es5_features(df):
    """Es5 — Squatting (8 features).
    Reduced from 11→8 to fix overfitting (67 samples / 11 features = 6:1 ratio → too low).

    REMOVED (3 features):
    - left/right_elbow_angle: Guo&Khan (2021) Table 2 best model for Ex5 excludes
      left_elbow_angle (ID 0). Arms are held static during squat → near-constant, adds noise.
    - elbow_angles_diff: meaningless without the elbow angles themselves.
    - knee_angle_velocity: uses only left_knee → biased for hemiplegic patients.
      With flip±1° augmentation, LSTM already learns temporal smoothness from sequence.
      Source: Liao et al. (2020), bilateral velocity required to be clinically meaningful.

    KEPT:
    - hand_shoulder_ratio, torso_tilted_angle: arm position + forward lean (Capecci 2019 CF)
    - left/right_shoulder_angle: Guo&Khan (2021) Table 1 Ex5 features
    - left/right_knee_angle: PRIMARY OUTCOME per Capecci 2019 (θ_l/r)
    - knee_ankle_ratio: frontal plane valgus detection (Proietti 2022)

    MUST MATCH backend/joint_features.py exactly.
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

        joint_data = apply_median_filter(joint_data, window=5)

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
