# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import joblib
from joint_features import (
    get_es1_features,
    get_es2_features,
    get_es3_features,
    get_es4_features,
    get_es5_features,
    append_temporal_statistics,
)
from motion_detector import detect_motion, calibrate_score, MotionResult


# Function to get dataframe columns
def get_dataframe_cols():
    # Define a dictionary with keypoints and their corresponding indices
    KEYPOINT_DICT = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }
    # Initialize an empty list to store the column names for the dataframe
    df_cols = []
    # Iterate over the keypoint names in the dictionary
    for keypoint_name in KEYPOINT_DICT:
        # For each keypoint, append three columns to the dataframe: y-coordinate, x-coordinate, and confidence
        df_cols.append(f"{keypoint_name}_y")
        df_cols.append(f"{keypoint_name}_x")
        df_cols.append(f"{keypoint_name}_confidence")
    return df_cols


DOWNSAMPLE_FACTOR = 5

SCALERS_DIR = os.environ.get("SCALERS_DIR", "models/")
_scalers_cache: dict = {}


def load_scaler(exercise_id: str):
    """Load the per-exercise StandardScaler fitted during training.
    Scalers are cached after first load."""
    if exercise_id in _scalers_cache:
        return _scalers_cache[exercise_id]
    path = os.path.join(SCALERS_DIR, f"scaler_{exercise_id}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found for {exercise_id} at {path}. "
            f"Run 03_extract_joint_features.py to generate scaler artifacts."
        )
    scaler = joblib.load(path)
    _scalers_cache[exercise_id] = scaler
    print(f"[ml_wrapper] Loaded scaler for {exercise_id} from {path} "
          f"(features={scaler.n_features_in_})")
    return scaler


FEATURE_EXTRACTORS = {
    "Es1": get_es1_features,
    "Es2": get_es2_features,
    "Es3": get_es3_features,
    "Es4": get_es4_features,
    "Es5": get_es5_features,
}


def _safe_extract_features(df, exercise_id):
    """Extract biomechanical features with NaN safety.
    
    The joint_features.py functions can produce NaN when keypoint
    coordinates are NaN (camera doesn't see body part). We replace
    NaN with 0 before returning so downstream StandardScaler works.
    """
    extractor = FEATURE_EXTRACTORS.get(exercise_id)
    if extractor is None:
        raise ValueError(f"Unsupported exercise_id: {exercise_id}")

    # Fill NaN in input keypoints BEFORE feature extraction
    # This prevents NaN in arctan2 calculations (joint_features.py:15)
    df_clean = df.fillna(0.0)

    features_df = extractor(df_clean)
    features = features_df.to_numpy(dtype=np.float32)

    # Replace any remaining NaN/inf from edge cases
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def _maybe_append_temporal_stats(features, exercise_id, scaler):
    """Auto-detect if scaler expects temporal stats and append if needed.

    Backward compatible: old scalers (F features) work without stats,
    new scalers (F+5 features) get temporal stats automatically.
    """
    expected = scaler.n_features_in_
    actual = features.shape[1]
    if expected == actual + 5:
        # New scaler: append temporal statistics
        features = append_temporal_statistics(features)
        print(f"[PIPELINE][{exercise_id}] Appended temporal stats: {actual} → {features.shape[1]} features")
    elif expected != actual:
        print(f"[PIPELINE][{exercise_id}] WARNING: scaler expects {expected} features, got {actual}")
    return features


def prepare_data(df, max_length, exercise_id):
    """Prepare input data for model inference.
    
    Pipeline: raw keypoints → feature engineering → [temporal stats] → StandardScaler → downsample → pad
    Must match training pipeline in 03_extract_joint_features.py exactly.
    """
    df = df.head(600)

    # --- DEBUG: Input DataFrame ---
    print(f"\n[PIPELINE][{exercise_id}] === START INFERENCE ===")
    print(f"[PIPELINE][{exercise_id}] Input: {df.shape[0]} frames, {df.shape[1]} columns")
    non_zero_cols = (df != 0).any().sum()
    nan_count = df.isna().sum().sum()
    print(f"[PIPELINE][{exercise_id}] Non-zero columns: {non_zero_cols}/{df.shape[1]}, NaN cells: {nan_count}")

    # Step 1: Feature engineering (angles, distances)
    features = _safe_extract_features(df, exercise_id)
    print(f"[PIPELINE][{exercise_id}] Features: shape={features.shape}, "
          f"min={features.min():.2f}, max={features.max():.2f}, mean={features.mean():.2f}")
    zero_rows = np.all(features == 0, axis=1).sum()
    print(f"[PIPELINE][{exercise_id}] All-zero feature rows: {zero_rows}/{features.shape[0]}")

    # Step 1.5: Temporal statistics (auto-detected from scaler)
    scaler = load_scaler(exercise_id)
    features = _maybe_append_temporal_stats(features, exercise_id, scaler)

    # Step 2: StandardScaler (CRITICAL - must match training)
    features_scaled = scaler.transform(features).astype(np.float32)
    print(f"[PIPELINE][{exercise_id}] Scaled: min={features_scaled.min():.2f}, "
          f"max={features_scaled.max():.2f}, mean={features_scaled.mean():.4f}")

    # Step 3: Downsample (every 5th frame, matching training)
    features_ds = features_scaled[::DOWNSAMPLE_FACTOR]
    print(f"[PIPELINE][{exercise_id}] Downsampled: {features_scaled.shape[0]} → {features_ds.shape[0]} frames")

    # Step 4: Pad/truncate to fixed max_length
    if features_ds.shape[0] > max_length:
        features_ds = features_ds[:max_length]
    else:
        pad_len = max_length - features_ds.shape[0]
        features_ds = np.pad(
            features_ds,
            ((0, pad_len), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    data = np.expand_dims(features_ds, axis=0)
    data = np.nan_to_num(data)

    non_zero_pct = 100 * np.count_nonzero(data) / data.size
    print(f"[PIPELINE][{exercise_id}] Final tensor: {data.shape}, "
          f"non-zero: {non_zero_pct:.1f}%, "
          f"range=[{data.min():.3f}, {data.max():.3f}]")

    return data


def prepare_data_with_motion(df, max_length, exercise_id):
    """Prepare input data AND run motion detection on raw features.

    Returns:
        tuple: (prepared_data, motion_result)
            - prepared_data: np.ndarray ready for model.predict()
            - motion_result: MotionResult from motion_detector
    """
    df = df.head(600)

    # Step 1: Feature engineering (angles, distances)
    features = _safe_extract_features(df, exercise_id)

    # Step 1.5: Motion detection on RAW features (before scaling)
    # This is the key — analyze the actual biomechanical signals for movement
    motion_result = detect_motion(features, exercise_id)
    print(f"[MOTION][{exercise_id}] active={motion_result.is_active}, "
          f"energy={motion_result.motion_energy:.3f}, "
          f"quality_factor={motion_result.quality_factor:.3f}, "
          f"variance={motion_result.details.get('avg_variance', 0):.4f}, "
          f"rom={motion_result.details.get('avg_rom', 0):.2f}, "
          f"displacement={motion_result.details.get('avg_displacement', 0):.4f}")

    # Step 1.7: Temporal statistics (auto-detected from scaler)
    scaler = load_scaler(exercise_id)
    features = _maybe_append_temporal_stats(features, exercise_id, scaler)

    # Step 2: StandardScaler
    features_scaled = scaler.transform(features).astype(np.float32)

    # Step 3: Downsample
    features_ds = features_scaled[::DOWNSAMPLE_FACTOR]

    # Step 4: Pad/truncate
    if features_ds.shape[0] > max_length:
        features_ds = features_ds[:max_length]
    else:
        pad_len = max_length - features_ds.shape[0]
        features_ds = np.pad(
            features_ds,
            ((0, pad_len), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    data = np.expand_dims(features_ds, axis=0)
    data = np.nan_to_num(data)

    return data, motion_result


def test_model_inference(model, exercise_id, max_length):
    """Test model with synthetic data to verify it produces varied outputs.
    
    Now includes motion detection and calibrated scores to show the
    effect of the motion gate on different input types.
    
    Returns dict with test results for diagnostic endpoint.
    """
    cols = get_dataframe_cols()
    results = {}

    # Test 1: All zeros (no movement)
    df_zeros = pd.DataFrame(
        np.zeros((300, len(cols)), dtype=np.float32),
        columns=cols,
    )
    data_zeros, motion_zeros = prepare_data_with_motion(df_zeros, max_length, exercise_id)
    pred_zeros = model.predict(data_zeros, verbose=0).flatten()
    raw_zeros = float(pred_zeros[0] * 100)
    cal_zeros = calibrate_score(raw_zeros, motion_zeros)
    results["zeros_raw"] = float(pred_zeros[0])
    results["zeros_score"] = raw_zeros
    results["zeros_calibrated"] = cal_zeros
    results["zeros_motion"] = {
        "active": motion_zeros.is_active,
        "energy": motion_zeros.motion_energy,
        "quality_factor": motion_zeros.quality_factor,
    }

    # Test 2: Simulated standing pose (realistic normalized coordinates)
    np.random.seed(42)
    n_frames = 300
    standing = np.zeros((n_frames, len(cols)), dtype=np.float32)
    # Approximate normalized keypoint positions for standing person
    pose_template = {
        "nose": (0.15, 0.50), "left_eye": (0.14, 0.48), "right_eye": (0.14, 0.52),
        "left_ear": (0.15, 0.46), "right_ear": (0.15, 0.54),
        "left_shoulder": (0.25, 0.40), "right_shoulder": (0.25, 0.60),
        "left_elbow": (0.35, 0.35), "right_elbow": (0.35, 0.65),
        "left_wrist": (0.45, 0.30), "right_wrist": (0.45, 0.70),
        "left_hip": (0.50, 0.43), "right_hip": (0.50, 0.57),
        "left_knee": (0.70, 0.43), "right_knee": (0.70, 0.57),
        "left_ankle": (0.90, 0.43), "right_ankle": (0.90, 0.57),
    }
    for joint, (y, x) in pose_template.items():
        y_col = f"{joint}_y"
        x_col = f"{joint}_x"
        conf_col = f"{joint}_confidence"
        if y_col in cols and x_col in cols:
            y_idx = cols.index(y_col)
            x_idx = cols.index(x_col)
            c_idx = cols.index(conf_col)
            # Add slight noise for realistic movement
            standing[:, y_idx] = y + np.random.normal(0, 0.002, n_frames)
            standing[:, x_idx] = x + np.random.normal(0, 0.002, n_frames)
            standing[:, c_idx] = 0.8 + np.random.uniform(0, 0.2, n_frames)

    df_standing = pd.DataFrame(standing, columns=cols)
    data_standing, motion_standing = prepare_data_with_motion(df_standing, max_length, exercise_id)
    pred_standing = model.predict(data_standing, verbose=0).flatten()
    raw_standing = float(pred_standing[0] * 100)
    cal_standing = calibrate_score(raw_standing, motion_standing)
    results["standing_raw"] = float(pred_standing[0])
    results["standing_score"] = raw_standing
    results["standing_calibrated"] = cal_standing
    results["standing_motion"] = {
        "active": motion_standing.is_active,
        "energy": motion_standing.motion_energy,
        "quality_factor": motion_standing.quality_factor,
    }

    # Test 3: Arm-raise exercise simulation
    arm_raise = standing.copy()
    for i in range(n_frames):
        t = i / n_frames
        raise_amount = 0.15 * np.sin(2 * np.pi * t)  # arms go up and down
        lw_y_idx = cols.index("left_wrist_y")
        rw_y_idx = cols.index("right_wrist_y")
        le_y_idx = cols.index("left_elbow_y")
        re_y_idx = cols.index("right_elbow_y")
        arm_raise[i, lw_y_idx] -= raise_amount
        arm_raise[i, rw_y_idx] -= raise_amount
        arm_raise[i, le_y_idx] -= raise_amount * 0.5
        arm_raise[i, re_y_idx] -= raise_amount * 0.5

    df_arm = pd.DataFrame(arm_raise, columns=cols)
    data_arm, motion_arm = prepare_data_with_motion(df_arm, max_length, exercise_id)
    pred_arm = model.predict(data_arm, verbose=0).flatten()
    raw_arm = float(pred_arm[0] * 100)
    cal_arm = calibrate_score(raw_arm, motion_arm)
    results["arm_raise_raw"] = float(pred_arm[0])
    results["arm_raise_score"] = raw_arm
    results["arm_raise_calibrated"] = cal_arm
    results["arm_raise_motion"] = {
        "active": motion_arm.is_active,
        "energy": motion_arm.motion_energy,
        "quality_factor": motion_arm.quality_factor,
    }

    # Analysis — use CALIBRATED scores for the comparison
    raw_scores = [raw_zeros, raw_standing, raw_arm]
    cal_scores = [cal_zeros, cal_standing, cal_arm]
    results["score_range"] = max(raw_scores) - min(raw_scores)
    results["calibrated_range"] = max(cal_scores) - min(cal_scores)
    results["model_is_responsive"] = results["score_range"] > 2.0
    results["all_scores"] = {
        "no_movement": round(raw_zeros, 1),
        "standing_still": round(raw_standing, 1),
        "arm_raise": round(raw_arm, 1),
    }
    results["calibrated_scores"] = {
        "no_movement": round(cal_zeros, 1),
        "standing_still": round(cal_standing, 1),
        "arm_raise": round(cal_arm, 1),
    }

    return results


# Function to reorder the columns of the dataframe
def reorder_dataframe(df):
    # Get the correct order of columns
    df_cols = get_dataframe_cols()
    # Reorder the columns of the dataframe; missing columns become NaN
    df = df.reindex(columns=df_cols)
    # Fill any NaN from missing columns
    df = df.fillna(0.0)
    return df
