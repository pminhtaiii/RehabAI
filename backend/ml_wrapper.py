# Importing necessary libraries
import os
import json
import pandas as pd
import numpy as np
import joblib
from joint_features import (
    get_es1_features,
    get_es2_features,
    get_es3_features,
    get_es4_features,
    get_es5_features,
)
from motion_detector import detect_motion, calibrate_score, MotionResult


MEDIAPIPE_BODY_KEYPOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def get_dataframe_cols():
    """Return expected column names for MediaPipe body keypoints.

    Each keypoint has 4 channels: x, y, z (3D coordinates) and visibility.
    Total: 12 keypoints × 4 = 48 columns.
    """
    df_cols = []
    for keypoint_name in MEDIAPIPE_BODY_KEYPOINTS:
        df_cols.append(f"{keypoint_name}_x")
        df_cols.append(f"{keypoint_name}_y")
        df_cols.append(f"{keypoint_name}_z")
        df_cols.append(f"{keypoint_name}_v")
    return df_cols


# Masking sentinel — must match training (Masking(mask_value=-999.0)).
# 0.0 is wrong because StandardScaler maps the feature mean to 0.0.
MASK_VALUE = -999.0

# Temporal downsampling — must match training (--downsample 5).
# Training applies stride=5 BEFORE scaling, reducing 25fps→5fps.
# Inference webcam is ~30fps, so stride=6 → 30/6=5fps to match training.
# Reference: KiMoRe dataset recorded at 25fps (Bassi et al., 2021).
DOWNSAMPLE_STRIDE = 5       # Training stride (25fps / 5 = 5fps)
WEBCAM_DOWNSAMPLE_STRIDE = 6  # Webcam stride (30fps / 6 = 5fps) — matches training fps


def temporal_downsample(feat_arr, stride, target_len):
    """Downsample + truncate — identical to training script.

    Keeps every `stride`-th frame, then if still longer than target_len,
    keeps head + tail (discards middle).
    """
    downsampled = feat_arr[::stride]
    T = len(downsampled)
    if T <= target_len:
        return downsampled
    half = target_len // 2
    head = downsampled[:half]
    tail = downsampled[T - (target_len - half):]
    return np.concatenate([head, tail], axis=0)


def trim_to_active_region(features: np.ndarray, min_active_ratio: float = 0.2) -> np.ndarray:
    """Trim idle prefix/suffix from a feature sequence (P3+P4).

    Uses frame-to-frame displacement with EMA smoothing to find
    the first and last active frames, then returns only that region.
    Preserves temporal continuity — does NOT concatenate non-contiguous
    segments, so the LSTM sees a valid time series.

    Args:
        features: (T, F) ndarray of raw features.
        min_active_ratio: Don't trim if active region < this fraction of total.

    Returns:
        Trimmed features array. Same as input if trimming is not applicable.
    """
    T = features.shape[0]
    if T < 10:
        return features

    # Per-frame displacement: RMS across features
    frame_diffs = np.sqrt(np.mean(np.diff(features, axis=0) ** 2, axis=1))

    # EMA smoothing (α=0.3) to reduce MediaPipe jitter
    alpha = 0.3
    smoothed = np.empty_like(frame_diffs)
    smoothed[0] = frame_diffs[0]
    for i in range(1, len(frame_diffs)):
        smoothed[i] = alpha * frame_diffs[i] + (1.0 - alpha) * smoothed[i - 1]

    # Adaptive threshold: median of non-trivial displacements × 0.15
    non_trivial = smoothed[smoothed > 1e-3]
    if len(non_trivial) < 5:
        return features  # Almost entirely static — let hard gate handle it

    threshold = float(np.median(non_trivial) * 0.15)

    # Classify each frame-pair as active/idle
    is_active = smoothed > threshold
    active_indices = np.where(is_active)[0]

    if len(active_indices) == 0:
        return features  # No active frames detected

    # Map diff indices back to feature indices:
    # diff[i] = features[i+1] - features[i], so active diff i → features [i, i+1]
    start_idx = max(0, int(active_indices[0]))
    end_idx = min(T - 1, int(active_indices[-1]) + 1)  # +1 for diff→frame offset
    active_length = end_idx - start_idx + 1

    # Safety: don't trim too aggressively
    if active_length < T * min_active_ratio:
        print(f"[ACTIVE_TRIM] Skip: active region too small "
              f"({active_length}/{T} = {active_length/T:.1%})")
        return features

    # Don't trim if barely anything would be removed
    if active_length >= T * 0.95:
        return features  # Already mostly active, no benefit from trimming

    trimmed = features[start_idx:end_idx + 1]
    print(f"[ACTIVE_TRIM] Trimmed {T} → {trimmed.shape[0]} frames "
          f"[{start_idx}:{end_idx+1}], removed "
          f"{T - trimmed.shape[0]} idle frames ({(T - trimmed.shape[0])/T:.0%})")

    return trimmed


def _auto_scale_score(raw_out):
    """Auto-detect normalization and convert model output to 0-50 score.

    v5 models output [0,1] (trained with y/50) → un-normalize ×50.
    Older models output [0,50] directly → use as-is.
    Threshold: output ≤ 1.5 → normalized; > 1.5 → raw.

    Returns score in [0, 50] — the native clinical scale from KiMoRe.
    """
    if raw_out <= 1.5:
        raw_ts = raw_out * 50.0
    else:
        raw_ts = raw_out
    return float(np.clip(raw_ts, 0, 50))

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


# ---- RF ensemble support ----

_rf_models_cache: dict = {}
_model_configs_cache: dict = {}


def load_rf_model(exercise_id: str):
    if exercise_id in _rf_models_cache:
        return _rf_models_cache[exercise_id]
    path = os.path.join(SCALERS_DIR, f"rf_model_{exercise_id}.joblib")
    if not os.path.exists(path):
        print(f"[ml_wrapper] RF model not found for {exercise_id} at {path} — LSTM-only mode")
        _rf_models_cache[exercise_id] = None
        return None
    rf_model = joblib.load(path)
    _rf_models_cache[exercise_id] = rf_model
    print(f"[ml_wrapper] Loaded RF model for {exercise_id} from {path}")
    return rf_model


def load_model_config(exercise_id: str) -> dict:
    if exercise_id in _model_configs_cache:
        return _model_configs_cache[exercise_id]
    path = os.path.join(SCALERS_DIR, f"model_config_{exercise_id}.json")
    if not os.path.exists(path):
        print(f"[ml_wrapper] Model config not found for {exercise_id} at {path} — using defaults")
        default = {"ensemble_alpha": 0.5}
        _model_configs_cache[exercise_id] = default
        return default
    with open(path, "r") as f:
        config = json.load(f)
    _model_configs_cache[exercise_id] = config
    print(f"[ml_wrapper] Loaded model config for {exercise_id}: alpha={config.get('ensemble_alpha', 0.5)}")
    return config


def extract_summary_features(feat_array: np.ndarray) -> np.ndarray:
    means = np.mean(feat_array, axis=0)
    stds = np.std(feat_array, axis=0)
    ranges = np.max(feat_array, axis=0) - np.min(feat_array, axis=0)
    return np.concatenate([means, stds, ranges]).reshape(1, -1)


def predict_ensemble(lstm_score: float, rf_score: float, alpha: float) -> float:
    return float(np.clip(alpha * lstm_score + (1 - alpha) * rf_score, 0, 50))


FEATURE_EXTRACTORS = {
    "Es1": get_es1_features,
    "Es2": get_es2_features,
    "Es3": get_es3_features,
    "Es4": get_es4_features,
    "Es5": get_es5_features,
}


def _safe_extract_features(df, exercise_id):
    """Extract 2D biomechanical features with NaN safety.
    
    The joint_features.py functions compute 2D angles (atan2(cross, dot))
    and 2D Euclidean distances. NaN inputs will produce NaN outputs,
    so we fill missing values before extraction.
    """
    extractor = FEATURE_EXTRACTORS.get(exercise_id)
    if extractor is None:
        raise ValueError(f"Unsupported exercise_id: {exercise_id}")

    # Fill NaN in input keypoints BEFORE feature extraction
    # Fix: forward-fill then backward-fill instead of fillna(0.0)
    # Reason: (0,0) is top-left corner of frame — filling NaN with 0
    # corrupts angle/distance features (e.g., elbow_angle from origin).
    # Forward-fill preserves temporal continuity; bfill handles leading NaN.
    df_clean = df.ffill().bfill().fillna(0.0)

    # DEBUG: Check input columns and sample values
    body_joints = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                   "left_wrist", "right_wrist", "left_hip", "right_hip"]
    for joint in body_joints[:4]:  # Check first 4 joints
        x_col = f"{joint}_x"
        y_col = f"{joint}_y"
        if x_col in df_clean.columns:
            x_vals = df_clean[x_col].values
            y_vals = df_clean[y_col].values
            x_nonzero = np.count_nonzero(x_vals)
            print(f"[FEAT_DEBUG][{exercise_id}] {joint}: x_nonzero={x_nonzero}/{len(x_vals)}, "
                  f"x_range=[{x_vals.min():.4f}, {x_vals.max():.4f}], "
                  f"y_range=[{y_vals.min():.4f}, {y_vals.max():.4f}]")
        else:
            print(f"[FEAT_DEBUG][{exercise_id}] {joint}: ❌ COLUMN {x_col} NOT FOUND!")

    features_df = extractor(df_clean)
    features = features_df.to_numpy(dtype=np.float32)

    # DEBUG: Check features BEFORE nan_to_num
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    zero_count = (features == 0).sum()
    total = features.size
    print(f"[FEAT_DEBUG][{exercise_id}] Features BEFORE cleanup: "
          f"shape={features.shape}, NaN={nan_count}/{total}, "
          f"Inf={inf_count}/{total}, Zero={zero_count}/{total}")
    if nan_count > 0:
        for i, col_name in enumerate(features_df.columns):
            col_nans = np.isnan(features[:, i]).sum()
            if col_nans > 0:
                print(f"[FEAT_DEBUG][{exercise_id}]   ⚠ {col_name}: {col_nans} NaN values")

    # Replace any remaining NaN/inf from edge cases
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features





def prepare_data(df, max_length, exercise_id):
    """Prepare input data for model inference.

    Pipeline (must match training in clinical_score_prediction_model.py):
      raw keypoints → feature engineering → temporal_downsample(stride=5)
      → StandardScaler → pad(-999) → predict
    """
    df = df.head(1500)  # 30fps × 50s safety margin

    # --- DEBUG: Input DataFrame ---
    print(f"\n[PIPELINE][{exercise_id}] === START INFERENCE ===")
    print(f"[PIPELINE][{exercise_id}] Input: {df.shape[0]} frames, {df.shape[1]} columns")
    non_zero_cols = (df != 0).any().sum()
    nan_count = df.isna().sum().sum()
    print(f"[PIPELINE][{exercise_id}] Non-zero columns: {non_zero_cols}/{df.shape[1]}, NaN cells: {nan_count}")

    # Step 1: Feature engineering (2D angles, distances, ratios)
    features = _safe_extract_features(df, exercise_id)
    print(f"[PIPELINE][{exercise_id}] Features (raw): shape={features.shape}, "
          f"min={features.min():.2f}, max={features.max():.2f}, mean={features.mean():.2f}")
    zero_rows = np.all(features == 0, axis=1).sum()
    print(f"[PIPELINE][{exercise_id}] All-zero feature rows: {zero_rows}/{features.shape[0]}")

    # Step 1.5: Temporal downsampling — MUST match training pipeline
    pre_ds = features.shape[0]
    features = temporal_downsample(features, DOWNSAMPLE_STRIDE, max_length)
    print(f"[PIPELINE][{exercise_id}] Downsample(stride={DOWNSAMPLE_STRIDE}): {pre_ds} → {features.shape[0]} frames")

    # Step 2: StandardScaler (CRITICAL - must match training)
    scaler = load_scaler(exercise_id)
    if scaler.n_features_in_ != features.shape[1]:
        print(f"[PIPELINE][{exercise_id}] WARNING: scaler expects {scaler.n_features_in_} features, got {features.shape[1]}")
    features_scaled = scaler.transform(features).astype(np.float32)
    print(f"[PIPELINE][{exercise_id}] Scaled: min={features_scaled.min():.2f}, "
          f"max={features_scaled.max():.2f}, mean={features_scaled.mean():.4f}")

    # Step 3: Pad/truncate to fixed max_length (no downsampling — Paper 1)
    if features_scaled.shape[0] > max_length:
        features_out = features_scaled[:max_length]
    else:
        pad_len = max_length - features_scaled.shape[0]
        features_out = np.pad(
            features_scaled,
            ((0, pad_len), (0, 0)),
            mode="constant",
            constant_values=MASK_VALUE,
        )
    print(f"[PIPELINE][{exercise_id}] After pad/truncate: {features_scaled.shape[0]} → {features_out.shape[0]} frames")

    data = np.expand_dims(features_out, axis=0)
    data = np.nan_to_num(data, nan=MASK_VALUE)

    # Count real (non-masked) values
    real_pct = 100 * np.sum(data != MASK_VALUE) / data.size
    print(f"[PIPELINE][{exercise_id}] Final tensor: {data.shape}, "
          f"real data: {real_pct:.1f}%, "
          f"range=[{data.min():.3f}, {data.max():.3f}]")

    return data


def prepare_data_with_motion(df, max_length, exercise_id, source="video"):
    df = df.head(1500)

    features = _safe_extract_features(df, exercise_id)

    stride = WEBCAM_DOWNSAMPLE_STRIDE if source == "webcam" else DOWNSAMPLE_STRIDE
    features = temporal_downsample(features, stride, max_length)

    motion_result = detect_motion(features, exercise_id)
    print(f"[MOTION][{exercise_id}] active={motion_result.is_active}, "
          f"energy={motion_result.motion_energy:.3f}, "
          f"quality_factor={motion_result.quality_factor:.3f}, "
          f"variance={motion_result.details.get('avg_variance', 0):.4f}, "
          f"rom={motion_result.details.get('avg_rom', 0):.2f}, "
          f"displacement={motion_result.details.get('avg_displacement', 0):.4f}")

    if motion_result.is_active:
        features = trim_to_active_region(features)

    raw_features_for_rf = features.copy()

    scaler = load_scaler(exercise_id)
    if scaler.n_features_in_ != features.shape[1]:
        print(f"[PIPELINE][{exercise_id}] WARNING: scaler expects {scaler.n_features_in_} features, got {features.shape[1]}")
    features_scaled = scaler.transform(features).astype(np.float32)

    if features_scaled.shape[0] > max_length:
        features_out = features_scaled[:max_length]
    else:
        pad_len = max_length - features_scaled.shape[0]
        features_out = np.pad(
            features_scaled,
            ((0, pad_len), (0, 0)),
            mode="constant",
            constant_values=MASK_VALUE,
        )

    data = np.expand_dims(features_out, axis=0)
    data = np.nan_to_num(data, nan=MASK_VALUE)

    return data, motion_result, raw_features_for_rf


def test_model_inference(model, exercise_id, max_length):
    """Test model with synthetic 3D pose data to verify it produces varied outputs.
    
    Uses MediaPipe-format keypoints (x, y, z, visibility) for all test cases.
    
    Returns dict with test results for diagnostic endpoint.
    """
    cols = get_dataframe_cols()
    results = {}

    # Test 1: All zeros (no movement)
    df_zeros = pd.DataFrame(
        np.zeros((300, len(cols)), dtype=np.float32),
        columns=cols,
    )
    data_zeros, motion_zeros, raw_zeros_rf = prepare_data_with_motion(df_zeros, max_length, exercise_id)
    pred_zeros = model.predict(data_zeros, verbose=0).flatten()
    raw_zeros = _auto_scale_score(pred_zeros[0])
    cal_zeros = calibrate_score(raw_zeros, motion_zeros)
    results["zeros_raw"] = float(pred_zeros[0])
    results["zeros_score"] = raw_zeros
    results["zeros_calibrated"] = cal_zeros
    results["zeros_motion"] = {
        "active": motion_zeros.is_active,
        "energy": motion_zeros.motion_energy,
        "quality_factor": motion_zeros.quality_factor,
    }

    # Test 2: Simulated standing pose (MediaPipe 3D normalized coordinates)
    np.random.seed(42)
    n_frames = 300
    standing = np.zeros((n_frames, len(cols)), dtype=np.float32)
    # Approximate MediaPipe normalized keypoint positions for standing person
    # Format: (x, y, z) where x/y are 0-1 normalized, z is depth relative to hip
    pose_template = {
        "left_shoulder":  (0.40, 0.25, -0.05),
        "right_shoulder": (0.60, 0.25, -0.05),
        "left_elbow":     (0.35, 0.35, -0.03),
        "right_elbow":    (0.65, 0.35, -0.03),
        "left_wrist":     (0.30, 0.45, -0.02),
        "right_wrist":    (0.70, 0.45, -0.02),
        "left_hip":       (0.43, 0.50, -0.01),
        "right_hip":      (0.57, 0.50, -0.01),
        "left_knee":      (0.43, 0.70,  0.00),
        "right_knee":     (0.57, 0.70,  0.00),
        "left_ankle":     (0.43, 0.90,  0.01),
        "right_ankle":    (0.57, 0.90,  0.01),
    }
    for joint, (x, y, z) in pose_template.items():
        x_col = f"{joint}_x"
        y_col = f"{joint}_y"
        z_col = f"{joint}_z"
        v_col = f"{joint}_v"
        if x_col in cols:
            x_idx = cols.index(x_col)
            y_idx = cols.index(y_col)
            z_idx = cols.index(z_col)
            v_idx = cols.index(v_col)
            standing[:, x_idx] = x + np.random.normal(0, 0.002, n_frames)
            standing[:, y_idx] = y + np.random.normal(0, 0.002, n_frames)
            standing[:, z_idx] = z + np.random.normal(0, 0.001, n_frames)
            standing[:, v_idx] = 0.8 + np.random.uniform(0, 0.2, n_frames)

    df_standing = pd.DataFrame(standing, columns=cols)
    data_standing, motion_standing, raw_standing_rf = prepare_data_with_motion(df_standing, max_length, exercise_id)
    pred_standing = model.predict(data_standing, verbose=0).flatten()
    raw_standing = _auto_scale_score(pred_standing[0])
    cal_standing = calibrate_score(raw_standing, motion_standing)
    results["standing_raw"] = float(pred_standing[0])
    results["standing_score"] = raw_standing
    results["standing_calibrated"] = cal_standing
    results["standing_motion"] = {
        "active": motion_standing.is_active,
        "energy": motion_standing.motion_energy,
        "quality_factor": motion_standing.quality_factor,
    }

    # Test 3: Arm-raise exercise simulation (3D movement)
    arm_raise = standing.copy()
    for i in range(n_frames):
        t = i / n_frames
        raise_amount = 0.15 * np.sin(2 * np.pi * t)  # arms go up and down
        lw_y_idx = cols.index("left_wrist_y")
        rw_y_idx = cols.index("right_wrist_y")
        le_y_idx = cols.index("left_elbow_y")
        re_y_idx = cols.index("right_elbow_y")
        # Move in y (vertical) and z (depth) for 3D effect
        arm_raise[i, lw_y_idx] -= raise_amount
        arm_raise[i, rw_y_idx] -= raise_amount
        arm_raise[i, le_y_idx] -= raise_amount * 0.5
        arm_raise[i, re_y_idx] -= raise_amount * 0.5
        # Slight z movement during arm raise
        lw_z_idx = cols.index("left_wrist_z")
        rw_z_idx = cols.index("right_wrist_z")
        arm_raise[i, lw_z_idx] -= raise_amount * 0.3
        arm_raise[i, rw_z_idx] -= raise_amount * 0.3

    df_arm = pd.DataFrame(arm_raise, columns=cols)
    data_arm, motion_arm, raw_arm_rf = prepare_data_with_motion(df_arm, max_length, exercise_id)
    pred_arm = model.predict(data_arm, verbose=0).flatten()
    raw_arm = _auto_scale_score(pred_arm[0])
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
    # Get the correct order of columns (MediaPipe 3D format)
    df_cols = get_dataframe_cols()
    # Reorder the columns of the dataframe; missing columns become NaN
    df = df.reindex(columns=df_cols)
    # Fill any NaN from missing columns
    df = df.fillna(0.0)
    return df
