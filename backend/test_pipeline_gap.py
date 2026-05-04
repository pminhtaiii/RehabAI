"""
Test the EXACT pipeline gap between synthetic diagnostic data vs real webcam data.
Simulates what really happens when frontend sends webcam CSV.

Run: python backend/test_pipeline_gap.py
Requires: tensorflow, joblib, numpy, pandas (run inside Docker or local venv)
"""
import os
import sys
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from ml_wrapper import (
    get_dataframe_cols, reorder_dataframe, prepare_data, load_scaler,
    MEDIAPIPE_BODY_KEYPOINTS, MASK_VALUE
)
from joint_features import get_es1_features


def simulate_frontend_csv(n_frames=150, movement_type="standing"):
    """Simulate EXACTLY what the frontend MiniDataFrame.to_csv() produces.
    
    Frontend sends: {joint}_x, {joint}_y, {joint}_confidence  (17 joints × 3 = 51 cols)
    Backend expects: {joint}_x, {joint}_y, {joint}_z, {joint}_v  (12 joints × 4 = 48 cols)
    """
    # 17 MoveNet keypoints (same as frontend KEYPOINT_DICT)
    FRONTEND_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    data = {}
    for kp in FRONTEND_KEYPOINTS:
        if movement_type == "standing":
            # Realistic standing pose in normalized [0,1] coordinates
            base_positions = {
                'nose': (0.5, 0.15),
                'left_eye': (0.48, 0.13), 'right_eye': (0.52, 0.13),
                'left_ear': (0.46, 0.14), 'right_ear': (0.54, 0.14),
                'left_shoulder': (0.42, 0.28), 'right_shoulder': (0.58, 0.28),
                'left_elbow': (0.38, 0.42), 'right_elbow': (0.62, 0.42),
                'left_wrist': (0.36, 0.55), 'right_wrist': (0.64, 0.55),
                'left_hip': (0.45, 0.55), 'right_hip': (0.55, 0.55),
                'left_knee': (0.44, 0.72), 'right_knee': (0.56, 0.72),
                'left_ankle': (0.43, 0.88), 'right_ankle': (0.57, 0.88),
            }
            bx, by = base_positions.get(kp, (0.5, 0.5))
            # Add realistic webcam jitter
            data[f'{kp}_x'] = np.clip(bx + np.random.normal(0, 0.005, n_frames), 0, 1)
            data[f'{kp}_y'] = np.clip(by + np.random.normal(0, 0.005, n_frames), 0, 1)
            data[f'{kp}_confidence'] = np.random.uniform(0.7, 0.99, n_frames)
        
        elif movement_type == "arm_raise":
            base_positions = {
                'nose': (0.5, 0.15),
                'left_eye': (0.48, 0.13), 'right_eye': (0.52, 0.13),
                'left_ear': (0.46, 0.14), 'right_ear': (0.54, 0.14),
                'left_shoulder': (0.42, 0.28), 'right_shoulder': (0.58, 0.28),
                'left_hip': (0.45, 0.55), 'right_hip': (0.55, 0.55),
                'left_knee': (0.44, 0.72), 'right_knee': (0.56, 0.72),
                'left_ankle': (0.43, 0.88), 'right_ankle': (0.57, 0.88),
            }
            bx, by = base_positions.get(kp, (0.5, 0.5))
            
            # Arms go up over time (simulating Es1)
            t = np.linspace(0, 2 * np.pi, n_frames)
            if kp in ['left_elbow', 'right_elbow']:
                bx_base = 0.38 if 'left' in kp else 0.62
                bx_arr = bx_base + np.random.normal(0, 0.005, n_frames)
                by_arr = 0.28 + 0.14 * np.sin(t) + np.random.normal(0, 0.005, n_frames)  # goes UP (y decreases)
                data[f'{kp}_x'] = np.clip(bx_arr, 0, 1)
                data[f'{kp}_y'] = np.clip(by_arr, 0, 1)
            elif kp in ['left_wrist', 'right_wrist']:
                bx_base = 0.36 if 'left' in kp else 0.64
                bx_arr = bx_base + np.random.normal(0, 0.005, n_frames)
                by_arr = 0.55 - 0.35 * np.abs(np.sin(t)) + np.random.normal(0, 0.005, n_frames)  # arms raise up
                data[f'{kp}_x'] = np.clip(bx_arr, 0, 1)
                data[f'{kp}_y'] = np.clip(by_arr, 0, 1)
            else:
                data[f'{kp}_x'] = np.clip(bx + np.random.normal(0, 0.005, n_frames), 0, 1)
                data[f'{kp}_y'] = np.clip(by + np.random.normal(0, 0.005, n_frames), 0, 1)
            
            data[f'{kp}_confidence'] = np.random.uniform(0.7, 0.99, n_frames)
        
        elif movement_type == "no_movement":
            data[f'{kp}_x'] = np.zeros(n_frames)
            data[f'{kp}_y'] = np.zeros(n_frames)
            data[f'{kp}_confidence'] = np.zeros(n_frames)
    
    return pd.DataFrame(data)


def main():
    exercise_id = "Es1"
    max_length = 150
    
    print("=" * 70)
    print("  Pipeline Gap Analysis: Frontend CSV → Model Prediction")
    print("=" * 70)
    
    # Load model
    import tensorflow as tf
    model_path = "models/ml_model_Es1.keras"
    if not os.path.exists(model_path):
        print(f"  Model not found at {model_path}")
        return
    model = tf.keras.models.load_model(model_path)
    print(f"  Model loaded: {model_path}")
    print(f"  Model input shape: {model.input_shape}")
    
    test_cases = ["no_movement", "standing", "arm_raise"]
    
    for movement in test_cases:
        print(f"\n{'─' * 70}")
        print(f"  Test: {movement} (simulated webcam)")
        print(f"{'─' * 70}")
        
        # 1. Simulate frontend CSV (exactly like MiniDataFrame.to_csv())
        frontend_df = simulate_frontend_csv(n_frames=900, movement_type=movement)  # 30fps * 30s
        csv_string = frontend_df.to_csv(index=False)
        
        print(f"  Frontend CSV: {frontend_df.shape[0]} rows, {frontend_df.shape[1]} columns")
        print(f"  Frontend columns (first 6): {list(frontend_df.columns[:6])}")
        
        # 2. Backend receives and reorders
        from io import StringIO
        raw_data = pd.read_csv(StringIO(csv_string))
        raw_data_ordered = reorder_dataframe(raw_data)
        
        print(f"  After reorder: {raw_data_ordered.shape[0]} rows, {raw_data_ordered.shape[1]} columns")
        print(f"  Backend columns (first 8): {list(raw_data_ordered.columns[:8])}")
        
        # Check how many columns are all zeros (z and v)
        zero_cols = [c for c in raw_data_ordered.columns if (raw_data_ordered[c] == 0).all()]
        print(f"  All-zero columns: {len(zero_cols)} → {zero_cols[:8]}...")
        
        # 3. Feature extraction
        from joint_features import get_es1_features
        features_raw = get_es1_features(raw_data_ordered.fillna(0.0))
        features = features_raw.to_numpy(dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"\n  Features: {features.shape}")
        for i, col in enumerate(features_raw.columns):
            vals = features[:, i]
            print(f"    {col}: min={vals.min():.4f}, max={vals.max():.4f}, "
                  f"mean={vals.mean():.4f}, std={vals.std():.4f}")
        
        # 4. Temporal downsample
        from ml_wrapper import temporal_downsample, DOWNSAMPLE_STRIDE
        features_ds = temporal_downsample(features, DOWNSAMPLE_STRIDE, max_length)
        print(f"\n  After downsample(stride={DOWNSAMPLE_STRIDE}): {features.shape[0]} → {features_ds.shape[0]} frames")
        
        # 5. Scaler transform
        scaler = load_scaler(exercise_id)
        print(f"  Scaler: n_features={scaler.n_features_in_}")
        print(f"  Scaler mean: {scaler.mean_}")
        print(f"  Scaler scale: {scaler.scale_}")
        
        features_scaled = scaler.transform(features_ds).astype(np.float32)
        print(f"  After scaling: min={features_scaled.min():.3f}, max={features_scaled.max():.3f}, "
              f"mean={features_scaled.mean():.4f}")
        
        # 6. Pad
        if features_scaled.shape[0] > max_length:
            features_out = features_scaled[:max_length]
        else:
            pad_len = max_length - features_scaled.shape[0]
            features_out = np.pad(features_scaled, ((0, pad_len), (0, 0)),
                                  mode="constant", constant_values=-999.0)
        
        real_pct = 100 * np.sum(features_out != -999.0) / features_out.size
        print(f"  After pad: {features_out.shape}, real data: {real_pct:.1f}%")
        
        # 7. Predict
        data = np.expand_dims(features_out, axis=0)
        data = np.nan_to_num(data, nan=-999.0)
        
        raw_pred = model.predict(data, verbose=0).flatten()[0]
        
        # Auto-detect normalization
        if raw_pred <= 1.5:
            score = float(np.clip(raw_pred * 50.0, 0, 50) * 2.0)
        else:
            score = float(np.clip(raw_pred, 0, 50) * 2.0)
        
        print(f"\n  🎯 Model raw output: {raw_pred:.6f}")
        print(f"  🎯 Final score (0-100): {score:.1f}")
    
    # Compare with diagnostic endpoint approach
    print(f"\n{'=' * 70}")
    print("  COMPARISON: Simulated Webcam vs Diagnostic Synthetic")
    print(f"{'=' * 70}")
    print("  Run `python test_deep_diagnostic.py` to see diagnostic scores.")
    print("  If webcam scores are all ~46-48 while diagnostic shows 28-93,")
    print("  the issue is that real webcam features → scaler → model sees OOD data.")


if __name__ == "__main__":
    main()
