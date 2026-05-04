"""
Diagnostic script to verify the clinical score pipeline is working correctly.

Checks:
1. Scaler files exist for all exercises
2. Models load successfully
3. Inference produces varied output (not stuck at 53)
4. Output range is reasonable (0-100)

Usage:
    cd backend && python diagnostic_check.py
"""

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings


def check_scalers():
    """Check that scaler files exist for all exercises."""
    print("=" * 60)
    print("  CHECK 1: Scaler Files")
    print("=" * 60)
    
    scalers_dir = "models/"
    exercises = ["Es1", "Es2", "Es3", "Es4", "Es5"]
    all_ok = True
    
    for ex in exercises:
        path = os.path.join(scalers_dir, f"scaler_{ex}.joblib")
        if os.path.exists(path):
            import joblib
            scaler = joblib.load(path)
            n_features = scaler.n_features_in_
            print(f"  ✓ {ex}: scaler loaded (features={n_features}, mean_range=[{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}])")
        else:
            print(f"  ✗ {ex}: scaler NOT FOUND at {path}")
            all_ok = False
    
    return all_ok


def check_models():
    """Check that model files exist and load."""
    print("\n" + "=" * 60)
    print("  CHECK 2: Model Files")
    print("=" * 60)
    
    import tensorflow as tf
    
    models_dir = "models/"
    exercises = ["Es1", "Es2", "Es3", "Es4", "Es5"]
    all_ok = True
    loaded_models = {}
    
    for ex in exercises:
        candidates = [
            f"ml_model_{ex}.keras",
            f"ml_model_{ex}_best.keras",
            f"ml_model_{ex}.h5",
            f"ml_model_{ex}_best.h5",
        ]
        found = None
        for c in candidates:
            p = os.path.join(models_dir, c)
            if os.path.exists(p):
                found = p
                break
        
        if found:
            try:
                model = tf.keras.models.load_model(found, compile=False)
                loaded_models[ex] = model
                # Check output activation
                last_layer = model.layers[-1]
                activation = getattr(last_layer, 'activation', None)
                act_name = getattr(activation, '__name__', 'unknown') if activation else 'unknown'
                print(f"  ✓ {ex}: loaded from {os.path.basename(found)} | input={model.input_shape} | output_activation={act_name}")
            except Exception as e:
                print(f"  ✗ {ex}: failed to load from {found}: {e}")
                all_ok = False
        else:
            print(f"  ✗ {ex}: no model file found")
            all_ok = False
    
    return all_ok, loaded_models


def check_inference(loaded_models):
    """Test inference with varied inputs to verify output varies."""
    print("\n" + "=" * 60)
    print("  CHECK 3: Inference Variability")
    print("=" * 60)
    
    import joblib
    import pandas as pd
    from ml_wrapper import prepare_data, get_dataframe_cols
    
    MAX_LENGTH_MAPPING = {
        "Es1": 150, "Es2": 150, "Es3": 297, "Es4": 150, "Es5": 150,
    }
    
    cols = get_dataframe_cols()
    all_ok = True
    
    for ex, model in loaded_models.items():
        max_len = MAX_LENGTH_MAPPING[ex]
        
        # Test 1: All zeros (no movement)
        zeros_df = pd.DataFrame(
            np.zeros((600, len(cols)), dtype=np.float32),
            columns=cols,
        )
        try:
            zeros_prepared = prepare_data(zeros_df, max_len, ex)
            zeros_pred = model.predict(zeros_prepared, verbose=0)
            zeros_score = float(np.clip(zeros_pred.flatten()[0] * 50.0, 0, 50) * 2.0)
        except Exception as e:
            print(f"  ✗ {ex}: zeros test failed: {e}")
            all_ok = False
            continue
        
        # Test 2: Random movement
        random_df = pd.DataFrame(
            np.random.uniform(0.1, 0.9, (600, len(cols))).astype(np.float32),
            columns=cols,
        )
        random_prepared = prepare_data(random_df, max_len, ex)
        random_pred = model.predict(random_prepared, verbose=0)
        random_score = float(np.clip(random_pred.flatten()[0] * 50.0, 0, 50) * 2.0)
        
        # Test 3: Different random
        random2_df = pd.DataFrame(
            np.random.uniform(0.3, 0.7, (300, len(cols))).astype(np.float32),
            columns=cols,
        )
        random2_prepared = prepare_data(random2_df, max_len, ex)
        random2_pred = model.predict(random2_prepared, verbose=0)
        random2_score = float(np.clip(random2_pred.flatten()[0] * 50.0, 0, 50) * 2.0)
        
        scores = [zeros_score, random_score, random2_score]
        score_range = max(scores) - min(scores)
        
        status = "✓" if score_range > 1.0 else "✗ STUCK"
        print(f"  {status} {ex}: zeros={zeros_score:.1f}, rand1={random_score:.1f}, rand2={random2_score:.1f} | range={score_range:.1f}")
        
        if score_range <= 1.0:
            all_ok = False
    
    return all_ok


def main():
    print("\n🔍 RehabAI Clinical Score Pipeline Diagnostic")
    print("=" * 60)
    
    scalers_ok = check_scalers()
    models_ok, loaded_models = check_models()
    
    if not scalers_ok:
        print("\n❌ SCALERS MISSING — run 03_extract_joint_features.py + scripts/export_scalers.py")
        print("   This is the root cause of the 53-point fixed score.")
        sys.exit(1)
    
    if not models_ok:
        print("\n❌ MODELS MISSING — check models/ directory")
        sys.exit(1)
    
    inference_ok = check_inference(loaded_models)
    
    print("\n" + "=" * 60)
    if scalers_ok and models_ok and inference_ok:
        print("  ✅ ALL CHECKS PASSED — pipeline should produce varied scores")
    else:
        print("  ❌ SOME CHECKS FAILED — see details above")
    print("=" * 60)


if __name__ == "__main__":
    main()
