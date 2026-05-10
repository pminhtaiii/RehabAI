# RehabAI Quick Reference Guide - Configuration & Code Samples

## At-a-Glance Reference

### Per-Exercise Configuration

```python
# Feature Counts & Configuration per Exercise

EXERCISE_CONFIG = {
    "Es1": {
        "name": "Lifting of Arms",
        "num_features": 6,
        "features": [
            "left_elbow_angle",
            "right_elbow_angle",
            "hand_shoulder_ratio",
            "torso_tilted_angle",
            "hand_tilted_angle",
            "elbow_angles_diff"
        ],
        "max_length": 150,
        "motion_thresholds": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5}
    },
    "Es2": {
        "name": "Lateral Trunk Tilt",
        "num_features": 6,
        "features": [
            "left_elbow_angle",
            "right_elbow_angle",
            "torso_tilted_angle",
            "elbow_angles_diff",
            "left_shoulder_angle",
            "right_shoulder_angle"
        ],
        "max_length": 150,
        "motion_thresholds": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5}
    },
    "Es3": {
        "name": "Trunk Rotation",
        "num_features": 9,
        "features": [
            "left_elbow_angle",
            "right_elbow_angle",
            "hand_shoulder_ratio",
            "torso_tilted_angle",
            "elbow_angles_diff",
            "left_shoulder_angle",
            "right_shoulder_angle",
            "left_arm_torso_angle",
            "right_arm_torso_angle"
        ],
        "max_length": 150,
        "motion_thresholds": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5}
    },
    "Es4": {
        "name": "Pelvis Rotation",
        "num_features": 2,
        "features": [
            "torso_tilted_angle",
            "knee_hip_ratio"
        ],
        "max_length": 150,
        "motion_thresholds": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5}
    },
    "Es5": {
        "name": "Squatting",
        "num_features": 7,
        "features": [
            "left_elbow_angle",
            "right_elbow_angle",
            "hand_shoulder_ratio",
            "torso_tilted_angle",
            "elbow_angles_diff",
            "left_shoulder_angle",
            "right_shoulder_angle"
        ],
        "max_length": 150,
        "motion_thresholds": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5}
    }
}
```

---

## Critical Code Snippets

### 1. Correct Temporal Downsampling Order

```python
# ✅ CORRECT ORDER (Feature → Downsample → Scale)
def correct_pipeline(df, exercise_id, max_length=150):
    # Step 1: Extract features
    features = extract_features(df, exercise_id)  # (N, F)
    
    # Step 2: Downsample BEFORE scaling
    features = temporal_downsample(features, stride=5, target_len=max_length)
    
    # Step 3: Scale
    scaler = load_scaler(exercise_id)
    features_scaled = scaler.transform(features)
    
    return features_scaled

# ❌ WRONG ORDER (Feature → Scale → Downsample)
def wrong_pipeline(df, exercise_id, max_length=150):
    features = extract_features(df, exercise_id)
    
    # BUG: Scaling before downsampling changes distribution!
    scaler = load_scaler(exercise_id)
    features_scaled = scaler.transform(features)
    
    # BROKEN: downsampled features won't match training distribution
    features = temporal_downsample(features_scaled, stride=5, target_len=max_length)
    
    return features
```

### 2. Masking Value Must Be -999.0

```python
# ✅ CORRECT: Using -999.0 as sentinel
def correct_masking(features, max_length=150):
    mask_value = -999.0  # Far outside [-3, 3] typical post-scaling range
    
    if features.shape[0] < max_length:
        pad_len = max_length - features.shape[0]
        features_padded = np.pad(
            features,
            ((0, pad_len), (0, 0)),
            mode="constant",
            constant_values=mask_value
        )
    
    # LSTM Masking layer skips -999.0 values
    return features_padded

# ❌ WRONG: Using 0.0 as sentinel
def wrong_masking(features, max_length=150):
    mask_value = 0.0  # BUG: Same as post-scaling mean!
    
    # After StandardScaler: mean ≈ 0.0
    # So LSTM cannot distinguish:
    # - Real feature value = 0.0 (legitimate data)
    # - Padding = 0.0 (to be ignored)
    # Result: Model learns to ignore real features!
    
    features_padded = np.pad(features, ..., constant_values=mask_value)
    return features_padded
```

### 3. Data Leakage Prevention in Cross-Validation

```python
# ✅ CORRECT: Fit scaler per-fold on train set only
def correct_cv(raw_features, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(raw_features):
        # Get train and validation raw features
        X_train_raw = [raw_features[i] for i in train_idx]
        X_val_raw = [raw_features[i] for i in val_idx]
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # FIX: Fit scaler ONLY on train data
        all_train_frames = np.concatenate(X_train_raw, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_train_frames)  # ← NO validation data here!
        
        # Apply same scaler to both train and validation
        X_train_scaled = [scaler.transform(x) for x in X_train_raw]
        X_val_scaled = [scaler.transform(x) for x in X_val_raw]
        
        # Train model on (X_train_scaled, y_train)
        # Evaluate on (X_val_scaled, y_val)

# ❌ WRONG: Fit scaler on entire dataset
def wrong_cv(raw_features, y, n_splits=5):
    # BUG: Validation data statistics in scaler!
    all_frames = np.concatenate(raw_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_frames)  # ← LEAKS validation data!
    
    for train_idx, val_idx in kfold.split(raw_features):
        X_train_scaled = [scaler.transform(x) for x in X_train_raw]
        X_val_scaled = [scaler.transform(x) for x in X_val_raw]
        # Validation metrics inflated because scaler knows val distribution
```

### 4. Correct Feature Extraction Match

```python
# ✅ Training feature extraction (03_extract_joint_features.py)
def training_features(joint_df, exercise):
    extractor = FEATURE_EXTRACTORS[exercise]
    return extractor(joint_df)

# ✅ Inference feature extraction (backend/joint_features.py)
def inference_features(joint_df, exercise_id):
    # Must use identical geometry functions and order
    extractor = FEATURE_EXTRACTORS[exercise_id]
    return extractor(joint_df)

# ❌ WRONG: Different feature order
def wrong_features(joint_df, exercise):
    # Training extracts: [left_elbow, right_elbow, hand_ratio, ...]
    # Inference extracts: [right_elbow, left_elbow, hand_ratio, ...]
    # → Scaler means/scales misaligned → garbage predictions
    pass
```

### 5. Aspect Ratio Correction

```python
# ✅ CORRECT: Apply 1.7778 factor to x
def correct_aspect_ratio(df):
    # 16:9 video: x normalized to [0,1] is stretched
    # Correction: x * 1920/1080 = x * 1.7778
    x_corrected = df["left_shoulder_x"].values * 1.7778
    y_corrected = df["left_shoulder_y"].values
    
    joint_2d = np.column_stack((x_corrected, y_corrected))
    return joint_2d

# ❌ WRONG: No correction
def wrong_aspect_ratio(df):
    x_raw = df["left_shoulder_x"].values  # Not corrected
    y_raw = df["left_shoulder_y"].values
    
    # Angles computed on stretched coordinate system
    # → Angles incorrect for isotropic space
    joint_2d = np.column_stack((x_raw, y_raw))
    return joint_2d
```

---

## Training Configuration Values

```python
# LSTM Training Parameters
TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 16,
    "learning_rate": 0.001,
    "learning_rate_init": 0.00001,  # Warmup start
    "learning_rate_warmup_epochs": 20,
    
    "dropout_rate": 0.3,
    "gradient_clip_value": 0.5,
    
    "lstm_units": [32, 16],  # Funnel: 32 → 16
    "dense_units": 8,
    
    "optimizer": "Adam",
    "loss": "Huber",
    
    "early_stopping_patience": 20,  # Epochs without improvement
    "reduce_lr_patience": 10,        # Epochs before LR reduction
    "reduce_lr_factor": 0.5,         # Multiply LR by this
    
    "k_folds": 5,
    "random_seed": 42,
    
    "max_length": 150,
    "downsample_stride": 5,
    "mask_value": -999.0,
}

# Model Architecture (per-exercise)
MODEL_ARCHITECTURE = {
    "input_shape": (150, None),  # (max_length, num_features varies)
    "layers": [
        {"type": "Input", "shape": (150, 6)},  # Example: Es1 has 6 features
        {"type": "Masking", "mask_value": -999.0},
        {"type": "LSTM", "units": 32, "return_sequences": True},
        {"type": "Dropout", "rate": 0.3},
        {"type": "LSTM", "units": 16, "return_sequences": False},
        {"type": "Dropout", "rate": 0.3},
        {"type": "Dense", "units": 8, "activation": "relu"},
        {"type": "Dense", "units": 1, "activation": "sigmoid"},
    ]
}
```

---

## File Paths & Directory Structure

```python
# Base Directory Configuration
BASE_DIR = 'C:/RehabAI'

# Data Pipeline Directories
PATHS = {
    # Stage 1 inputs
    "raw_videos": f'{BASE_DIR}/01_raw_data',
    
    # Stage 1 outputs
    "raw_joints": f'{BASE_DIR}/03_raw_joints',
    
    # Stage 2 outputs
    "final_datasets": f'{BASE_DIR}/05_final_datasets',
    
    # Stage 3 outputs
    "scaler_artifacts": f'{BASE_DIR}/models',
    
    # Stage 4 outputs
    "trained_models": f'{BASE_DIR}/models',
    
    # Stage 5 models
    "inference_models": f'{BASE_DIR}/models',
    "inference_scalers": f'{BASE_DIR}/models',
}

# Key File Names
FILE_NAMES = {
    "metadata_csv": "KiMoRe_final.csv",
    "features_csv": "{ID}_{exercise}_features.csv",
    "scaler": "scaler_Es{X}.joblib",
    "model": "ml_model_Es{X}.keras",
    "model_legacy": "ml_model_Es{X}.h5",
}

# Example Full Paths
EXAMPLE_PATHS = {
    "joint_csv": "03_raw_joints/healthy/professional/KIMORE_P001/Es1/video_mediapipe.csv",
    "features_csv": "05_final_datasets/KIMORE_P001_Es1_features.csv",
    "scaler_Es1": "models/scaler_Es1.joblib",
    "model_Es1": "models/ml_model_Es1.keras",
}
```

---

## Debugging Checklist

### Model Predicts Constant Score (e.g., always 0.5)
```
[ ] Check StandardScaler was fit ONLY on train set (not entire dataset)
[ ] Verify downsampling applied BEFORE scaling in inference
[ ] Ensure mask_value = -999.0 (not 0.0)
[ ] Confirm LR warmup enabled in training
[ ] Monitor training loss curve (should decrease)
[ ] Check if model overfitting (val_loss plateaus while train_loss decreases)
[ ] Inspect prepared data shape: (1, 150, F)
```

### Inference Returns Shape Mismatch Error
```
[ ] Verify model.input_shape matches prepared data
[ ] Check prepared shape: (1, 150, F) where F matches model
[ ] Ensure feature count matches: scaler.n_features_in_ == F
[ ] Validate max_length=150 is correct for this exercise
[ ] Confirm model loaded successfully (not None in cache)
```

### Scores Too High or Low
```
[ ] Check motion quality_factor is reasonable (0.1-1.0)
[ ] Verify output scaling: *50 for clinical, *2 for display
[ ] Compare with training metrics (MAE, R²)
[ ] Validate clinical score labels in training [0, 50]
[ ] Check if specific patients/exercises are outliers
```

### Motion Detection Always Inactive
```
[ ] Verify feature values non-zero (keypoints detected)
[ ] Check motion thresholds not too aggressive
[ ] Confirm downsample preserves motion signature
[ ] Analyze variance/ROM/displacement per exercise
[ ] Check if exercise requires larger ROM (Es4 is subtle)
```

---

## Key Metrics & Performance Targets

### Training Set Sizes
```
Per Exercise Typical:  ~72 samples
After 5-Fold Split:    ~60 train, ~12 validation per fold
After Downsampling:    100-200 frames per sample (from ~500)
After Padding:         150 frames per sample (fixed length)
```

### Expected Model Performance (from cv evaluation)
```
Metric                  Target Range        Notes
─────────────────────────────────────────────────────────
MAE (Mean Absolute)     ±3-5 score points   Error ~6-10% of range
MSE                     15-30               Squared error
R² correlation          0.6-0.8             Explains 60-80% variance
Pearson r               0.7-0.9             Linear correlation
Spearman ρ              0.7-0.9             Rank correlation

Per-fold variance       < 10% relative      Cross-fold consistency
```

### Inference Latency Targets
```
Component              Typical Time        Target
─────────────────────────────────────────────────────
Feature extraction     5ms                 < 10ms
Downsampling          1ms                 < 5ms
StandardScaler        2ms                 < 5ms
Padding               1ms                 < 5ms
LSTM inference        50-200ms            < 250ms GPU
                                          < 500ms CPU
─────────────────────────────────────────────────────
Total per request     60-260ms            < 300ms
```

### Model File Sizes
```
Exercise   Model Size    Scaler Size   Total
─────────────────────────────────────────────
Es1        ~12 KB        ~1 KB         ~13 KB
Es3        ~13 KB        ~2 KB         ~15 KB (largest)
Es4        ~10 KB        ~0.5 KB       ~10.5 KB (smallest)
```

---

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: incompatible tensor shapes` | max_length mismatch | Verify max_length=150 matches model.input_shape |
| `FileNotFoundError: scaler_Es1.joblib` | Scaler not found | Run 03_extract_joint_features.py first |
| `Model predicts 0.5 always` | Incorrect mask_value or LR too high | Check mask_value=-999.0; enable LR warmup |
| `NaN in model output` | NaN/inf in prepared data | Add np.nan_to_num() before padding |
| `Shape (X, 150, 9) vs expected (X, 150, 6)` | Feature count mismatch | Verify exercise_id matches scaler features |
| `MemoryError on large batch` | Batch size too large | Reduce batch_size; use model.predict() instead |
| `Scaler n_features mismatch` | Scaler from wrong exercise | Check scaler_{ExX}.joblib matches exercise_id |
| `Motion always inactive` | Thresholds too strict | Adjust EXERCISE_THRESHOLDS or check input variance |

---

## Performance Optimization Tips

### Inference Optimization
```python
# 1. Use inference-optimized model format
# ✅ .keras format (modern, faster loading)
model = tf.keras.models.load_model('ml_model_Es1.keras')

# ❌ .h5 legacy format (slower)
model = tf.keras.models.load_model('ml_model_Es1.h5')

# 2. Load model once, reuse across requests
ml_models_cache = {}
for ex in ["Es1", "Es2", "Es3", "Es4", "Es5"]:
    ml_models_cache[ex] = tf.keras.models.load_model(...)

# 3. Batch inference where possible
scores = model.predict(np.array([data1, data2, ...]))  # Batch

# 4. Use CPU-optimized scaler transform
scaler.transform(features)  # Vectorized NumPy

# 5. Profile to find bottleneck
import cProfile
cProfile.run('prepare_data(...)')
```

### Training Optimization
```python
# 1. Use early stopping to avoid unnecessary epochs
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
]

# 2. Reduce learning rate on plateau
callbacks += [
    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)
]

# 3. Use data generators for large datasets
train_gen = keras.preprocessing.image.ImageDataGenerator(...)
model.fit(train_gen, ...)

# 4. Mixed precision training for GPU (optional)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

---

**End of Quick Reference**
