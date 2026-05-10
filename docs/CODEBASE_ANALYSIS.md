# RehabAI Codebase Analysis - Comprehensive Data & Model Pipeline

**Generated:** May 5, 2026  
**Scope:** Complete analysis of data flow, model architecture, configuration, and potential issues

---

## 1. SYSTEM OVERVIEW

### Purpose
RehabAI is an **AI-powered rehabilitation system** that provides real-time feedback for therapeutic exercises using:
- **Real-time pose estimation** (MoveNet on frontend, MediaPipe on backend)
- **Biomechanical feature extraction** from body joints
- **Clinical score prediction** using LSTM neural networks (0-100 scale)
- **Motion detection** to validate actual exercise performance
- **Personalized feedback** based on exercise execution quality

### Five Therapeutic Exercises
- **Es1**: Arm lifts (lifting of arms)
- **Es2**: Lateral trunk tilt
- **Es3**: Trunk rotation
- **Es4**: Pelvis rotation
- **Es5**: Squatting movements

### Architecture Stack
```
Frontend (React 18 + TensorFlow.js)
  ↓ (MoveNet LIGHTNING @ 30 FPS, 17 keypoints)
WebSocket + REST API
  ↓
Backend (FastAPI + TensorFlow 2.21)
  ├─ Real-time feature extraction
  ├─ LSTM model inference
  ├─ Motion detection & validation
  └─ Feedback generation
  ↓
SQLite Database (user profiles, progress, scores)
```

---

## 2. DATA PIPELINE FLOW (5 Stages)

### **Stage 1: Raw Video Input → Joint Extraction**
**File:** `01_extract_joint_positions.py`  
**Input:** Raw MP4 video files from KiMoRe dataset  
**Output:** CSV files with 3D joint positions  

**Process:**
1. Uses **MediaPipe PoseLandmarker (Heavy model)** for high accuracy
2. Extracts **12 body-only keypoints** (shoulders, elbows, wrists, hips, knees, ankles)
3. Each keypoint has 4 channels: x, y, z (3D), visibility → **48 columns per frame**
4. Keypoints normalized to [0,1] range (x, y) with depth relative to hip midpoint (z)

**Configuration:**
- **Model variant:** Heavy (highest accuracy, slower processing)
- **Video mode:** VIDEO (not image mode) for temporal consistency
- **Frame timestamp:** Global monotonically increasing (required by MediaPipe API)
- **Output file naming:** `{video}_mediapipe.csv` (suffix prevents overwriting)

**Key Bodymarks Extracted:**
```python
BODY_LANDMARKS = {
    left_shoulder: idx=11, right_shoulder: 12,
    left_elbow: 13,       right_elbow: 14,
    left_wrist: 15,       right_wrist: 16,
    left_hip: 23,         right_hip: 24,
    left_knee: 25,        right_knee: 26,
    left_ankle: 27,       right_ankle: 28,
}
```

**Column format per keypoint:**
```
{name}_x, {name}_y, {name}_z, {name}_v
Example: left_shoulder_x, left_shoulder_y, left_shoulder_z, left_shoulder_v
```

---

### **Stage 2: Joint CSVs → Metadata Dataset**
**File:** `02_prepare_dataset.py`  
**Input:** Extracted joint CSVs + clinical assessment labels  
**Output:** Master metadata CSV `KiMoRe_final.csv`  

**Process:**
1. Scans directory tree: `{JOINTS_DIR}/{clinical_group}/{expertise}/{subject_id}/{exercise}/`
2. For each exercise directory:
   - Finds MediaPipe CSV (prefer `*_mediapipe.csv`)
   - Looks up corresponding MP4 video file
   - Reads clinical score from Excel (ClinicalAssessment_*.xlsx)
   - Counts frames in CSV
3. Builds combined metadata table with all paths

**Metadata Columns:**
```
ID, clinical_group, expertise, exercise, 
video, joint_positions, clinical_score, #frames
```

**Example Row:**
```
ID=KIMORE_P001
clinical_group=healthy
expertise=professional
exercise=Es1
video=/01_raw_data/healthy/professional/KIMORE_P001/Es1/rgb/video.mp4
joint_positions=/03_raw_joints/healthy/professional/KIMORE_P001/Es1/video_mediapipe.csv
clinical_score=45.5
#frames=672
```

**Clinical Score Lookup:**
- Searches multiple directory patterns (flexible KiMoRe dataset layout)
- Column name: `clinical TS Ex#{exercise_num}` (e.g., `clinical TS Ex#1` for Es1)
- Uses only **Total Score (TS)** field (0-50 range), NOT sum of sub-scores
- Returns NaN if not found (handled downstream)

**Stats Tracking:**
- Total entries processed
- Missing video files
- Missing clinical scores
- Per-exercise breakdown with score ranges

---

### **Stage 3: Joint Positions → 2D Biomechanical Features**
**File:** `03_extract_joint_features.py`  
**Input:** Master CSV `KiMoRe_final.csv`  
**Output:** Per-sample CSVs in `{FINAL_DATASET_DIR}`, StandardScaler pickles, master features CSV  

**Process:**

#### 3.1: Aspect Ratio Correction
MediaPipe normalizes x, y to [0,1]. For 16:9 video, this stretches the x-axis.
```python
# Correction: x * 1.7778 = x * (1920/1080)
# Makes 2D space isotropic for angle calculations
x_corrected = x_normalized * 1.7778
```

#### 3.2: Exercise-Specific Feature Extraction
Uses **paper-defined optimal feature subsets** (Table 2 from Guo & Khan 2021):

**Es1 (Arm Lifts) - 6 Features:**
1. Left elbow angle (shoulder→elbow→wrist)
2. Right elbow angle
3. Hand-shoulder ratio (distance_hands / distance_shoulders)
4. Torso tilted angle (torso_vector vs vertical)
5. Hand tilted angle (hand_vector vs horizontal)
6. Elbow angles difference (|left - right|)

**Es2 (Lateral Trunk Tilt) - 6 Features:**
1. Left elbow angle
2. Right elbow angle
3. Torso tilted angle
4. Elbow angles difference
5. Left shoulder angle (hip→shoulder→elbow)
6. Right shoulder angle

**Es3 (Trunk Rotation) - 9 Features:**
1-6: [Same as Es1]
7. Left shoulder angle
8. Right shoulder angle
9. Left arm-torso angle (torso_vector vs left_arm_vector)
10. Right arm-torso angle

**Es4 (Pelvis Rotation) - 2 Features:**
1. Torso tilted angle
2. Knee-hip ratio (distance_knees / distance_hips)

**Es5 (Squatting) - 7 Features:**
1-6: [Es1 features + shoulder angles]
7. Left shoulder angle
8. Right shoulder angle

#### 3.3: Feature Computation Methods
**Angles (2D):**
```python
# Uses atan2(cross_product, dot_product)
angle = arctan2(cross, dot)
# Returns degrees [0, 180], absolute (not signed)
```

**Distances (2D Euclidean):**
```python
distance = sqrt((x2-x1)² + (y2-y1)²)
```

**Ratios:**
```python
ratio = distance_pair1 / (distance_pair2 + 1e-8)  # 1e-8 prevents division by zero
```

**Vectors:**
```python
torso_vector = hip_midpoint - shoulder_midpoint
# Points downward (y increases downward in video coordinates)
```

#### 3.4: Standardization (Per-Exercise)
For each exercise:
1. Extract all raw features from training set
2. Fit **StandardScaler** on TRAINING DATA ONLY (prevents data leakage)
3. Save scaler to `scaler_{ExX}.joblib`
4. Apply scaler to ALL features: `(feature - mean) / std`

**Scaler artifacts saved:**
```
scaler_Es1.joblib
scaler_Es2.joblib
scaler_Es3.joblib
scaler_Es4.joblib
scaler_Es5.joblib
```

---

### **Stage 4: Features → Training Dataset Preparation**
**File:** `training_models/clinical_score_prediction_model.py`  
**Input:** Scaled features from Stage 3  
**Output:** Trained LSTM models (`ml_model_Es*.keras`), training metrics  

**Process:**

#### 4.1: Load and Downsample
```python
def temporal_downsample(feat_arr, stride=5, target_len=150):
    """
    Reduces sequence length to prevent overfitting on small dataset (72 samples).
    
    Example: 
      - Raw video 25fps, ~27s duration → ~675 frames
      - Stride=5: 675 → 135 frames (acceptable for LSTM with 72 samples)
      - Frame-adjacent frames are nearly identical → redundant info
    
    Strategy if longer than target:
      - Keep head (first 75 frames)
      - Keep tail (last 75 frames)
      - Discard middle (often plateau phase less important than start/end)
    """
```

**Critical:** Downsampling applied BEFORE StandardScaler in training.

#### 4.2: Data Split & Scaler Fit (Per-Fold)
**5-Fold Cross-Validation with data leakage prevention:**

```python
For each fold:
  1. Split indices: train_idx (80%), val_idx (20%)
  2. FIT scaler ONLY on train_idx raw features
     → Prevents leakage of validation data statistics
  3. Transform BOTH train and val with same scaler
  4. Pad/truncate to max_length=150
  5. Build + train model
  6. Evaluate on validation set
```

**Max sequence length calculation:**
```python
max_length = percentile_95(all_sequence_lengths)
# Uses 95th percentile to avoid outlier-driven excessive padding
```

#### 4.3: Model Architecture (Practical v6)
```
Input: (batch_size=16, sequence_length=150, num_features=2-9)
  ↓
Masking(mask_value=-999.0)  
  [Ignores padded frames; LSTM skips masked timesteps]
  ↓
LSTM(32 units, return_sequences=True)
  [Output: (batch, 150, 32)]
  ↓
Dropout(0.3)  [30% neurons randomly disabled during training]
  ↓
LSTM(16 units, return_sequences=False)
  [Output: (batch, 16) — returns only last timestep]
  ↓
Dropout(0.3)
  ↓
Dense(8, activation='relu')
  ↓
Dense(1, activation='sigmoid')
  [Output: (batch, 1) ∈ [0, 1]]
```

**Output scaling:**
- Sigmoid output [0, 1] represents normalized clinical score
- Multiply by 50 to get [0, 50] clinical score range
- Multiply by 2 for display [0, 100] scale (may be adjusted per feedback)

**Compilation:**
```python
optimizer: Adam(lr=0.001, clipvalue=0.5)
loss: Huber  [Smooth L1, robust to outliers]
metrics: ['mae']
```

**Training hyperparameters:**
```python
epochs: 200
batch_size: 16
learning_rate: 0.001 (with ReduceLROnPlateau)
callbacks:
  - ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)
  - EarlyStopping(patience=20, restore_best_weights=True)
  - LR Warmup (from lr/100 → target_lr over 20 epochs)
```

**Training Dataset:**
- ~72 samples per exercise (after filtering NaNs)
- Imbalanced: scores not uniformly distributed
- Small size: aggressive dropout (0.3) + masking required

---

### **Stage 5: Real-time Inference Pipeline**
**File:** `backend/ml_wrapper.py`, `backend/main.py`  
**Input:** Live webcam frames → MoveNet keypoints  
**Output:** Clinical score (0-100), motion quality, feedback  

**Real-time Inference Steps:**

```
1. Frontend captures ~30 frames @ 30fps (1 second)
   → Sends raw keypoints (48 columns per frame)
   
2. Backend receives DataFrame(30, 48)
   
3. Feature Extraction
   - SafeExtractFeatures(df, exercise_id)
   - Handles NaN/inf with safety cleanup
   - Outputs (30, F) where F=2-9 features
   
4. Temporal Downsampling (stride=5)
   - 30 frames → 6 frames @ ~6fps
   - Outputs (6, F)
   
5. StandardScaler Transform
   - Loads scaler_{ExX}.joblib
   - Transforms: (feature - mean) / std
   - Outputs (6, F) normalized
   
6. Pad/Truncate to max_length=150
   - If < 150: pad with sentinel -999.0
   - If > 150: truncate to first 150
   - Outputs (150, F)
   
7. Reshape for batch inference
   - (150, F) → (1, 150, F) [batch_size=1]
   
8. LSTM Model.predict()
   - Masking layer ignores -999 values
   - Outputs float ∈ [0, 1]
   
9. Motion Detection (parallel)
   - Analyze (6, F) RAW features (pre-scaling)
   - Compute: variance, ROM, frame_displacement
   - Generate MotionResult with quality_factor ∈ [0.1, 1.0]
   
10. Score Calibration
    - raw_score = model_output * 50
    - calibrated = raw_score * motion_quality_factor
    - Clamped [0, 100]
    
11. Return: {score, motion_energy, quality_factor, details}
```

**Critical Invariants:**
- Must downsample BEFORE scaling (matches training pipeline)
- Must use same scaler from training
- Max sequence length MUST be 150 (model input shape locked)
- Masking value MUST be -999.0 (not 0, which is post-scaling mean)

**Latency Breakdown:**
- Feature extraction: ~5ms
- Downsampling: ~1ms
- Scaling: ~2ms
- LSTM inference: ~50ms (GPU) to ~200ms (CPU)
- Total per request: ~60-260ms

---

## 3. DATA PROCESSING AT EACH STAGE

### Stage 1: MediaPipe Extraction

**Input validation:**
- Video format: MP4
- Video codec: H.264 expected
- Frame rate: 25fps typical (detected via `cv2.CAP_PROP_FPS`)

**Data cleaning:**
- Fills visibility < 0.5 frames (low confidence) → output NaN/0 during inference
- Removes frames where pose detection fails (landmark list incomplete)

**Output quality checks:**
- Validates each frame has 33 landmarks (MediaPipe standard)
- Checks x, y ∈ [0, 1] (normalized coordinates)
- Checks z ∈ [-1, 2] (reasonable depth range)

---

### Stage 2: Metadata Assembly

**Lookups:**
1. **Joint CSV selection:** Prefers `*_mediapipe.csv` over other CSVs
2. **Video file search:** 
   - First tries `{exercise_dir}/rgb/*.mp4`
   - Then tries `{exercise_dir}/*.mp4`
3. **Clinical score lookup:**
   - Multiple path patterns to handle KiMoRe dataset variations
   - Subject ID with underscore/space variants
   - Recursive search fallback

**Data quality:**
- Tracks missing videos/scores separately
- Reports per-exercise statistics

---

### Stage 3: Feature Engineering

**Geometric transformations:**
1. **Aspect ratio correction:** x *= 1.7778 (16:9 video normalization)
2. **Angle calculations:** 
   - Uses atan2 for numerical stability
   - Absolute value → angles always [0, 180°]
   - No signed angles
3. **Distance calculations:**
   - Euclidean norm with epsilon guard (1e-8)
   - Prevents division by zero in ratios

**NaN handling:**
```python
# Input: df with potential NaN in keypoints
df_clean = df.fillna(0.0)  # Conservative: 0 = rest position
features = extractor(df_clean)
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
```

**Scaling:**
- **Per-exercise scalers** (separate for Es1-Es5)
- **Per-fold training** (scaler fit only on train fold, prevents leakage)
- **Output:** zero-mean, unit-variance features

---

### Stage 4: Model Training

**Data pipeline:**
1. Load raw features (variable sequence lengths: ~100-500 frames)
2. Downsample (stride=5)
3. Fit StandardScaler on train subset only
4. Transform train + validation
5. Pad to max_length=150
6. Train with batch_size=16, epochs=200

**Training checks:**
- **Early stopping:** if val_loss doesn't improve for 20 epochs
- **LR annealing:** reduce LR by 0.5× if val_loss plateau (patience=10)
- **Gradient clipping:** max gradient norm = 0.5 (prevents explosion)

**Overfitting prevention:**
- Aggressive dropout (0.3) due to small dataset (72 samples)
- Masking layer (ignores padding, reduces effective seq length)
- Huber loss (robust to outliers)
- Data augmentation: temporal downsampling + padding (implicit variation)

---

### Stage 5: Real-time Inference

**Preprocessing matches training:**
- Same feature extractors (must match exactly)
- Same downsampling rate (stride=5)
- Same StandardScaler artifacts
- Same max_length (150)
- Same masking value (-999.0)

**Motion quality assessment:**
```python
Motion metrics (on RAW features, before scaling):
  1. Temporal variance (std of features over time)
  2. Range of motion (max - min per feature)
  3. Frame displacement (avg |frame[i+1] - frame[i]|)

Quality factor = [0.1, 1.0]
  - if is_active: 0.5 + 0.5 * (motion_energy / expected_energy)
  - if not_active: 0.3 * motion_energy  (heavy penalty)

Final score = model_output * 50 * quality_factor
```

---

## 4. MODEL ARCHITECTURE

### LSTM Model (Practical v6)

**Why LSTM?**
- Captures temporal dependencies in exercise motion
- Handles variable-length sequences via masking
- Learns sequential patterns (smooth vs jerky, symmetric vs asymmetric)

**Layer breakdown:**

| Layer | Type | Config | Output Shape | Purpose |
|-------|------|--------|--------------|---------|
| Input | Input | - | (batch, 150, F) | Raw features |
| Masking | Masking | mask_value=-999 | (batch, 150, F) | Ignore padding |
| LSTM_1 | LSTM | 32 units, ret_seq=True | (batch, 150, 32) | Temporal encoding |
| Drop_1 | Dropout | rate=0.3 | (batch, 150, 32) | Regularization |
| LSTM_2 | LSTM | 16 units, ret_seq=False | (batch, 16) | Temporal compression |
| Drop_2 | Dropout | rate=0.3 | (batch, 16) | Regularization |
| Dense_1 | Dense | 8 units, relu | (batch, 8) | Feature bottleneck |
| Output | Dense | 1 unit, sigmoid | (batch, 1) | Score [0,1] |

**Parameter count:**
- Per-exercise varies based on num_features (F)
- Es1 (F=6): ~3,200 parameters
- Es3 (F=9): ~4,000 parameters (most complex)
- Es4 (F=2): ~2,400 parameters (simplest)

**Masking mechanism:**
```python
# LSTM automatically skips padded timesteps
# Input: [real_feature, real_feature, ..., -999.0, -999.0, ...]
# LSTM processes only real frames, effectively reduces sequence length
# Masking layer output: mask tensor propagated through network
```

**Output scaling:**
```python
# Sigmoid output ∈ [0, 1]
# Maps to [0, 50] clinical score: output * 50
# Display scale [0, 100]: output * 50 * 2
# (Factor of 2 may be tuned per feedback requirements)
```

**Loss function - Huber:**
```python
# Smooth L1: combines MSE (near 0) + MAE (far from 0)
# Robust to outliers in clinical scores
# Prevents gradient explosion from large errors
```

---

## 5. KEY CONFIGURATIONS & HARDCODED PARAMETERS

### Critical Values (DO NOT CHANGE without retraining)

| Parameter | Value | File | Purpose | Impact |
|-----------|-------|------|---------|--------|
| **max_length** | 150 | ml_wrapper.py, main.py | Model input sequence length | Model input shape locked; must match during training |
| **downsample_stride** | 5 | ml_wrapper.py, clinical_score_prediction_model.py | Temporal decimation | 25fps → 5fps; must match training |
| **mask_value** | -999.0 | ml_wrapper.py, clinical_score_prediction_model.py | Padding sentinel | Different from post-scaling mean (0.0); LSTM skips this value |
| **aspect_ratio_factor** | 1.7778 | joint_features.py, backend/joint_features.py | 16:9 video normalization | x *= 1.7778 for isotropic angle math |

### Per-Exercise Feature Counts

| Exercise | Features | File | Reasoning |
|----------|----------|------|-----------|
| Es1 | 6 | joint_features.py | Arm lifts: focus on elbow angles, hand positions |
| Es2 | 6 | joint_features.py | Lateral tilt: shoulder + elbow angles |
| Es3 | 9 | joint_features.py | Trunk rotation: most complex, needs arm-torso angles |
| Es4 | 2 | joint_features.py | Pelvis rotation: minimal features (torso, knees) |
| Es5 | 7 | joint_features.py | Squatting: arm + shoulder + hip angles |

### Motion Detection Thresholds

```python
EXERCISE_THRESHOLDS = {
    "Es1": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es2": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es3": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es4": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es5": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
}

EXPECTED_ENERGY = {all exercises: 1.0}
```

**Activation logic:** `is_active = True` if ≥2 metrics exceed thresholds

### Training Hyperparameters

| Param | Value | Tuning |
|-------|-------|--------|
| **Batch size** | 16 | Fixed; matches paper v6 |
| **Epochs** | 200 | Max; stopped early if no improvement |
| **Learning rate** | 0.001 | Initial; reduced by ReduceLROnPlateau |
| **LR warmup** | lr/100 over 20 epochs | Prevents initial gradient explosion |
| **Dropout** | 0.3 | Aggressive due to small dataset |
| **LSTM units** | 32 → 16 | Funnel architecture |
| **Dense units** | 8 | Feature bottleneck |
| **Gradient clip** | 0.5 | Prevents gradient explosion |
| **EarlyStopping patience** | 20 epochs | Overfitting threshold |
| **ReduceLROnPlateau patience** | 10 epochs | LR annealing threshold |

---

## 6. DATA VALIDATION & QUALITY CHECKS

### At Each Pipeline Stage

#### Stage 1 - Joint Extraction
**Checks performed:**
- Frame count: ≥5 frames required
- Keypoint completeness: all 12 body landmarks detected
- Coordinate ranges: x, y ∈ [0,1]; z ∈ [-1, 2]
- Visibility scores: logged but not filtered

**Failure modes:**
- Missing poses → NaN values in CSV
- Low visibility → 0.0 recorded (treated as missing during inference)

---

#### Stage 2 - Metadata Assembly
**Checks performed:**
- Joint CSV exists and is readable
- Video file located (may be missing - recorded as NaN)
- Clinical score found (may be missing - recorded as NaN)
- Frame count extracted correctly

**Stats collected:**
- Total entries, missing videos, missing scores per exercise
- Score ranges per exercise

---

#### Stage 3 - Feature Engineering
**Input validation:**
- DataFrame columns match expected MediaPipe format (48 columns)
- Fills NaN keypoints with 0.0 before feature extraction

**Computation safety:**
- 1e-8 epsilon in denominator (prevents div-by-zero)
- NaN/inf handling: `nan_to_num(nan=0, posinf=0, neginf=0)`

**Scaler validation:**
- `scaler.n_features_in_` must match feature count
- Scaling applied only to features ≥5 frames
- Saves scaler to disk with exercise identifier

---

#### Stage 4 - Training
**Dataset checks:**
- Minimum 5 samples per fold (72 / 5 = ~14 per fold typically)
- Sequence lengths: reported min/max/mean
- Score ranges: reported min/max/mean/std
- All-zero sequences skipped (indicates pose detection failure)

**Training validation:**
- Batch losses logged every 50 epochs
- Val loss monitored for EarlyStopping
- Model checkpoints saved on val loss improvement
- Final fold metrics: MAE, MSE, R², Pearson/Spearman correlation

---

#### Stage 5 - Inference
**Input checks:**
```python
# Safety limits
df = df.head(1500)  # 30fps * 50s margin

# Feature extraction validation
nan_count = np.isnan(features).sum()
inf_count = np.isinf(features).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values in features")

# Scaler shape validation
if scaler.n_features_in_ != features.shape[1]:
    print(f"WARNING: scaler expects {scaler.n_features_in_} features, got {features.shape[1]}")

# Motion detection validation
if motion_result.is_active == False:
    quality_factor = 0.1 (heavy penalty, but not zero)
```

**Output validation:**
- Score clamped to [0, 100]
- Motion energy ∈ [0, 1]
- Quality factor ∈ [0.1, 1.0]

---

## 7. POTENTIAL ISSUES & RISKS

### 🔴 HIGH SEVERITY

#### 1. **Data Leakage in Scaler Fit** (PARTIALLY FIXED)
**Risk:** If StandardScaler fit on entire dataset before split, validation set statistics leak into training.

**Current Status:** 
- ✅ Fixed in `clinical_score_prediction_model.py` (scaler fit per-fold on train only)
- ❌ Risk remains if old scripts used or not followed

**Consequence:** Inflated validation metrics, poor real-world generalization

**Mitigation:**
- Always fit scaler on train set ONLY
- Verify scaler is loaded from `scaler_{Ex}.joblib` during inference (per-fold scalers not saved)

---

#### 2. **Downsampling Order (Critical Invariant)**
**Risk:** If downsampling applied AFTER StandardScaler, distribution changes and model fails.

**Current Status:** 
- ✅ Correct in all pipelines (downsample → scale)
- ⚠️ Must be maintained in any refactors

**Consequence:** Feature distribution completely different from training; model predicts garbage

**Validation:**
```python
# Correct order:
features = extract_features()
features = temporal_downsample(features, stride=5, target_len=150)  # BEFORE scaling
features_scaled = scaler.transform(features)

# WRONG:
features_scaled = scaler.transform(extract_features())
features_downsampled = temporal_downsample(features_scaled, ...)  # BREAKS
```

---

#### 3. **Fixed Max Sequence Length (Model Invariant)**
**Risk:** Model input shape is locked at (batch, 150, F). Cannot change without retraining.

**Current Status:** 
- ✅ Hardcoded in multiple places (prevents accidental changes)
- ⚠️ Mismatch between training and inference will crash model.predict()

**Consequence:** ValueError: incompatible tensor shapes during inference

**Safeguard:**
```python
# In main.py
MAX_LENGTH_MAPPING = {
    "Es1": 150,
    "Es2": 150,
    ...
}

# Validate preprocessing shape
expected_tail = _get_model_input_tail_shape(model)  # (150, num_features)
actual_tail = tuple(prepared.shape[1:])
if expected_tail != actual_tail:
    raise ValueError(f"Shape mismatch: {expected_tail} vs {actual_tail}")
```

---

#### 4. **Masking Value Must Be -999.0 (Not 0.0)**
**Risk:** Post-scaling, feature mean = 0.0. If mask_value = 0.0, LSTM cannot distinguish real zeros from padding.

**Current Status:** 
- ✅ Correctly set to -999.0 in both training and inference
- ⚠️ Documentation may not be clear

**Consequence:** Model learns to ignore real features that happen to be 0.0; poor performance

**Evidence:**
```python
# After StandardScaler: feature_scaled = (feature - mean) / std
# Mean feature value post-scaling ≈ 0.0
# So if mask_value = 0.0, LSTM masks real data!

# Correct:
mask_value = -999.0  # Far outside [-3, 3] typical post-scaling range
```

---

#### 5. **Training Dataset Imbalance**
**Risk:** ~72 samples total per exercise; clinical scores likely not uniformly distributed (e.g., more healthy subjects → clustered scores around 40-50).

**Current Status:** 
- ⚠️ No stratified sampling mentioned in CV folds
- ⚠️ Potential fold imbalance

**Consequence:** Some folds over/under-represented for certain score ranges; poor generalization on underrepresented scores

**Mitigation:**
- Use `StratifiedKFold` based on score bins
- Monitor per-fold score distributions
- Report cross-validation variance

---

### 🟡 MEDIUM SEVERITY

#### 6. **Aspect Ratio Correction Factor (1.7778)**
**Risk:** Hardcoded to 1920/1080. May not match actual video resolution.

**Current Status:** 
- ⚠️ Hardcoded; no validation of actual video dimensions
- ✅ 16:9 is standard, likely safe

**Consequence:** If videos are 4:3 (1.333) or other aspect ratio, angle calculations wrong

**Mitigation:**
```python
# Better: detect actual aspect ratio
width, height = get_video_dimensions(video_path)
aspect_ratio = width / height
x_corrected = x_normalized * aspect_ratio

# Current: assumes 16:9
x_corrected = x_normalized * 1.7778
```

---

#### 7. **Feature Extraction NaN Handling**
**Risk:** Missing keypoints (NaN in input) filled with 0.0 → assumed rest position. May introduce bias.

**Current Status:** 
- ⚠️ Conservative fill (0.0), but no logging of NaN count per sample
- ✅ Debug output available in ml_wrapper.py

**Consequence:** Samples with many missing joints still used; features may be meaningless

**Mitigation:**
- Log NaN percentages per sample
- Skip samples with >20% missing keypoints
- Track missing joint patterns (which joints fail most often?)

---

#### 8. **Motion Detection Thresholds (Uniform Across Exercises)**
**Risk:** All exercises use identical thresholds (min_variance=5.0, min_rom=8.0, min_displacement=0.5). May not reflect exercise-specific motion profiles.

**Current Status:** 
- ⚠️ Thresholds not validated or tuned
- ✅ Thresholds conservative (not requiring excessive motion)

**Consequence:** False negatives (classify real motion as static) or false positives (classify noise as motion)

**Tuning needed:**
- Analyze motion profiles per exercise (variance/ROM/displacement distributions)
- Set exercise-specific thresholds at 2× or 3× typical values
- Monitor false positive/negative rates

---

#### 9. **Clinical Score Range Mismatch**
**Risk:** Training labels are "clinical_score" (0-50 from KiMoRe), but display scale may be 0-100 (multiplied by 2).

**Current Status:** 
- ⚠️ Scaling factor (×2) applied in inference; may not be intentional
- ❓ No documentation of why ×2

**Consequence:** Confusion between model output (0-50) and display score (0-100)

**Code:**
```python
def _auto_scale_score(raw_out):
    if raw_out <= 1.5:
        raw_ts = raw_out * 50.0  # Sigmoid [0,1] → [0,50]
    else:
        raw_ts = raw_out
    return float(np.clip(raw_ts, 0, 50) * 2.0)  # ← ×2 HERE (undocumented)
```

---

#### 10. **LSTM with Sigmoid Output (Potential Collapse)**
**Risk:** Sigmoid + small dataset (72 samples) + aggressive dropout may cause model to collapse to constant output (~0.5).

**Current Status:** 
- ✅ Mitigations in place: Huber loss, gradient clipping, LR warmup
- ✅ Training logs indicate model learning (not collapsed)
- ⚠️ Monitor during training

**Consequence:** Model predicts identical scores regardless of input (stuck at 0.5 after scaling)

**Monitoring:**
- Log min/max/mean predictions per epoch
- Alert if std(predictions) < 0.05 (indicates collapse)
- Use LR warmup (start from lr/100, ramp up)

---

### 🟢 LOW SEVERITY

#### 11. **Cold Start Latency**
**Risk:** Backend loads all 5 LSTM models + scalers at startup (TensorFlow lazy loading overhead).

**Current Status:** 
- ✅ Models cached in memory after first load
- ✅ Startup health check allows retry
- ⚠️ First request may see high latency

**Consequence:** User experience: delay before first feedback (~1-3 seconds)

**Mitigation:**
- Pre-warm models on startup (dummy prediction)
- Use Gunicorn workers for parallel requests
- Cache scaler objects (already done)

---

#### 12. **Missing Video Files in Dataset**
**Risk:** Some KiMoRe dataset entries have missing video files; recorded as NaN but may cause confusion.

**Current Status:** 
- ✅ Tracked in metadata CSV
- ✅ Reported in statistics

**Consequence:** Training set smaller than expected if filtered by non-NaN video

**Current approach:** Proceed with available data; videos not needed for training (only joint CSVs used)

---

#### 13. **PoseLandmarker Model Variant (Heavy)**
**Risk:** Using "heavy" model (slowest, most accurate). Data preparation slow; inference unnecessary.

**Current Status:** 
- ✅ Acceptable for offline data prep (runs once)
- ❌ Could switch to "full" or "lite" for faster prep

**Consequence:** Data preparation takes longer; no impact on real-time inference (MediaPipe not used at inference time)

---

## 8. DEBUGGING CHECKLIST

### If Model Predicts Constant Scores
1. ✅ Check scaler was fit on train set only (not entire dataset)
2. ✅ Verify downsampling applied BEFORE scaling
3. ✅ Ensure mask_value = -999.0 (not 0.0)
4. ✅ Check LR warmup is enabled (prevents initial collapse)
5. ✅ Monitor training curves (loss should decrease)

### If Inference Crashes
1. ✅ Check prepared data shape: (1, 150, F)
2. ✅ Verify model.input_shape matches
3. ✅ Ensure scaler_{ExX}.joblib exists and matches feature count
4. ✅ Check for NaN/inf in prepared data after scaling

### If Scores Seem Too High/Low
1. ✅ Verify motion quality factor is reasonable (not 0.1 for real motion)
2. ✅ Check if output scaling (×50, ×2) is correct
3. ✅ Compare inference output with training metrics (MAE, R²)
4. ✅ Validate clinical score labels are in correct range [0, 50]

### If Motion Detection Always Inactive
1. ✅ Check feature values are non-zero (keypoint extraction working)
2. ✅ Verify motion thresholds are not too aggressive
3. ✅ Confirm temporal downsampling preserves motion signature
4. ✅ Check if exercise requires larger movements (Es4 pelvis rotation is subtle)

---

## 9. SUMMARY TABLE: Data & Model Pipeline

| Stage | Input | Process | Output | Critical Invariants |
|-------|-------|---------|--------|---------------------|
| **1** | MP4 video | MediaPipe PoseLandmarker (Heavy) | `*_mediapipe.csv` (48 cols) | MediaPipe 3D; body-only 12 joints |
| **2** | Joint CSVs | Metadata assembly + clinical score lookup | `KiMoRe_final.csv` | Flexible path lookup; recursive search |
| **3** | Joint positions | 2D feature extraction + aspect ratio correction + StandardScaler | Per-sample CSVs + `scaler_{ExX}.joblib` | Exercise-specific features; scaler fit on train only |
| **4** | Features | 5-fold CV: downsample → scale → pad → LSTM train | `ml_model_Es*.keras` | Downsample BEFORE scale; max_length=150; mask_value=-999 |
| **5** | Webcam frames | Feature extraction → downsample → scale → pad → LSTM inference | Clinical score [0, 100] | Matches training pipeline exactly |

---

## 10. RECOMMENDATIONS FOR IMPROVEMENTS

1. **Add comprehensive logging:** Per-sample NaN counts, motion metrics, feature statistics
2. **Implement stratified K-fold:** Ensure score range distribution balanced across folds
3. **Exercise-specific motion thresholds:** Tune based on actual motion profiles
4. **Aspect ratio detection:** Read from video metadata instead of hardcoding 1.7778
5. **Model input shape validation:** Auto-detect and assert match at inference
6. **Per-fold scaler artifacts:** Save training-fold scalers for reproducibility (optional)
7. **Continuous model monitoring:** Track inference score distributions for drift
8. **Documentation of scaling factors:** Clarify why ×50 and ×2 (or remove if unintended)
9. **Unit tests for pipeline:** Ensure downsampling order, scaler fit, masking all work correctly
10. **Data augmentation:** Consider rotation/scale jitter to increase effective training set size

---

**End of Analysis**

Generated with comprehensive code review of all data preparation, training, and inference pipelines.
