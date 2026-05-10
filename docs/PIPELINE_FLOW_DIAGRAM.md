# RehabAI Pipeline Flow Diagram - Visual Architecture

## System-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REHABILAI SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ FRONTEND (React 18 + MediaPipe Tasks Vision)                    │    │
│  │ • PoseLandmarker FULL (GPU delegate, ~25 FPS)                   │    │
│  │ • 12 body joints × (x,y,z,v) = 48 columns per frame           │    │
│  │ • Webcam capture + real-time visualization                      │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                             │
│                    REST API + WebSocket                                   │
│                             │                                             │
│  ┌──────────────────────────▼──────────────────────────────────────┐    │
│  │ BACKEND (FastAPI + TensorFlow 2.21)                             │    │
│  │ • Feature extraction (joint_features.py)                        │    │
│  │ • Motion detection (motion_detector.py)                         │    │
│  │ • LSTM inference (ml_wrapper.py)                                │    │
│  │ • Real-time feedback generation                                 │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                             │
│                  SQLAlchemy ORM / SQLite                                  │
│                             │                                             │
│  ┌──────────────────────────▼──────────────────────────────────────┐    │
│  │ DATABASE (SQLite)                                                │    │
│  │ • Users, Exercises, Progress Tracking                           │    │
│  │ • Clinical scores and motion metrics                            │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline - 5 Stages

### STAGE 1: Video → Joint Extraction
```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Raw MP4 Video Files (KiMoRe Dataset)                    │
│ Location: 01_raw_data/{group}/{expertise}/{subject_id}/{Es}    │
│ Format: H.264, ~25 FPS, variable length (20-30 seconds)        │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ PROCESS: 01_extract_joint_positions.py                         │
│ ─────────────────────────────────────────────────────────────  │
│ • Download MediaPipe PoseLandmarker (Heavy model)              │
│ • Per frame: detect 33 landmarks → extract 12 body joints     │
│ • Output per joint: x, y, z, visibility (4 channels)          │
│ • Safety: Global monotonic timestamp                           │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ OUTPUT: {video_name}_mediapipe.csv                             │
│ Location: 03_raw_joints/{group}/{expertise}/{subject_id}/{Es}  │
│ Format: (N_frames, 48 columns)                                 │
│   Columns: {joint}_x, {joint}_y, {joint}_z, {joint}_v          │
│   Example: left_shoulder_x, left_shoulder_y, ...              │
│ Data type: float32                                             │
│                                                                 │
│ Coordinates:                                                    │
│   x, y ∈ [0, 1] (normalized)                                   │
│   z ∈ [-1, 2] (depth relative to hip)                          │
│   v ∈ [0, 1] (visibility/confidence)                           │
└────────────────────────────────────────────────────────────────┘
```

### STAGE 2: Joints → Metadata Dataset
```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Joint CSVs + Clinical Assessment Excel Files           │
│ • Joint CSVs: 03_raw_joints/{group}/{expertise}/{Id}/{Es}/     │
│ • Clinical scores: 02_clinical_labels/{group}/{expertise}/{Id} │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ PROCESS: 02_prepare_dataset.py                                 │
│ ─────────────────────────────────────────────────────────────  │
│ For each {group}/{expertise}/{subject_id}/{exercise}:         │
│   1. Find *_mediapipe.csv (joint positions)                    │
│   2. Locate corresponding MP4 video                            │
│   3. Read clinical score from Excel                            │
│   4. Count frames in CSV                                       │
│   5. Build metadata row                                        │
│                                                                │
│ Column lookup for clinical scores:                             │
│   File: ClinicalAssessment_*.xlsx                              │
│   Column: "clinical TS Ex#{exercise_num}"                      │
│   Range: [0, 50] (Total Score, not sum of subscores)           │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ OUTPUT: KiMoRe_final.csv                                       │
│ Location: 05_final_datasets/                                   │
│ Format: (N_samples, 8 columns)                                 │
│                                                                │
│ Columns:                                                        │
│  1. ID:                 Subject identifier                     │
│  2. clinical_group:     "healthy" or "patient"                 │
│  3. expertise:          "professional", "novice", etc.         │
│  4. exercise:           "Es1", "Es2", ..., "Es5"              │
│  5. video:              Path to MP4 (may be NaN)               │
│  6. joint_positions:    Path to *_mediapipe.csv                │
│  7. clinical_score:     [0, 50] or NaN                         │
│  8. #frames:            Integer frame count                    │
│                                                                │
│ Stats per exercise:                                             │
│   ~72 total samples (varies by Es)                             │
│   Score range: typically 20-50                                 │
│   Frame count: 300-600 frames (12-24s @ 25fps)                │
└────────────────────────────────────────────────────────────────┘
```

### STAGE 3: Joints → 2D Features + Scalers
```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: KiMoRe_final.csv                                        │
│ For each row:                                                  │
│   • Load joint_positions CSV (N_frames, 48 columns)           │
│   • Load clinical_score label                                 │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ PROCESS: 03_extract_joint_features.py                          │
│ ─────────────────────────────────────────────────────────────  │
│                                                                │
│ STEP 1: Aspect Ratio Correction                               │
│  For 16:9 video: x_corrected = x_raw * 1.7778                 │
│  Purpose: Make 2D space isotropic for angle calculations      │
│                                                                │
│ STEP 2: Exercise-Specific Feature Extraction                  │
│  Es1 (6 features):                                             │
│   • left_elbow_angle: angle at elbow vertex                   │
│   • right_elbow_angle                                          │
│   • hand_shoulder_ratio: dist(hands) / dist(shoulders)        │
│   • torso_tilted_angle: angle between torso and vertical      │
│   • hand_tilted_angle: angle between hand vector and horiz.   │
│   • elbow_angles_diff: |left - right| elbow angle            │
│                                                                │
│  Es2 (6 features):                                             │
│   • left_elbow_angle, right_elbow_angle                        │
│   • torso_tilted_angle, elbow_angles_diff                      │
│   • left_shoulder_angle, right_shoulder_angle                  │
│                                                                │
│  Es3 (9 features):                                             │
│   • All of Es2 (6) + left_arm_torso_angle +                   │
│   • right_arm_torso_angle                                      │
│                                                                │
│  Es4 (2 features):                                             │
│   • torso_tilted_angle, knee_hip_ratio                         │
│                                                                │
│  Es5 (7 features):                                             │
│   • Es1 (6) + left_shoulder_angle + right_shoulder_angle      │
│                                                                │
│ STEP 3: StandardScaler Fit (Per-Exercise, Per-Fold)           │
│  • Fit ONLY on training set (prevents data leakage)           │
│  • Transform: (feature - mean) / std                          │
│  • Output: zero-mean, unit-variance features                  │
│  • Save: scaler_Es*.joblib                                    │
│                                                                │
│ Geometry Functions Used:                                       │
│  • Angle: atan2(cross_product, dot_product) → degrees [0,180]│
│  • Distance: Euclidean norm with 1e-8 epsilon                 │
│  • Ratio: distance_a / (distance_b + 1e-8)                    │
│  • Vector: endpoint - startpoint                              │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ OUTPUT: Two artifact types                                     │
│                                                                │
│ A) Feature CSVs (per sample)                                   │
│    Location: 05_final_datasets/{Id}_{exercise}_features.csv    │
│    Format: (N_frames, F) where F = 2-9 features              │
│    Data: Already scaled (mean=0, std=1)                       │
│    Example Es1:                                                │
│      Shape: (600, 6)                                           │
│      Columns: left_elbow_angle, right_elbow_angle, ...        │
│                                                                │
│ B) StandardScaler Artifacts (per exercise)                     │
│    Location: models/scaler_Es*.joblib                          │
│    Contains:                                                   │
│      • mean_: [μ₀, μ₁, ..., μₓ]                              │
│      • scale_: [σ₀, σ₁, ..., σₓ]                              │
│      • n_features_in_: F (2-9)                               │
│    Usage: (feature - mean_) / scale_                          │
└────────────────────────────────────────────────────────────────┘
```

### STAGE 4: Features → LSTM Model Training
```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Feature CSVs + Clinical Scores                          │
│ Location: 05_final_datasets/                                   │
│ Dataset size: ~72 samples per exercise                         │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ PROCESS: training_models/clinical_score_prediction_model.py    │
│ ─────────────────────────────────────────────────────────────  │
│                                                                │
│ 5-Fold Cross-Validation Pipeline:                             │
│                                                                │
│ For each fold:                                                │
│                                                                │
│  ┌─ Load Raw Features                                          │
│  │  • Variable sequence lengths: 100-500 frames               │
│  │  • Shape: (N_train, T_i, F) where T_i varies              │
│  │                                                            │
│  ├─ Split: train_idx (80%), val_idx (20%)                     │
│  │                                                            │
│  ├─ Temporal Downsample (stride=5)                            │
│  │  • 25fps → 5fps                                            │
│  │  • 600 frames → ~120 frames                               │
│  │  • Strategy if > 150: keep head + tail, discard middle    │
│  │                                                            │
│  ├─ Fit StandardScaler on TRAIN SET ONLY                      │
│  │  • Prevents validation set leakage                         │
│  │  • Scaler not saved (per-fold only)                       │
│  │                                                            │
│  ├─ Transform Features                                        │
│  │  • Apply scaler to train + validation                      │
│  │  • Output: mean=0, std=1                                  │
│  │                                                            │
│  ├─ Pad/Truncate to max_length=150                            │
│  │  • Shape: (N_train, 150, F) and (N_val, 150, F)           │
│  │  • Padding value: -999.0 (sentinel)                       │
│  │  • Replace NaN/inf: nan_to_num(..., nan=-999)             │
│  │                                                            │
│  ├─ Build LSTM Model                                          │
│  │  • Input: (batch, 150, F)                                 │
│  │  • Masking(mask_value=-999.0)                             │
│  │  • LSTM(32, return_sequences=True) + Dropout(0.3)         │
│  │  • LSTM(16, return_sequences=False) + Dropout(0.3)        │
│  │  • Dense(8, relu) + Dense(1, sigmoid)                     │
│  │  • Loss: Huber | Optimizer: Adam(lr=0.001)                │
│  │  • Gradient clipping: 0.5                                 │
│  │                                                            │
│  ├─ Train                                                     │
│  │  • Epochs: 200 (max) with early stopping                  │
│  │  • Batch size: 16                                         │
│  │  • Learning rate warmup: from 0.00001 to 0.001 (20 ep)   │
│  │  • Callbacks:                                             │
│  │    - ReduceLROnPlateau: factor=0.5, patience=10           │
│  │    - EarlyStopping: patience=20                           │
│  │                                                            │
│  └─ Evaluate on Validation Set                                │
│     • Metrics: MAE, MSE, R², Pearson/Spearman correlation   │
│                                                                │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ OUTPUT: Trained LSTM Models                                    │
│ Location: models/ml_model_Es*.keras                            │
│                                                                │
│ Per exercise (Es1-Es5):                                        │
│  • Model weights: trained via 5-fold CV                        │
│  • Input shape: (batch, 150, F) where F=2-9                  │
│  • Output: float ∈ [0, 1] (sigmoid)                           │
│  • Conversion to score: multiply by 50 → [0, 50]             │
│  • Display scale: multiply by 2 → [0, 100]                   │
│                                                                │
│ Model statistics:                                              │
│  • Es1: 3,200 parameters                                       │
│  • Es3: 4,000 parameters (largest)                            │
│  • Es4: 2,400 parameters (smallest)                           │
└────────────────────────────────────────────────────────────────┘
```

### STAGE 5: End-of-Session Inference
```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Complete 30s Exercise Recording                         │
│ Source: Frontend MediaPipe PoseLandmarker (Full)               │
│ Format: ~900 frames @ 30fps, 48 columns (12 body joints ×4)   │
│ Columns: {joint}_x, {joint}_y, {joint}_z, {joint}_v           │
│ Timing: sent ONCE after 30s session completes                  │
└────────────┬─────────────────────────────────────────────────────┘
             │
        REST POST /api/clinical_score/{exercise_id}
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ Backend Pipeline (unchanged)                                   │
│ ─────────────────────────────────────────────────────────────  │
│                                                                │
│ 1. Parse CSV → reorder to 12 MediaPipe joints (48 cols)       │
│ 2. Feature extraction: 2D angles/distances (only x,y used)    │
│ 3. Temporal downsample stride=5: ~900 → ~180 frames           │
│ 4. Motion detection on raw features                            │
│ 5. StandardScaler transform                                    │
│ 6. Pad/truncate to 150 with -999.0 sentinel                   │
│ 7. LSTM inference → [0,1] → ×50 → ×2 = [0,100]              │
│                                                                │
│ Total latency: 60-260ms (single request after session)        │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ OUTPUT: Final Clinical Score + Motion Metrics                  │
│ HTTP Response (JSON):                                          │
│ {                                                              │
│   "score": 75,              ← Display score [0, 100]           │
│   "clinical_score": 37.5,   ← Raw clinical score [0, 50]       │
│   "motion_energy": 0.85,    ← Motion intensity [0, 1]          │
│   "quality_factor": 0.95,   ← Quality multiplier [0.1, 1.0]   │
│   "is_active": true,        ← Exercise detected                │
│ }                                                              │
└────────────────────────────────────────────────────────────────┘
```

---

## Critical Invariants & Data Flow

```
╔════════════════════════════════════════════════════════════════╗
║ CRITICAL INVARIANTS (DO NOT CHANGE WITHOUT RETRAINING)        ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║ 1. max_length = 150  ← Model input shape locked               ║
║    (batch, 150, num_features)                                 ║
║                                                                ║
║ 2. downsample_stride = 5  ← Temporal decimation               ║
║    Order: extract → downsample(5) → scale                     ║
║    NOT: extract → scale → downsample                          ║
║                                                                ║
║ 3. mask_value = -999.0  ← Padding sentinel                    ║
║    (Not 0.0! Post-scaling mean ≈ 0.0)                         ║
║                                                                ║
║ 4. aspect_ratio_factor = 1.7778  ← 16:9 video                │
║    x_corrected = x_normalized * 1.7778                        ║
║                                                                ║
║ 5. scaler fit ONLY on train set per fold                      │
║    Prevents validation data leakage                           ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## Data Quality & Validation Flow

```
Stage 1 Validation       Stage 2 Validation       Stage 3 Validation
(Joint Extraction)      (Metadata)               (Features)
│                       │                        │
├─ Frame count ≥ 5      ├─ CSV exists            ├─ NaN handling
├─ 12 joints detected   ├─ Video found (or NaN)  ├─ Div-by-zero guards
├─ x,y ∈ [0,1]         ├─ Score found (or NaN)  ├─ NaN/inf cleanup
├─ z ∈ [-1,2]          ├─ Frame count extracted ├─ Feature range check
└─ visibility logged    └─ Per-exercise stats    └─ Scaler validation
                                                  
Stage 4 Validation       Stage 5 Validation
(Training)              (Inference)
│                       │
├─ Sequence lengths OK  ├─ Input shape (1,150,F)
├─ Score ranges good    ├─ Feature count matches
├─ Scaler fit on train  ├─ Scaler artifact exists
├─ Model training loss  ├─ No NaN in prepared data
└─ Val metrics reported └─ Inference output ∈ [0,1]
```

---

**End of Flow Diagram**
