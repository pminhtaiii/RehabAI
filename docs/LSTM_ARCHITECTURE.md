# RehabAI LSTM Architecture - Kiến Trúc Chi Tiết

## 📊 **Tổng Quan Pipeline**

```
[Frontend Camera]
      ↓ (MoveNet 640×480 @ 30fps)
[17 keypoints]
      ↓ (Extract x, y, confidence)
[30 frames] → CSV
      ↓
[Backend - ml_wrapper.py]
      ↓ (Extract 2D features)
[Feature Array: 6-9 features per exercise]
      ↓ (Temporal downsampling stride=5)
[150 frames]
      ↓ (StandardScaler transform)
[Normalized features]
      ↓
[LSTM Model] ← **THIS DOCUMENT**
      ↓ (Output: sigmoid 0-1)
[Clinical Score 0-100]
```

---

## 🧠 **LSTM Architecture (Practical v6)**

### **Layer 1: Input Layer (Tầng Đầu Vào)**

```
Input Shape: (batch_size, sequence_length=150, num_features)

Per-Exercise Feature Count:
  - Es1: 6 features (left_elbow_angle, right_elbow_angle, hand_shoulder_ratio, ...)
  - Es2: 6 features
  - Es3: 9 features (most complex)
  - Es4: 2 features (simplest)
  - Es5: 7 features

Total input parameters = batch × 150 × F (F = 2 to 9)

Example (Es1):
  Input shape: (32, 150, 6)  ← 32 samples, 150 timesteps, 6 features
```

**Masking Layer - Xử Lý Padded Sequences:**
```python
x = Masking(mask_value=-999.0)(inputs)

Purpose:
  - Ignore padded frames (value = -999.0)
  - Prevent padding from affecting training
  - LSTM skip masked timesteps automatically
  
Example:
  Input sequence: [f0, f1, f2, ..., f149]
  Some are real (features), others are -999 (padding)
  Masking layer: LSTM only processes real frames
```

---

### **Layer 2: LSTM Layers (Tầng LSTM Chính)**

#### **First LSTM Layer - LSTM_1 (32 neurons)**

```python
x = LSTM(32, return_sequences=True, name='lstm_1')(x)

Configuration:
  - Units: 32 (neurons/hidden state size)
  - return_sequences: True → output ALL timesteps
  - Input shape: (batch, 150, 6)
  - Output shape: (batch, 150, 32)

Function:
  - Learns temporal patterns across 150 timesteps
  - Each neuron captures different motion characteristics
  - Example patterns learned:
    * "Smooth vs jerky motion"
    * "Increasing vs decreasing angle"
    * "Symmetric vs asymmetric movement"
    
Why return_sequences=True:
  → Need intermediate outputs for second LSTM layer
  → Each timestep gets 32-dimensional representation
```

**Dropout Layer 1 - Drop_1**

```python
x = Dropout(0.3, name='drop_1')(x)

Function:
  - Randomly disable 30% of neurons during training
  - Purpose: Prevent overfitting on small dataset (72 samples)
  - Output shape: (batch, 150, 32)

Dropout Rate 0.3 Justification:
  - 0.2: Too weak (still overfit)
  - 0.3: Sweet spot for small dataset
  - 0.5+: Too aggressive (underfitting)
```

#### **Second LSTM Layer - LSTM_2 (16 neurons)**

```python
x = LSTM(16, return_sequences=True, name='lstm_2')(x)

Configuration:
  - Units: 16 (fewer than LSTM_1 = "funnel" architecture)
  - return_sequences: True → output ALL timesteps
  - Input shape: (batch, 150, 32)
  - Output shape: (batch, 150, 16)

Funnel Architecture (32 → 16):
  Rationale:
    - First LSTM captures "raw" temporal patterns
    - Second LSTM compresses to high-level features
    - Mimics biological neural hierarchy
  
  Benefits:
    - Reduces model parameters → fit on small dataset
    - Forces information bottleneck (learns important features)
    - Prevents overfitting
```

**Dropout Layer 2 - Drop_2**

```python
x = Dropout(0.3, name='drop_2')(x)

Same as Dropout_1:
  - 30% neurons disabled during training
  - Output shape: (batch, 150, 16)
```

---

### **Layer 3: Temporal Aggregation (Tầng Tổng Hợp Thời Gian)**

```python
x = GlobalAveragePooling1D(name='gap')(x)

Input: (batch, 150, 16)
Output: (batch, 16)

Function:
  - Average across ALL 150 timesteps
  - Each feature dimension becomes 1 value
  
Example:
  x[0] = [0.1, 0.5, -0.2, ..., 0.8]  (150 values)
  gap(x[0]) = mean([0.1, 0.5, -0.2, ..., 0.8]) = 0.35
  
Why NOT Last Timestep?
  ❌ LSTM(return_sequences=False) → only uses last frame
     Problem: Motion quality depends on WHOLE sequence, not just end
  
  ✓ GlobalAveragePooling1D → uses ALL frames
     Better: "Average motion quality across entire exercise"

Alternative - Could Use:
  - GlobalMaxPooling1D: "Peak motion" instead of average
  - Attention layer: "Focus on important frames"
  (But GAP simpler + works well for small dataset)
```

---

### **Layer 4: Dense Layers (Tầng Kết Nối Đầy Đủ)**

#### **Dense_1 - Bottleneck Layer**

```python
x = Dense(8, activation='relu', name='dense_1')(x)

Configuration:
  - Units: 8 neurons
  - Activation: ReLU (Rectified Linear Unit)
  - Input shape: (batch, 16)
  - Output shape: (batch, 8)

Function:
  - Learns non-linear combinations of LSTM features
  - ReLU: max(0, x) → allows modeling complex interactions
  
  Example learned patterns:
    "If (angle1 - angle2) > 5° AND motion_energy > 0.3: good_form"

Why 8 units?
  - Less than input (16) → compression
  - Enough parameters to learn task-specific features
  - Small enough to avoid overfitting
```

#### **Output Layer - Output**

```python
outputs = Dense(1, activation='sigmoid', name='output')(x)

Configuration:
  - Units: 1 (single prediction)
  - Activation: Sigmoid (σ(x) = 1/(1+e^-x))
  - Input shape: (batch, 8)
  - Output shape: (batch, 1)

Sigmoid Function:
  - Maps any input → [0, 1] range
  - σ(-∞) → 0, σ(0) → 0.5, σ(+∞) → 1
  
  Output Meaning:
    0.0-1.0 (sigmoid) → Multiply by 100 → Clinical Score [0-100]
    
    Example:
      model_output = 0.65
      clinical_score = 0.65 × 100 = 65/100
      
Why Sigmoid (Not Linear)?
  ✓ Sigmoid: Bounded output (0-1) → prevents score collapse
  ✓ Sigmoid: Matches training normalization (y/50)
  ✗ Linear: Unbounded → can output -500 or +10000 (useless)
```

---

## 📋 **Complete Model Summary**

```
Model: Clinical Score Predictor v6
─────────────────────────────────────────────────────────────

Input (None, 150, 6)
  ↓
Masking (mask_value=-999.0)
  ↓
LSTM_1 (32 units, return_sequences=True)  [Output: (None, 150, 32)]
  ↓
Dropout_1 (0.3)
  ↓
LSTM_2 (16 units, return_sequences=True)  [Output: (None, 150, 16)]
  ↓
Dropout_2 (0.3)
  ↓
GlobalAveragePooling1D  [Output: (None, 16)]
  ↓
Dense_1 (8 units, relu)  [Output: (None, 8)]
  ↓
Output (1 unit, sigmoid)  [Output: (None, 1) → Clinical Score]

─────────────────────────────────────────────────────────────

Total Parameters: ~2,000 (small for small dataset)

Training Configuration:
  - Loss Function: Huber (smooth + robust)
  - Optimizer: Adam (learning rate=0.001, clipvalue=0.5)
  - Metrics: Mean Absolute Error (MAE)
  - Validation: 5-Fold Cross-Validation
```

---

## 🔄 **Data Flow During Training**

```python
# Step 1: Load raw features
X_train.shape = (40, 150, 6)  # 40 samples, 150 frames, 6 features per Es1
y_train.shape = (40,)         # Clinical scores 0-100

# Step 2: Model forward pass
x = Masking()(X_train)                    # (40, 150, 6) → skip -999
x = LSTM(32, return_sequences=True)(x)    # (40, 150, 6) → (40, 150, 32)
x = Dropout(0.3)(x)                       # (40, 150, 32) → (40, 150, 32) [30% dropped]
x = LSTM(16, return_sequences=True)(x)    # (40, 150, 32) → (40, 150, 16)
x = Dropout(0.3)(x)                       # (40, 150, 16) → (40, 150, 16) [30% dropped]
x = GlobalAveragePooling1D()(x)           # (40, 150, 16) → (40, 16) [average over timesteps]
x = Dense(8, relu)(x)                     # (40, 16) → (40, 8)
y_pred = Dense(1, sigmoid)(x)             # (40, 8) → (40, 1) [output: 0-1]

# Step 3: Loss & Backprop
y_normalized = y_train / 50                      # [0-100] → [0-1]
loss = huber_loss(y_normalized, y_pred)
loss.backward()
optimizer.step()

# Step 4: Output conversion
clinical_score = y_pred * 100              # [0-1] → [0-100]
```

---

## 🎯 **Inference Flow**

```
Frontend (Exercise.jsx)
  ↓ 30 frames @ 30fps + MoveNet
  ↓ Send CSV (left_shoulder_x, left_shoulder_y, ...)
  
Backend (ml_wrapper.py)
  ↓ Parse CSV
  ↓ Extract features (2D angles, distances)
  ↓ Temporal downsampling (stride=5) → 6 frames
  ↓ StandardScaler transform (use scaler_Es1.joblib)
  ↓ Pad to 150 frames with -999.0
  
LSTM Inference
  ↓ Input: (1, 150, 6) shape
  ├─ Masking: ignore 144 padded frames
  ├─ LSTM_1(32): process 6 real frames + 144 masked frames → 150 timesteps
  ├─ LSTM_2(16): compress 32 → 16 dimensions
  ├─ GAP: average across 150 timesteps → 16 values
  ├─ Dense(8): learn complex patterns
  └─ Output Sigmoid: predict 0-1
  
  ↓ Output = 0.72 (sigmoid)
  ↓ Clinical Score = 0.72 × 100 = 72/100
```

---

## ⚠️ **Key Design Decisions**

| Decision | Rationale | Impact |
|----------|-----------|--------|
| **2 LSTM layers (32→16)** | Funnel learns hierarchical features | Prevents overfitting on 72 samples |
| **return_sequences=True** | Need intermediate outputs for 2nd LSTM | Captures motion throughout full sequence |
| **GlobalAveragePooling1D** | Average quality across entire motion | Better than just last timestep |
| **Dropout 0.3** | Balance overfitting reduction vs underfitting | Tested on small dataset |
| **Sigmoid activation** | Bounds output [0, 1] | Prevents score collapse |
| **Huber loss** | Smooth + robust to outliers | Better than MSE for small dataset |
| **Adam optimizer LR=0.001** | Moderate learning rate | Faster convergence + stable training |
| **Masking with -999.0** | Skip padded frames | Match training preprocessing |

---

## 🧪 **Testing Model Output**

```python
# Verify output diversity (avoid output collapse)
test_inputs = [
    np.zeros((1, 150, 6)),      # all zeros
    np.ones((1, 150, 6)),       # all ones
    np.random.randn(1, 150, 6)  # random
]

outputs = [model.predict(inp) for inp in test_inputs]
# Expected: [0.1-0.3, 0.4-0.6, 0.5-0.9] (diverse)
# Bad: [0.53, 0.53, 0.53] (output collapse!)
```

---

**Cập nhật: May 4, 2026**
