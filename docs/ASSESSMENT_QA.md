# RehabAI - Assessment Q&A Bank

**For Teachers/Interviewers** - Comprehensive question set for evaluating understanding of RehabAI architecture

---

## 📋 Table of Contents
1. [Cơ Bản (Beginner)](#cơ-bản-beginner)
2. [Trung Bình (Intermediate)](#trung-bình-intermediate)
3. [Nâng Cao (Advanced)](#nâng-cao-advanced)
4. [Thiết Kế Kiến Trúc (Architecture Design)](#thiết-kế-kiến-trúc-architecture-design)
5. [Troubleshooting & Debugging](#troubleshooting--debugging)

---

## 🟢 Cơ Bản (Beginner)

### Q1: Dự án RehabAI được dùng để làm gì?
**Trả lời mong đợi:**
RehabAI là một hệ thống phục hồi chức năng AI-powered giúp bệnh nhân thực hiện các bài tập vật lý trị liệu với phản hồi real-time. Hệ thống:
- Sử dụng MoveNet (pose detection) để theo dõi chuyển động của bệnh nhân
- Tính toán điểm lâm sàng (0-100) dựa trên chất lượng thực hiện bài tập
- Cung cấp phản hồi real-time bằng tiếng Việt để điều chỉnh tư thế
- Theo dõi tiến độ phục hồi theo thời gian

**Điểm cộng:**
- Nếu họ đề cập đến 5 loại bài tập (Es1-Es5): lifting, lateral tilt, trunk rotation, pelvis rotation, squatting

---

### Q2: Có bao nhiêu mô hình pose detection được sử dụng và chúng được dùng ở đâu?
**Trả lời mong đợi:**
2 mô hình:
1. **MoveNet LIGHTNING** - Frontend (browser), chạy real-time @ 30 FPS qua TensorFlow.js, output 17 keypoints với confidence scores
2. **MediaPipe PoseLandmarker** - Backend (data preparation), dùng cho training data, output 33 keypoints với tọa độ 3D

**Câu hỏi follow-up:**
- Tại sao cần hai cái khác nhau?
  - MoveNet: nhanh, nhẹ, tối ưu cho browser
  - MediaPipe: chính xác hơn, 3D information, tối ưu cho training
- Điều này có tạo ra vấn đề không? (→ Keypoint index mismatch)

---

### Q3: Bài tập Es1 là gì? Nó sử dụng bao nhiêu features?
**Trả lời mong đợi:**
- **Es1 = Lifting of arms (Nâng tay)**
- **6 features:**
  1. Left elbow angle
  2. Right elbow angle
  3. Hand-shoulder ratio
  4. Torso tilted angle
  5. Hand tilted angle
  6. Elbow angles difference

**Điểm cộng nếu:**
- Giải thích được từng feature có ý nghĩa gì (vd: elbow angle = angle at vertex elbow)

---

### Q4: Tại sao cần StandardScaler trong pipeline ML?
**Trả lời mong đợi:**
Để normalize các features về cùng scale:
- Biến đổi: `(feature - mean) / std` → mean = 0, std = 1
- Mục đích: Ngăn chặn high-variance features (vd: angle 0-180°) cản rãi low-variance features (vd: ratio 0-2)
- LSTM nhạy cảm với scale → chưa normalize → model learning khó khăn hoặc output collapse

**Điểm cộng nếu:**
- Nếu đề cập đến "stuck at 53" problem (model bị stuck ở điểm cố định vì features không normalize)

---

### Q5: File `.joblib` chứa gì?
**Trả lời mong đợi:**
StandardScaler object lưu:
- `mean_` - trung bình của training features (một vector per exercise)
- `scale_` - độ lệch chuẩn của training features
- `n_features_in_` - số features

**KHÔNG phải:**
- ❌ Model weights
- ❌ Mean của model predict
- ❌ Output của neural network

**Ví dụ Es1:**
```
scaler_Es1.mean_ = [45.5°, 49.0°, 1.230, 12.3°, 5.1°, 3.8°]
scaler_Es1.scale_ = [3.2°, 4.1°, 0.12, 2.0°, 0.8°, 0.45°]
```

---

### Q6: Pipeline inference có mấy bước?
**Trả lời mong đợi:**
6 bước chính:
1. **Parse CSV** - Frontend gửi 30 frames raw keypoints
2. **Feature extraction** - Tính angles, distances, ratios từ keypoints
3. **Temporal downsampling** - Stride=5: 30 frames → 6 frames (~6 FPS)
4. **StandardScaler normalization** - Dùng scaler_Ex.joblib
5. **Padding** - Pad đến 150 frames với sentinel -999.0
6. **LSTM inference** - Dự đoán clinical score [0-100]

**Điểm cộng nếu:**
- Giải thích TẠI SAO phải pad tới 150:
  - Model input shape fixed: `(batch, 150, 6)` → được định nghĩa khi training
  - Training data: 750 frames (30s @ 25fps) → downsample stride=5 → 150 frames
  - Inference phải match training shape
  - Masking layer tự động ignore -999 values khi compute
- Nếu họ biết là 150 = CRITICAL invariant (không được đổi)

---

### Q7: Tại sao cần downsampling?
**Trả lời mong đợi:**
- Raw video 25fps có ~675 frames, quá dài cho LSTM với 72 training samples
- Stride=5 → 675 frames → ~135 frames (manageable)
- Frame liên tiếp gần như giống nhau, downsample không mất info
- **CRITICAL**: Phải apply BEFORE StandardScaler để match training pipeline

---

### Q8: Motion detection dùng để làm gì?
**Trả lời mong đợi:**
Xác định bệnh nhân có thực sự làm bài tập hay đang đứng yên. Nếu không có motion:
- Không reset score hoàn toàn (tránh 0 nếu gesture nhỏ)
- Áp dụng quality_factor penalty (nhân score với hệ số < 1.0)

**3 metrics:**
1. Temporal variance - features thay đổi bao nhiêu
2. Range of motion - max - min per feature
3. Frame displacement - avg |frame[i+1] - frame[i]|

---

### Q8.5: Tại sao model input shape PHẢI là (1, 150, 6) - không thể thay đổi?
**Trả lời mong đợi:**

**LSTM input shape được fix khi training:**
```python
# Training
X_train shape: (n_samples, 150, 6)
model = Sequential([Masking(mask_value=-999.0), LSTM(...)])
model.fit(X_train, y_train)
# Model graph LOCK input shape: (None, 150, 6)
```

**Không thể thay đổi khi inference:**
```python
# Inference
X_infer shape phải = (1, 150, 6)  # MUST match!
model.predict(X_infer)  # ✓ Works

# Nếu pad tới 200:
X_infer_wrong shape = (1, 200, 6)
model.predict(X_infer_wrong)  # ❌ Shape mismatch error!
```

**Tại sao 150 được chọn:**
- Full training exercise: 30s @ 25fps = 750 frames
- Downsample stride=5: 750 → 150 frames
- 150 là compromise:
  - Đủ dài để capture motion sequence (không mất info)
  - Đủ ngắn để train được với 72 samples (avoid overfitting)

**Hệ quả:**
- Nếu inference batch có < 150 frames → pad với -999.0
- Nếu > 150 frames → truncate (discard tail)
- Masking layer tự động ignore -999 values → không ảnh hưởng

**Chi tiết Training Process (Cách Tạo Shape 150):**

```python
# Step 1: Load raw features (variable length)
raw_features, y = load_raw_features(csv_path, downsample_stride=5)
# Original: min=402, max=891, mean=672 frames
# After downsample stride=5: min=80, max=178, mean=134 frames
# → VẬN KHÁC NHAU!

# Step 2: 5-fold CV - Fit StandardScaler per-fold
for fold, (train_idx, val_idx) in kfold.split(y):
    raw_train = [raw_features[i] for i in train_idx]
    scaler = StandardScaler()
    scaler.fit(np.vstack(raw_train))  # Combine all train frames
    
# Step 3: Scale + Pad to FIXED (N, 150, 6)
X_train = scale_and_pad(raw_train, scaler, max_length=150)
# Pre-allocate: np.full((N, 150, 6), MASK_VALUE=-999.0)
# For each video i:
#   - Scale video i
#   - If len(video_i) < 150: copy data, rest = -999.0
#   - If len(video_i) > 150: truncate, keep only first 150
# Result: X_train shape = (40, 150, 6) ✓

# Step 4: Build model với fixed input shape
model = Sequential([
    Input(shape=(150, 6)),  # ← EXACTLY 150 hardcoded
    Masking(mask_value=-999.0),  # Skip -999 frames
    LSTM(32, return_sequences=True),
    Dropout(0.3),
    LSTM(16),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 5: Train
model.fit(X_train, y_train, epochs=200)
# Model graph LOCKED: input = (batch, 150, 6)
```

**Điểm cộng nếu:**
- Giải thích được training tạo shape như thế nào
- Biết Masking layer skip -999 values
- Hiểu tại sao không thể đổi 150 thành 200 hoặc 100 (input mismatch)

---

## 🟡 Trung Bình (Intermediate)

### Q9: Giải thích điều gì xảy ra khi frontend gửi pose keypoints tới backend qua WebSocket.
**Trả lời mong đợi:**

```
Timeline:
0-5000ms: User thực hiện bài tập
   ├─ MoveNet detect 30 FPS
   └─ framesRef.current accumulate frames

5000ms: ⭐ Batch 1 (30 frames)
   ├─ Frontend gửi: {"type": "clinical_score", "csvString": "..."}
   └─ Backend nhận:
      ├─ Parse CSV → DataFrame(30, 51)
      ├─ Extract features → (30, 6)
      ├─ Downsample stride=5 → (6, 6)
      ├─ StandardScaler → (6, 6) normalized
      ├─ Pad → (1, 150, 6)
      ├─ LSTM predict → score
      └─ detect_motion() → quality_factor
   └─ Backend gửi back: {"type": "clinical_score", "score": 72.5, ...}

10000ms: ⭐ Batch 2 (30 frames mới)
   └─ Lặp lại...
```

**Chi tiết quan trọng:**
- CSV có 51 cột (17 keypoints × 3: x, y, confidence)
- 30 hàng (1 giây @ 30 FPS)
- Backend chỉ lấy 12 body keypoints (drop face/ears)
- Downsampling → chỉ 6 frames đi vào LSTM

---

### Q10: Tại sao backend cần sử dụng CÙNG StandardScaler (từ .joblib) khi inference, không thể retrain/refit được?
**Trả lời mong đợi:**

**Lý do:**
StandardScaler phải match training distribution để model hoạt động đúng.

**Ví dụ:**
```
Training:
  Công thức: (X_train - mean_train) / std_train
  mean_train = 45.5°, std_train = 3.2°
  Model learn trên data normalized này
  
Inference - ✅ ĐÚNG:
  scaler = load('scaler_Es1.joblib')  # mean=45.5, std=3.2
  X_infer = (45.8° - 45.5) / 3.2 = 0.094
  model.predict(0.094) → tốt, in-distribution
  
Inference - ❌ SAI (refit):
  scaler_NEW = StandardScaler()
  scaler_NEW.fit(X_infer_only)  # mean_new=65°, std_new=5.0°
  X_infer = (45.8° - 65°) / 5.0° = -3.84
  model.predict(-3.84) → Out-of-distribution → Bad prediction!
```

**Kết luận:** Refitting scaler sẽ khiến model nhận data ngoài distribution training → sai kết quả.

---

### Q11: Các exercise Es1 đến Es5 có thể dùng chung một StandardScaler không? Tại sao?
**Trả lời mong đợi:**

**Câu trả lời: KHÔNG**

**Lý do:**
1. **Khác features**: Es1 có 6 features, Es3 có 9, Es4 chỉ có 2
2. **Khác distribution**: 
   ```
   Es1 (arm lift):
     torso_tilted_angle range: 0-30°, mean: 8°, std: 5°
   
   Es5 (squatting):
     torso_tilted_angle range: 0-70°, mean: 45°, std: 15°
   ```
3. **Khác phần cơ thể:** Es1 focus trên arms, Es5 focus trên legs

**Hậu quả nếu dùng chung:**
- Es1 scaler + Es5 data → features giá trị lớn gấp 10 lần training distribution
- Model: "This is weird, out of range" → Sai prediction

---

### Q12: Giải thích aspect ratio correction (x * 1.7778) trong feature extraction.
**Trả lời mong đợi:**

**Vấn đề:**
- MediaPipe normalize x, y ∈ [0, 1] giả sử square video
- Thực tế video 16:9 (1920×1080)
- Nếu không fix: x-axis bị stretch → angles tính toán bị distort

**Giải pháp:**
- Nhân x với 1920/1080 = 1.7778
- Làm không gian isotropic (vuông đúng) → angles tính chính xác

**Ví dụ:**
```
Raw MediaPipe: (0.5, 0.5) - tâm video
✓ Aspect-corrected: (0.5×1.7778, 0.5) = (0.889, 0.5)
  → Ngay tâm khi tính angle

✗ Không fix: (0.5, 0.5) 
  → Bị lệch, angle sai vì x bị stretch
```

---

### Q13: Downsampling áp dụng ở đâu trong pipeline - TRƯỚC hay SAU khi scale?
**Trả lời mong đợi:**

**Câu trả lời: TRƯỚC (downsampling → StandardScaler)**

**Chứng minh:**
```python
# Code từ ml_wrapper.py
features = _safe_extract_features(df, exercise_id)
features = temporal_downsample(features, DOWNSAMPLE_STRIDE, max_length)  # TRƯỚC
motion_result = detect_motion(features, exercise_id)  # Motion check on raw
features_scaled = scaler.transform(features)  # SAU
```

**Tại sao điều này quan trọng:**
- Training script (03_extract_joint_features.py) cũng làm theo cách này
- Nếu đảo ngược: StandardScaler sẽ fit trên 675 frames, inference chỉ có 6 frames → mismatch
- Motion detection thresholds (min_variance=5.0°) có ý nghĩa trên raw features, không trên normalized

---

### Q14: Có bao nhiêu WebSocket endpoints và chúng để làm gì?
**Trả lời mong đợi:**

**1 endpoint chính:** `/ws/session/{exercise_id}`

**2 message types:**
1. **"feedback"** - Real-time feedback (mỗi ~30 frames)
   - Input: `referenceJointValues`, `currentJointValues`
   - Output: Angle violations, Vietnamese messages
   
2. **"clinical_score"** - Batch scoring (mỗi 5 giây)
   - Input: `csvString` (CSV của accumulated frames)
   - Output: Clinical score [0-100], motion_detected, motion_energy, quality_factor

**Lợi thế WebSocket:**
- Low latency (không HTTP overhead)
- Bidirectional (server push feedback)
- Persistent connection (không reconnect mỗi frame)

---

### Q15: Giải thích "stuck at 53" problem và cách detect/fix nó.
**Trả lời mong đợi:**

**"Stuck at 53" = Model output collapse:**
- Tất cả input → output ~53 (hoặc ~50), không phụ thuộc input
- Nguyên nhân root causes:
  1. StandardScaler missing → features unscaled → model OOD
  2. Model checkpoint sai → random weights
  3. Padding sentinel = 0 thay -999 → model learn treat padding as "normal"
  4. Feature extraction return all-zeros/NaNs

**Detection** (diagnostic_check.py):
```python
# Test with diverse inputs
test_scores = [model.predict(input1), model.predict(input2), ...]
score_range = max(test_scores) - min(test_scores)
if score_range < 1.0:
    print("⚠ Output collapse detected!")
```

**Fix:**
- Kiểm tra scaler file có tồn tại không
- Kiểm tra model có load được không
- Verify features không phải all-zero
- Ensure padding value = -999.0 (match Masking layer)

---

## 🔴 Nâng Cao (Advanced)

### Q16: Giải thích data flow từ training (clinical_score_prediction_model.py) sang inference (ml_wrapper.py) - điểm khác nhau và invariants.
**Trả lời mong đợi:**

| Khía cạnh | Training | Inference | Invariant? |
|----------|----------|-----------|-----------|
| **Source** | CSV batch files | WebSocket JSON | ❌ Khác |
| **Feature extraction** | 03_extract_joint_features.py | joint_features.py | ✅ Phải identical |
| **Downsampling** | Stride-5 BEFORE scaler fit | Stride-5 BEFORE scaler transform | ✅ CRITICAL |
| **StandardScaler** | Fit on training, save .joblib | Load .joblib, transform only | ✅ CRITICAL |
| **Padding value** | -999.0 (match Masking layer) | -999.0 | ✅ MUST match |
| **Output format** | Raw model [0,1] or [0,50] | Auto-detect + rescale 0-100 | ❌ OK khác |
| **Motion detection** | Post-hoc label | Applied to raw features | ❌ Khác |

**CRITICAL INVARIANTS (phải identical):**
1. Feature extraction code
2. Downsampling trước StandardScaler
3. StandardScaler mean/std
4. Padding sentinel = -999.0

**Hậu quả vi phạm:**
```
Nếu training dùng stride=5, inference dùng stride=3:
  Training: 675 frames → 135 frames
  Inference: 675 frames → 225 frames
  
  Scaler fit trên 135 frames, transform 225 frames
  → Shape mismatch → Error hoặc incorrect normalization
```

---

### Q17: Vì sao motion detection được chạy trên raw features (trước scaling) thay vì scaled features?
**Trả lời mong đợi:**

**Lý do:**
1. **Thresholds có ý nghĩa clinically:**
   ```
   Raw features:
     min_variance = 5.0°  → "Min angle variance 5 degrees"
     min_rom = 8.0°       → "Range of motion at least 8 degrees"
   
   Scaled features:
     min_variance = 0.2σ  → "Min angle variance 0.2 std deviations"
                            (abstract, không clinically meaningful)
   ```

2. **Đảm bảo consistency với training:**
   - Training pipeline cũng tính motion metrics trên raw features
   - Inference phải match

3. **Prevents circular dependency:**
   - Scaling phụ thuộc vào StandardScaler
   - Motion detection độc lập → robust ngay cả nếu scaler lỗi

**Code:**
```python
features = temporal_downsample(features, DOWNSAMPLE_STRIDE, max_length)
motion_result = detect_motion(features, exercise_id)  # ← RAW
features_scaled = scaler.transform(features)
```

---

### Q18: Vì sao lại không áp dụng padding (với -999.0) trước lúc gữa dữ liệu tới model?
**Trả lời mong đợi:**

**Giải thích padding strategy:**

```
Scenario: Có 6 frames sau downsampling, model expect 150 frames

Option 1 - Pad TRƯỚC scaling:
  [f0, f1, f2, f3, f4, f5, -999, -999, ...] (150 frames)
  ↓
  StandardScaler.transform()  → Lỗi! -999 ngoài training range
  → Scaled value rất lớn (-∞ ~ ∞)

Option 2 - Pad SAU scaling (✓ ĐÚNG):
  [f0_scaled, f1_scaled, f2_scaled, f3_scaled, f4_scaled, f5_scaled]
  ↓
  Pad: [f0_scaled, ..., f5_scaled, -999, -999, ...] (150 frames)
  ↓
  LSTM với Masking(mask_value=-999.0) → ignore masked frames
```

**Lý do Masking layer:**
```python
model = Sequential([
    Masking(mask_value=-999.0),  # ← Ignore frames with value -999
    LSTM(32, ...),
    ...
])
```

**Kết luận:**
- Padding phải SAU scaling để -999 không bị transform
- LSTM Masking layer tự động ignore masked frames khi compute loss/gradients

---

### Q19: Nếu frontend gửi 30 frames ES NHƯNG backend downsampling stride=5 → 6 frames. Vì sao không phải 30 frames?
**Trả lời mong đợi:**

**Hiểu sai phổ biến:**
- Người tinh tế khác: có 30 frames/batch, downsampling stride=5 → 6 frames
- Nhưng NNVS (neural network view): 6 frames quá ngắn, model expect 150 frames

**Giải thích:**
```
Design decision: Stride=5 downsampling

Lý do chọn stride=5:
- Training data: full exercise ~25-30 giây @ 25fps = ~750 frames
- Quá dài cho LSTM (gradient exploding/vanishing)
- Stride=5: 750 → 150 frames (manageable)

Inference data: 30 frames @ 30fps
- Chỉ 1 giây từ user
- Stride=5: 30 → 6 frames
- Không đủ 150 → Pad với -999 → 150 frames

Question: Vì sao không dùng stride=3 hay stride=2 khi inference?
Trả lời: Phải match training! Training xài stride=5 everywhere.
  Nếu inference dùng stride=3:
    30 frames → 10 frames
    Pad: 10 + 140 masked frames
    Scaler fit trên 150 real frames, transform 10 real + 140 masked
    → Inconsistent → Bad results
```

---

### Q20: Tại sao việc dùng MoveNet (frontend) + MediaPipe (backend) lại tạo ra vấn đề? Cách giải quyết?
**Trả lời mong đợi:**

**Vấn đề:**
```
MoveNet output:           MediaPipe format:
17 keypoints              33 keypoints (landmarks từ cả body, hand, face)

Frontend gửi index:       Backend mong đợi:
  left_shoulder: 5          left_shoulder: 11
  left_elbow: 7             left_elbow: 13
  left_wrist: 9             left_wrist: 15
  left_hip: 11              left_hip: 23
  
Nếu không mapping:
  Feature tính từ indices sai
  → Angle = atan2(shoulder_wrong - elbow_wrong - wrist_wrong)
  → Tính nhầm → Inference sai
```

**Cách giải quyết hiện tại:**
```python
# Exercise.jsx
KEYPOINT_DICT = {
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    ...
}

frameData = {}
for (const [jointName, jointIndex] of Object.entries(KEYPOINT_DICT)) {
    const keypoint = normalizedKeypoints[jointIndex];
    frameData[jointName + '_x'] = keypoint.x;
    frameData[jointName + '_y'] = keypoint.y;
}
// Gửi frameData via CSV → Backend nhận với tên đúng (left_shoulder, v.v)
```

**Cách này tránh được:**
- Frontend gửi CSV với header là tên joint (left_shoulder, right_shoulder, ...)
- Backend không cần biết index → chỉ cần tên là đúng
- Backend tìm cột 'left_shoulder_x', 'left_shoulder_y' rồi tính feature

**Risk vẫn tồn tại:**
- Nếu frontend gửi sai index (vd: index 6 thay vì 5), CSV sẽ có giá trị sai
- Backend không thể detect → sai diagnosis

**Cách fix tốt hơn:**
```
Option 1: Frontend → MediaPipe (thay MoveNet)
  - Require WASM/Python backend cho browser
  - Slow hơn, complexity tăng
  
Option 2: Backend xác nhân keypoint order
  - Add checksum/hash của 3-4 joints vào CSV
  - Backend verify checksum trước tính features
  - Fail fast nếu sai
```

---

## 🏗️ Thiết Kế Kiến Trúc (Architecture Design)

### Q21: Tại sao lại chọn LSTM thay vì CNN, Transformer, hoặc regression model khác?
**Trả lời mong đợi:**

**Context:**
- Dataset: ~72 samples per exercise (rất nhỏ)
- Input: Sequence temporal (25-50 giây)
- Output: Single value (clinical score)

**Lý do LSTM:**
1. **Temporal dependency:** Exercise là sequence action, frame liên tiếp liên quan → LSTM track dependencies
2. **Small dataset:** 
   - CNN cần nhiều data hơn (lớp convolution nhiều parameters)
   - Transformer chậm hội tụ với 72 samples
   - LSTM: fewer parameters, tốt cho small data
3. **Variable-length sequences:** LSTM + Masking xử lý tốt padded sequences
4. **Interpretable:** LSTM states có ý nghĩa, không black box như Transformer

**LSTM vs alternatives:**
- **GRU**: Tương tự LSTM, đơn giản hơn, ít parameters → cũng ổn, nhưng LSTM safer
- **Attention/Transformer**: Cần 10x data hơn
- **Linear regression**: Ignores temporal structure, poor performance
- **Random Forest**: Không handle sequences tốt
- **SVM**: Cũng không temporal

**Kết luận:** LSTM is sweet spot cho small temporal dataset.

---

### Q22: Vì sao clinical_score_prediction_model.py fit scaler trên toàn bộ training data (không K-fold split)?
**Trả lời mong đợi:**

**Thiết kế quyết định:**
```
Pass 1: Extract features từ tất cả videos
Pass 2: Fit StandardScaler trên tất cả features
Pass 3: Train model với K-fold CV
```

**Lý do:**
1. **StandardScaler chỉ là preprocessing:**
   - Không "learn" relationships như model
   - Fitting trên toàn bộ data → compute mean/std của toàn bộ distribution
   - Là statistics, không learned parameters

2. **Để match inference:**
   - Inference cũng xài scaler fit trên toàn bộ training data
   - Nếu training xài scaler từ fold này, inference xài scaler từ fold kia → mismatch
   - Must be consistent

3. **K-fold CV chỉ cho model:**
   ```
   Pass 2: scaler = fit(all_data) → save scaler_Es1.joblib
   Pass 3: for each fold:
     train_data, test_data = split()
     model.fit(scaler.transform(train_data))
     eval(model(scaler.transform(test_data)))
   ```

**Analogy:**
- Scaler = "ruler" để measure feature scale (dùng ruler toàn bộ, không per-fold)
- Model = "learner" (học per-fold để avoid data leakage)

---

### Q23: Vì sao database chọn SQLite thay vì PostgreSQL hoặc MongoDB?
**Trả lời mong đợi:**

**SQLite pros:**
- ✅ Lightweight, không server → Docker dễ, development nhanh
- ✅ Embedded → local .db file, không cần database service
- ✅ ACID compliant → data integrity
- ✅ SQL standard → easy query

**SQLite cons:**
- ❌ Single-writer limitation → concurrency issue khi many users submit score cùng lúc
- ❌ No row-level security → production multi-tenant unsafe
- ❌ Scaling limited → data size > 10GB ↑ performance issue
- ❌ No built-in replication → backup phức tạp

**Lựa chọn khác:**
- **PostgreSQL:** Production-grade, multi-user, scale tốt, nhưng overkill cho MVP
- **MongoDB:** No-schema flexibility, nhưng RehabAI schema fixed (Users, Exercises, ProgressTracker)

**Kết luận:**
SQLite tốt cho prototype/MVP. Scaling sang PostgreSQL later.

---

### Q24: Architecture có vấn đề gì về latency? Bất kỳ bottleneck nào?
**Trả lời mong đợi:**

**Latency Analysis:**

| Component | Latency | Bottleneck? | Root cause |
|-----------|---------|------------|-----------|
| MoveNet inference (30 FPS) | ~33ms/frame | No | TensorFlow.js optimized |
| WebSocket roundtrip | ~50ms | Maybe | Network latency |
| CSV parsing (30 frames) | ~5ms | No | Simple split() |
| Feature extraction (6-9 rules) | ~30-80ms | **YES** | 2D angle math, multiple loops |
| Motion detection | ~10-20ms | No | 3 metrics, minimal calc |
| StandardScaler transform | ~5-10ms | No | NumPy optimized |
| **LSTM inference** | **100-300ms** | **CRITICAL** | GPU-dependent, model size |
| Total per batch | **200-500ms** | | |

**Bottleneck giải quyết:**
```python
# ml_wrapper.py uses run_in_threadpool
score_result = await run_in_threadpool(
    _predict_clinical_score_from_csv,
    csv_string,
    ...
)
```
→ Move CPU-bound ops to thread pool, không block event loop

**Potential improvements:**
1. GPU for LSTM → 50-100ms (vs 300ms CPU)
2. Batch multiple inference requests
3. Cache extracted features (nếu user retry)
4. Move feature extraction to frontend (TensorFlow.js)

---

### Q25: Tại sao không dùng WebSocket cho TẤT CẢ communication (thay vì REST endpoints)?
**Trả lời mong đợi:**

**Current design:**
- REST: `/api/login`, `/api/signup`, `/api/exercises` (stateless, simple)
- WebSocket: `/ws/session/{exercise_id}` (stateful, real-time)

**Vì sao không all-WebSocket:**

| Aspekt | REST | WebSocket |
|--------|------|-----------|
| **Auth** | Clear (Bearer token header) | Tricky (auth per connection) |
| **Caching** | Easy (HTTP cache headers) | Complex (in-memory cache) |
| **Scaling** | Stateless → easy horizontal | Stateful → sticky session needed |
| **Connection loss** | Automatic retry (HTTP) | Manual reconnect logic |
| **Latency** | Higher overhead | Lower overhead |

**Current tradeoff:**
- Auth/user management → REST (stateless, cache-friendly)
- Real-time feedback → WebSocket (low latency)

**Why all-WebSocket bad:**
```
Example: 1000 concurrent users
  - All WebSocket → 1000 persistent connections per server
  - Load balancer cần sticky session → routing complex
  - One server down → 1000 users disconnected
  
  - Hybrid (most REST) → load balancer stateless
  - WebSocket users scale independently
```

---

## 🐛 Troubleshooting & Debugging

### Q26: Model output stuck at 53. Làm sao debug?
**Trả lời mong đợi:**

**Step-by-step:**

```python
# Step 1: Check scaler file
import os
if not os.path.exists('models/scaler_Es1.joblib'):
    print("❌ Scaler missing!")
else:
    print("✅ Scaler found")

# Step 2: Verify scaler loading
import joblib
scaler = joblib.load('models/scaler_Es1.joblib')
print(f"Scaler mean: {scaler.mean_}")
print(f"Scaler scale: {scaler.scale_}")

# Step 3: Check model loading
import tensorflow as tf
model = tf.keras.models.load_model('models/ml_model_Es1.keras')
print(f"Model input shape: {model.input_shape}")

# Step 4: Test inference with diverse inputs
import numpy as np
test_inputs = [
    np.zeros((1, 150, 6)),     # all zeros
    np.ones((1, 150, 6)),      # all ones
    np.random.randn(1, 150, 6) # random
]
scores = [model.predict(inp).item() for inp in test_inputs]
print(f"Scores: {scores}")
score_range = max(scores) - min(scores)
if score_range < 1.0:
    print("❌ Output collapse detected!")
else:
    print("✅ Model outputs vary")

# Step 5: Check feature extraction
from joint_features import get_es1_features
test_df = pd.DataFrame({
    'left_shoulder_x': [0.3] * 30,
    'left_shoulder_y': [0.4] * 30,
    ...  # all 12 joints
})
features = get_es1_features(test_df)
print(f"Features shape: {features.shape}")
print(f"Features contain NaN: {features.isna().any().any()}")
print(f"Features contain Inf: {np.isinf(features).any().any()}")

# Step 6: Check padding
if (features.shape[0] < 150):
    features_padded = np.pad(features, ...)
    if np.isnan(features_padded).any():
        print("❌ Padding introduced NaN!")
```

**Common causes & fixes:**
1. **Missing scaler** → Regenerate: `python 03_extract_joint_features.py`
2. **Model checkpoint broken** → Retrain: `python clinical_score_prediction_model.py`
3. **Padding value = 0.0** → Fix: change constant_values=MASK_VALUE (-999.0)
4. **Feature extraction NaN** → Check input keypoints not all zeros

---

### Q27: Inference produce score 5, kỳ vọng 70. Có thể là vì sao?
**Trả lời mong đợi:**

**Diagnostic:**

```python
# Hypothesis 1: Motion not detected
motion_result = detect_motion(features, 'Es1')
if not motion_result.is_active:
    print(f"Motion not detected! motion_energy={motion_result.motion_energy}")
    print(f"Quality factor: {motion_result.quality_factor}")
    # Fix: User didn't move enough, score penalized by quality_factor
    
# Hypothesis 2: Distribution shift (user new, style different)
raw_score = model.predict(normalized_data)
if raw_score > 1.5:
    score_100 = float(np.clip(raw_score, 0, 50) * 2.0)  # 0-100
else:
    score_100 = raw_score * 50  # if already normalized
# Check if scaling correct
print(f"Raw score: {raw_score}, After scaling: {score_100}")

# Hypothesis 3: Feature extraction issue
print(f"Feature ranges: {features.min(axis=0)} to {features.max(axis=0)}")
print(f"Expected ranges (from training): [check paper/training logs]")
if feature range way off:
    print("❌ Features out of training distribution!")

# Hypothesis 4: StandardScaler applied multiple times
features_scaled_1x = scaler.transform(features)
features_scaled_2x = scaler.transform(features_scaled_1x)  # ❌ Wrong!
print(f"1x scaling: {features_scaled_1x[0]}")
print(f"2x scaling: {features_scaled_2x[0]}")  # Much smaller
```

**Most likely:** Motion not detected → quality_factor penalty.
```
raw_score = 70
motion_energy = 0.2 (low)
quality_factor = 0.1 (penalized)
final_score = 70 * 0.1 = 7
```

---

### Q28: Frontend gửi CSV, backend nhận nhưng feature extraction return NaN. Debug?
**Trả lời mong đợi:**

**Causes:**

```python
# Cause 1: Input keypoints all zero
if np.all(df == 0):
    print("❌ All input keypoints are zero!")
    # Check MoveNet detection failure
    # Check frontend not sending correct data

# Cause 2: Missing columns
required_cols = ['left_shoulder_x', 'left_shoulder_y', ...]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    # Frontend CSV header mismatch

# Cause 3: Feature extraction bug
from joint_features import get_es1_features
features = get_es1_features(df)
nan_count = np.isnan(features).sum()
if nan_count > 0:
    # Identify which features have NaN
    for i, col in enumerate(features.columns):
        nan_in_col = np.isnan(features.iloc[:, i]).sum()
        if nan_in_col > 0:
            print(f"NaN in {col}: {nan_in_col} frames")
    
    # Example: NaN in angle calculation
    # → One of 3 points (first, middle, end) might be all zeros
    df[['left_shoulder_x', 'left_shoulder_y']].describe()

# Cause 4: Padding before feature extraction
# ❌ WRONG: 
#   df_padded = np.pad(df, ...)  # Padding with -999
#   features = get_es1_features(df_padded)
#   → Feature calc use -999 → NaN
# 
# ✅ RIGHT:
#   features = get_es1_features(df)  # Features first
#   features_padded = np.pad(features, ...)  # Then pad
```

**Fix:**
```python
# Defensive programming
df_clean = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
features = get_es1_features(df_clean)
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
```

---

### Q29: A-B testing: khi nào model output không đáng tin?
**Trả lời mong đợi:**

**Red flags:**

1. **motion_detected = False**
   - User không cử động
   - Score sẽ bị penalty
   - Fix: User phải làm bài tập rõ rệt hơn

2. **motion_energy < 0.3**
   - Chỉ làm nhỏ/chậm
   - quality_factor penalty apply
   - Warning: "Bạn vui lòng làm bài tập rõ rệt hơn"

3. **Feature values way out of training range**
   - Vd: elbow_angle > 180° (impossible)
   - Indication: User positioning wrong hoặc MoveNet fail
   - Debug: Check keypoint visualization

4. **Latency > 1000ms**
   - Model/server overload
   - Inference bị queue
   - Warning: "Server bận, vui lòng thử lại"

5. **Score jump từ 20 → 80 trong 1 frame**
   - Indication: Motion detection false positive hoặc extreme keypoint jump
   - Debug: Check frame-to-frame consistency

**Best practice:**
```python
if not motion_result.is_active:
    print("⚠️  Warning: Low motion detected")
    
if motion_result.quality_factor < 0.3:
    print("⚠️  Warning: Poor exercise quality")
    
if inference_latency > 500:
    print("⚠️  Warning: Server slow")
    
if score_jumped > 30 since last frame:
    print("⚠️  Warning: Anomaly detected, verify data")
```

---

### Q30: So sánh khi user làm bài tập ES1 vs ES4 với cùng chất lượng thực hiện, tại sao score khác nhau?
**Trả lời mong đợi:**

**Vì mô hình khác nhau:**

```
Es1 (Arm Lifts) - 6 features:
  ├─ left_elbow_angle
  ├─ right_elbow_angle
  ├─ hand_shoulder_ratio
  ├─ torso_tilted_angle
  ├─ hand_tilted_angle
  └─ elbow_angles_diff

Es4 (Pelvis Rotation) - 2 features:
  ├─ torso_tilted_angle
  └─ knee_hip_ratio
```

**Vì sao score khác:**

| Reason | Impact |
|--------|--------|
| **Model khác** | Es1 LSTM trained on 50 Es1 videos, Es4 LSTM trained on 50 Es4 videos |
| **Features khác** | Es1 focus arms (6 features), Es4 focus legs (2 features) |
| **Training distribution khác** | Es1 users: arm angle range 40-120°; Es4 users: hip angle range 20-100° |
| **StandardScaler khác** | scaler_Es1.mean = [45.5, 49.0, ...], scaler_Es4.mean = [60.0, ...] |
| **Grading scale khác** | Es1 model trained to differentiate Es1 quality; Es4 model trained for Es4 quality |

**Ví dụ:**
```
User A (excellent Es1 form):
  Score: 85/100 (arms high, smooth, symmetric)

User A cố gắng làm Es4 (nhưng không trained):
  Attempt to do pelvic rotation, nhưng não clear pelvis motion
  Es4 model doesn't have enough arm-feature to score well
  Score: 30/100 (mô hình khác, user không trained bài này)
```

**Moral:** Score không absolute "quality", mà "quality relative to model's training data"

---

## 📝 Meta-Questions for Deep Understanding

### Q31: Nếu có thêm 1000 training samples thay vì 72 hiện tại, bạn sẽ thay đổi architecture như thế nào?
**Expected to discuss:**
- LSTM có thể giữ, nhưng Transformer/CNN trở thành viable
- Feature engineering có thể reduce (model tự học features)
- Dùng transfer learning từ pretrained models
- Multi-task learning (predict score + predict exercise type simultaneously)
- Data augmentation strategies

### Q32: Designing a new exercise Es6 (new type). Step-by-step?
**Expected to discuss:**
1. Collect ~50 videos từ experts
2. Extract MediaPipe landmarks (01_extract_joint_positions.py)
3. Label với clinical scores (02_prepare_dataset.py)
4. Design features (select từ 12 available từ paper)
5. Fit StandardScaler (03_extract_joint_features.py → scaler_Es6.joblib)
6. Train LSTM (clinical_score_prediction_model.py)
7. Add feedback rules (feedback_engine.py) cho Es6
8. Test & validate (diagnostic_check.py)

### Q33: Hệ thống có thể handle multi-user realtime sessions không? Scale bao xa?
**Expected to discuss:**
- Current: SQLite (single writer) → bottleneck at concurrent writes
- WebSocket per user → server memory usage O(n_users)
- Feature extraction/LSTM per user → CPU bottleneck (threadpool help)
- Scale: 10-100 concurrent users on modern server
- Scale to 1000+: Need PostgreSQL + Redis + Kubernetes

---

## 📚 Rubric Scoring Guidelines

| Score | Criteria |
|-------|----------|
| **1/5** | Biết dự án về phục hồi chức năng, biết có pose detection |
| **2/5** | Hiểu cơ bản data flow, features, downsampling |
| **3/5** | Giải thích StandardScaler, downsampling order, motion detection role |
| **4/5** | Biết training vs inference invariants, detect issues (output collapse, Q29) |
| **5/5** | Có thể design new exercise, defend architecture decisions, suggest improvements |

---

**Last Updated:** May 2026  
**Total Questions:** 33 (10 Beginner + 10 Intermediate + 5 Advanced + 5 Architecture + 3 Debugging)
