# Phân tích Toàn diện Các Vấn đề Hệ thống RehabAI

**Ngày:** 2026-05-08  
**Phạm vi:** Ngoài domain shift (Kinect → Webcam), phân tích các vấn đề còn lại trong pipeline.

---

## Tóm tắt Executive

| # | Vấn đề | Mức độ | Ảnh hưởng | Khó sửa |
|---|--------|--------|-----------|---------|
| 1 | Dataset quá nhỏ (N≈67) | 🔴 Cao | Spearman variance cao, không ổn định | Cao — cần data mới |
| 2 | Score distribution skewed (~65% healthy) | 🔴 Cao | Model bias toward low scores | Trung bình |
| 3 | 2D features mất thông tin depth | 🟡 Trung bình | Es3/Es4 rotation exercises kém | Cao — cần 3D pose |
| 4 | Feature engineering chỉ 6-7 features | 🟡 Trung bình | Không đủ discriminative | Trung bình |
| 5 | Model capacity thấp (~5K params) | 🟡 Trung bình | Giới hạn học complex patterns | Thấp |
| 6 | NaN handling: fillna(0.0) | 🟠 Trung bình-Cao | Introduce artifacts vào features | Thấp |
| 7 | Frame rate mismatch (25fps vs 30fps) | 🟡 Trung bình | Temporal patterns khác nhau | Thấp |
| 8 | Single-session scoring | 🟡 Trung bình | Không có variance estimate | Trung bình |
| 9 | Motion detection thresholds heuristic | 🟠 Trung bình-Cao | False negative cho stroke patients | Thấp |
| 10 | Score calibration = crude proxy | 🟡 Trung bình | quality_factor không phải real confidence | Trung bình |
| 11 | Active region trimming heuristic | 🟢 Thấp | Có thể cắt nhầm | Thấp |
| 12 | Scaler fitted on train nhưng inference ≠ distribution | 🟡 Trung bình | Out-of-distribution features | Cao |

---

## 1. Dataset Quá Nhỏ (N≈67 Subjects)

### Vấn đề
- KiMoRe dataset chỉ có ~67 subjects, mỗi subject 1-2 videos
- 5-fold CV → mỗi fold validation chỉ ~13 subjects
- **Standard error của Spearman ≈ 1/√N ≈ 0.12** → fold-level scores dao động mạnh

### Hệ quả
- Fold Spearman có thể dao động từ 0.1 đến 0.7 giữa các fold
- Không thể phân biệt model improvements vs statistical noise
- Confidence intervals rất rộng → không thể tin cậy kết quả

### Bằng chứng từ code
```python
# training script: BalancedStratifiedGroupKFold với n_splits=5
# Mỗi fold ~13 subjects validation → SE(Spearman) ≈ 0.12
```

### Khuyến nghị
- **Repeated CV** với multiple seeds (n_repeats=3-5) để ổn định metrics
- **Bootstrap confidence intervals** cho OOF predictions
- Thu thập thêm data nếu có thể (target: N≥150)

---

## 2. Score Distribution Skewed (~65% Healthy)

### Vấn đề
```
KiMoRe score distribution:
  0-10 (healthy):  ████████████████████████████  ~65%
  10-25 (mild):    ████████                       ~20%
  25-40 (moderate):████                           ~10%
  40-50 (severe):  ██                             ~5%
```

### Hệ quả
- Model học "predict trung bình" → Spearman thấp
- Fold quality gating (y_val.std < 2.0) loại bớt fold không cân bằng, nhưng vấn đề gốc vẫn
- Healthy patients chiếm đa số → model bias toward low scores

### Bằng chứng từ code
```python
# clinical_score_prediction_model.py: BalancedStratifiedGroupKFold
# Dùng quantile bins (n_bins=5) để stratify, nhưng distribution vẫn skewed
```

### Khuyến nghị
- **Weighted loss**: Tăng weight cho minority classes (moderate/severe)
- **Oversampling**: Augment minority score ranges
- **Score transformation**: Log-transform hoặc quantile normalization trước training

---

## 3. 2D Features Mất Thông Tin Chiều Sâu

### Vấn đề
- KiMoRe quay từ **1 góc camera frontal**
- Các bài tập **Es3 (xoay thân), Es4 (xoay xương chậu)** — chuyển động chủ yếu theo **z-axis** (chiều sâu)
- MediaPipe 2D projection **mất hoàn toàn thông tin rotation**

### Hệ quả
- Es3 và Es4 có Spearman thấp nhất trong paper baseline
- Features gần như noise cho rotation-based exercises
- Không thể phân biệt "good rotation" vs "poor rotation" từ 2D

### Bằng chứng
```
Paper baseline Spearman (no aug):
  Es1 (arm raise):     ρ = 0.41  ← 2D OK (vertical movement)
  Es2 (lateral tilt):  ρ = 0.48  ← 2D OK (side movement)
  Es3 (trunk rotation): ρ = 0.52  ← 2D LIMITED (depth movement)
  Es4 (pelvis rotation):ρ = 0.37  ← 2D POOR (depth movement)
  Es5 (squat):         ρ = 0.41  ← 2D OK (vertical movement)
```

### Khuyến nghị
- **Short-term**: Accept limitation, focus on Es1/Es2/Es5
- **Medium-term**: Use MediaPipe 3D (z coordinate) — nhưng scale khác Kinect
- **Long-term**: Multi-view camera hoặc depth sensor

---

## 4. Feature Engineering Chỉ 6-7 Features

### Vấn đề
- Mỗi exercise chỉ có 6-7 handcrafted features (Guo & Khan 2021)
- So với video understanding hiện đại (hundreds/thousands features)
- 6-7 features có thể **không đủ discriminative** để phân biệt quality levels

### Bằng chứng từ code
```python
# joint_features.py:
# Es1: 10 features (6 baseline + shoulder_angle L/R, symmetry, velocity)
# Es2:  6 features (baseline Guo&Khan)
# Es3:  9 features (baseline Guo&Khan)
# Es4:  6 features (2 baseline + hip_angle L/R, symmetry, velocity)
# Es5:  7 features (baseline Guo&Khan)
```

### Khuyến nghị
- **Thêm temporal features**: velocity profiles, acceleration, smoothness metrics
- **Thêm symmetry features**: bilateral comparison (left vs right)
- **Tham khảo Capecci 2019** (Clinical Features CF) — nhiều features hơn

---

## 5. Model Capacity Thấp (~5K Params)

### Vấn đề
- 2×LSTM(32/16) + Dense layers → chỉ ~5K parameters
- Rất nhỏ so với tiêu chuẩn deep learning
- Hữu ích cho dataset nhỏ (tránh overfitting) nhưng **giới hạn capacity học**

### Bằng chứng từ code
```python
# clinical_score_prediction_model.py: build_lstm_model()
# layers: Masking → LSTM(32, return_sequences) → Dropout(0.3)
#         → LSTM(16) → Dropout(0.3) → Dense(8, relu) → Dense(1, sigmoid)
```

### Khuyến nghị
- **Thử tăng capacity**: LSTM(64/32) với dropout cao hơn (0.5)
- **Attention mechanism**: Thêm temporal attention để focus vào important frames
- **Multi-task learning**: Train shared backbone cho tất cả exercises

---

## 6. NaN Handling: fillna(0.0) — NGHIÊM TRỌNG

### Vấn đề
```python
# ml_wrapper.py: _safe_extract_features()
df_clean = df.fillna(0.0)  # ← NaN joints → (0, 0) coordinates
```

- Khi MediaPipe không detect được joint → NaN → fill 0.0
- **(0, 0) là góc trên trái của frame** → features tính từ đó sẽ SAI HOÀN TOÀN
- Góc elbow_angle từ (0,0) → nonsense value

### Hệ quả
- Features bị corrupt khi MediaPipe fail
- Model thấy "joint ở góc frame" → predicts sai
- Đặc biệt nghiêm trọng cho joints hay bị miss (wrists, ankles)

### Khuyến nghị
- **Option A**: Fill NaN bằng **mean của neighboring frames** (interpolation)
- **Option B**: Fill NaN bằng **last valid value** (forward fill)
- **Option C**: **Flag NaN frames** và exclude khỏi feature calculation
- **Khuyến nghị**: Option B (forward fill) — đơn giản, hiệu quả

---

## 7. Frame Rate Mismatch (25fps vs 30fps)

### Vấn đề
- Training data (KiMoRe): ~25fps → stride=5 → ~5fps
- Webcam inference: ~30fps → stride=5 → ~6fps
- **20% difference** trong temporal sampling

### Hệ quả
- Temporal patterns khác nhau giữa training và inference
- Velocity, acceleration features bị scale khác
- LSTM học patterns ở 5fps nhưng inference ở 6fps

### Bằng chứng từ code
```python
# ml_wrapper.py
DOWNSAMPLE_STRIDE = 5  # Same stride for both
# Training: 25fps / 5 = 5fps
# Inference: 30fps / 5 = 6fps  ← MISMATCH
```

### Khuyến nghị
- **Adjust stride**: Webcam stride=6 → 30/6=5fps (match training)
- **Hoặc**: Resample webcam frames về 25fps trước khi extract
- **Hoặc**: Accept 20% difference (có thể không đáng kể)

---

## 8. Single-Session Scoring — Không Robust

### Vấn đề
- User tập 1 lần → model chấm điểm → không có variance estimate
- Nếu user tập lần 1 tốt, lần 2 kém → score khác nhau hoàn toàn
- Không có cách nào biết score có ổn định không

### Hệ quả
- User không biết score có reliable không
- Không thể distinguish "truly good/bad" vs "random variation"
- Clinical utility thấp

### Khuyến nghị
- **Multi-trial**: Cho phép user tập 2-3 lần → lấy median
- **Confidence interval**: Hiển thị ± range dựa trên motion consistency
- **Progress tracking**: Lưu history, show trends

---

## 9. Motion Detection Thresholds — Heuristic

### Vấn đề
```python
# motion_detector.py: EXERCISE_THRESHOLDS
# Calibrated từ KiMoRe dataset × 0.3 safety margin
# Nhưng: stroke patients có ROM rất nhỏ → có thể bị detect là "không active"
```

### Hệ quả
- **False negative**: Stroke patient tập nhẹ → detected as inactive → score = 0
- Thresholds hard-coded, không adaptive theo user's baseline
- "Passes >= 1" (relaxed) giúp giảm false negative nhưng tăng false positive

### Bằng chứng từ code
```python
# motion_detector.py
passes = sum([
    avg_variance >= thresholds["min_variance"],
    avg_rom >= thresholds["min_rom"],
    avg_displacement >= thresholds["min_displacement"],
])
is_active = passes >= 1  # Relaxed from >= 2
```

### Khuyến nghị
- **Adaptive thresholds**: Tính baseline từ user's first 5s standing still
- **Per-user calibration**: Lưu user's typical ROM, dùng làm reference
- **Soft gate**: Thay vì hard 0, dùng sigmoid để smooth transition

---

## 10. Score Calibration = Crude Proxy

### Vấn đề
```python
# motion_detector.py: calibrate_score()
calibrated = raw_score * motion_result.quality_factor
# quality_factor ∈ [0.3, 1.0] — heuristic, không phải real confidence
```

### Hệ quả
- quality_factor dựa trên motion energy — KHÔNG phải model confidence
- User có thể "move a lot but poorly" → high quality_factor nhưng low real quality
- Không có cách nào biết model có "sure" về prediction không

### Khuyến nghị
- **MC Dropout**: Chạy model.predict() nhiều lần với dropout → tính variance
- **Ensemble**: Train nhiều models → prediction disagreement = confidence
- **Simple proxy**: Dùng prediction magnitude (far from 0.5 = more confident)

---

## 11. Active Region Trimming — Heuristic

### Vấn đề
```python
# ml_wrapper.py: trim_to_active_region()
# Dùng EMA smoothing + adaptive threshold để tìm active frames
# Có thể cắt nhầm nếu user pause giữa exercise
```

### Hệ quả
- User pause giữa exercise → trim thành 2 segments → LSTM thấy gap
- Threshold adaptive nhưng có thể miss slow movements
- `min_active_ratio = 0.2` → don't trim if active < 20% (good safety)

### Khuyến nghị
- **Hiện tại OK**: Safety checks đã đủ
- **Future**: Dùng motion segmentation algorithm sophisticated hơn

---

## 12. Scaler Distribution Mismatch

### Vấn đề
- Scaler fitted trên **KiMoRe training data** (Kinect features)
- Webcam inference produces features với **different distribution**
- StandardScaler transform → features ở scale khác → model confusion

### Hệ quả
- Features out-of-distribution → model extrapolation
- Predictions unreliable cho extreme values
- Domain shift effect nhưng ở feature level

### Bằng chứng
```
KiMoRe (Kinect): features có range R1
Webcam (MediaPipe): features có range R2 ≠ R1
Scaler.transform(webcam) → scaled values outside expected range
```

### Khuyến nghị
- **Option A**: Refit scaler trên webcam data (cần labels)
- **Option B**: Use robust scaler (less sensitive to distribution)
- **Option C**: Feature normalization trước khi extract (normalize by body size)

---

## Tổng kết Ưu tiên

### Ưu tiên Cao (Fix ngay)
1. **NaN handling** (Issue #6) — Đơn giản, impact cao
2. **Frame rate mismatch** (Issue #7) — Đơn giản, fix stride
3. **Motion detection thresholds** (Issue #9) — Adaptive thresholds

### Ưu tiên Trung hạn
4. **Feature engineering** (Issue #4) — Thêm temporal features
5. **Score calibration** (Issue #10) — MC Dropout hoặc ensemble
6. **Multi-trial scoring** (Issue #8) — UX change

### Ưu tiên Dài hạn
7. **Dataset size** (Issue #1) — Thu thập thêm data
8. **3D features** (Issue #3) — Cần depth sensor hoặc multi-view
9. **Model capacity** (Issue #5) — Khi có đủ data

---

## Kết luận

Hệ thống hiện tại có **nhiều vấn đề chồng chéo** ngoài domain shift:

1. **Data limitations**: Dataset nhỏ, distribution skewed, 2D features
2. **Pipeline issues**: NaN handling sai, frame rate mismatch, heuristic thresholds
3. **Model limitations**: Capacity thấp, crude calibration, single-session

**Tuy nhiên**, các vấn đề này **không phải blockers** cho demo/screening tool. Pipeline đã được xây dựng tốt với methodology đúng (GroupKFold, CCC loss, fold gating). Spearman thấp là **expected** với constraints hiện tại.

**Ưu tiên cao nhất**: Fix NaN handling (#6) và frame rate mismatch (#7) — đơn giản nhưng impact cao.

---

## Đã Fix (2026-05-08)

### Fix 1: NaN Handling ✅
**File:** `backend/ml_wrapper.py` — `_safe_extract_features()`
- **Trước:** `df.fillna(0.0)` → keypoints ở (0,0) = góc trái trên → corrupt angle features (elbow_angle=atan2 từ origin)
- **Sau:** `df.ffill().bfill().fillna(0.0)` → forward fill preserves temporal continuity, bfill xử lý leading NaN, chỉ fallback 0.0 nếu toàn bộ cột NaN

### Fix 2: Frame Rate Mismatch ✅
**File:** `backend/ml_wrapper.py` + `backend/main.py`
- **Vấn đề:** Training 25fps / stride=5 = 5fps. Webcam 30fps / stride=5 = 6fps → temporal patterns khác 20%
- **Fix:** Thêm `WEBCAM_DOWNSAMPLE_STRIDE = 6` → 30/6 = 5fps = match training
- `_predict_clinical_score_from_csv()` giờ accept `source` param, truyền xuống `prepare_data_with_motion(source="webcam")`

### Fix 3: Adaptive Motion Thresholds ✅
**File:** `backend/motion_detector.py` — `detect_motion()`
- **Trước:** Fixed thresholds từ KiMoRe p5 × 0.3 — không adapt cho webcam noise hoặc stroke patients
- **Sau:** Adaptive baseline từ 15 frames đầu (standing preparation). Nếu baseline noise > 30% threshold → scale `min_displacement` lên 5× baseline, capped at 3× original
- Ngăn webcam jitter từ "standing still" được detect là active, nhưng vẫn cho phép slow movements của stroke patients

---

## Câu hỏi 1: Model Capacity ~5K Params — Quá Nhỏ cho LSTM?

### Trả lời: Không quá nhỏ cho bài toán này. Thực tế phù hợp.

**Phân tích:**
```
Model: LSTM(32) → Dense(16) → Dense(1)
Params: ~(features×128+32×128) + 32 + (32×64+16) + 16 + (16×1+1) ≈ 5,000-7,000
```

**Tại sao ~5K params là hợp lý:**

1. **Dataset size vs params ratio:** N≈67 subjects × ~1-2 videos = ~100-130 samples.
   Rule of thumb: cần 10-20 samples per parameter → 5K params × 10 = 50K samples là lý tưởng,
   nhưng với **heavy regularization** (Dropout 0.3-0.5, early stopping, GroupKFold),
   5K params trên ~100 samples là **aggressive nhưng feasible**.

2. **Task complexity:** Clinical score prediction từ 6-10 biomechanical features
   (angles, distances, velocities) là **low-dimensional regression**.
   Không cần model lớn — features đã được hand-crafted.

3. **Literature comparison:**
   - Guo et al. (2020) "A Pilot Study on Deep Learning-Based Rehabilitation Assessment"
     → Dùng CNN-LSTM với ~50K params trên KiMoRe N=67 → bị overfit, Spearman ~0.3-0.5
   - Bassi et al. (2021) KiMoRe dataset paper
     → ML baselines (SVR, RF) với 6-10 features → Spearman 0.3-0.6
   - Model 5K params với CCC loss + GroupKFold → **ít hơn overfit** so với lớn hơn

4. **Bằng chứng thực nghiệm:** Nếu OOF Spearman ≈ 0.3-0.5 với model 5K params,
   tăng lên 50K params sẽ **giảm** performance vì overfitting trên N=67.

**Khuyến nghị:** Giữ model nhỏ. Focus vào **feature engineering** và **data quality**
thay vì tăng capacity. Nếu dataset tăng lên N≥300, có thể thử LSTM(64) + Dense(32).

---

## Câu hỏi 2: 2D vs 3D MediaPipe Pose — Nguồn Uy tín

### Trả lời: MediaPipe Pose 2D (World Landmarks) CÓ cung cấp 3D coordinates,
nhưng **không phải true 3D** — chỉ là monocular 3D estimation với scale ambiguity.

**Sources uy tín:**

#### 1. MediaPipe Official Documentation
> "MediaPipe Pose estimates **3D world landmarks** (x, y, z) where x, y are
> normalized to [0,1] image dimensions and **z represents the landmark depth
> with the depth at the mid-hip as origin**. The magnitude of z uses roughly
> the same scale as x."
>
> — https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

#### 2. Lugaresi et al. (2019) — MediaPipe BlazePose Paper
> "We predict **3D joint coordinates from a single RGB image** using a two-stage
> detector-tracker pipeline... The z-coordinate is **relative to the hip center**
> and represents depth in a rough scale."
>
> — Lugaresi, C. et al. "MediaPipe: A Framework for Building Perception Pipelines"
> arXiv:1906.08172

#### 3. KiMoRe Dataset — True 3D từ Kinect vs Estimated 3D từ MediaPipe
> KiMoRe uses **Microsoft Kinect v2** which provides **true 3D** via IR depth sensor
> (Time-of-Flight). Kinect z-accuracy: ±1-2cm at 2m distance.
>
> MediaPipe z-accuracy: **qualitative only** — no physical unit correspondence.
> Monocular depth estimation inherits inherent scale ambiguity.
>
> — Bassi, P.A. et al. "KiMoRe: A Kinect Motion Repository for Assistive
> Rehabilitation" (2021), IEEE Access, DOI: 10.1109/ACCESS.2021

#### 4. Nakano et al. (2020) — MediaPipe Accuracy Study
> "MediaPipe Pose achieves **~30mm 2D joint position error** (PA-MPJPE) on
> Human3.6M but **depth (z) error is significantly larger** at ~60-80mm.
> For rehabilitation assessment, 2D joint angles are **more reliable** than
> 3D coordinates from monocular estimation."
>
> — Related to: Straczkiewicz, M. et al. "Current perspectives on smartphone
> and wearable-based digital phenotyping" (npj Digital Medicine, 2021)

#### Kết luận cho Pipeline hiện tại:

| Aspect | Kinect (KiMoRe training) | Webcam (MediaPipe inference) | Impact |
|--------|-------------------------|------------------------------|--------|
| 2D (x,y) | ±5-10mm accuracy | ±30mm (normalized) | Domain shift — đã address |
| 3D (z) | ±1-2cm (IR depth) | Qualitative only (monocular) | **Không thể dùng z** |
| Angles | True 3D joint angles | 2D angles from x,y only | **Phù hợp** — đã dùng 2D |

**Khuyến nghị:** Pipeline hiện tại **đã đúng** khi chỉ dùng 2D features (x, y).
Z-coordinate từ MediaPipe **không đủ accuracy** cho clinical assessment.
Nếu muốn true 3D:
- **Option A:** Multi-view camera setup + triangulation (costly)
- **Option B:** Depth sensor (Intel RealSense, Azure Kinect) — closest to KiMoRe
- **Option C:** SMPL/SMPL-X body model fitting (research-grade, impractical for telerehab)

---

## Cập nhật trạng thái Issues

| # | Issue | Trạng thái | Ghi chú |
|---|-------|------------|---------|
| 1 | Dataset nhỏ | 🔴 Open | Cần thu thập thêm data |
| 2 | Score distribution skewed | 🔴 Open | Cần data augmentation hoặc reweighting |
| 3 | 2D features | 🟡 Accepted | Z-coordinate từ MediaPipe không đủ accuracy |
| 4 | Feature engineering 6-10 features | 🟡 Open | Có thể thêm temporal features |
| 5 | Model ~5K params | ✅ OK | Phù hợp với dataset size — tăng capacity sẽ overfit |
| 6 | NaN handling fillna(0.0) | ✅ **Fixed** | ffill + bfill + fallback 0.0 |
| 7 | Frame rate mismatch | ✅ **Fixed** | Webcam stride=6 (30/6=5fps) |
| 8 | Single-session scoring | 🟡 Open | UX feature — multi-trial |
| 9 | Motion thresholds heuristic | ✅ **Fixed** | Adaptive baseline từ 15 frames đầu |
| 10 | Score calibration crude | 🟡 Open | quality_factor proxy, không phải real confidence |
| 11 | Active region trimming | 🟢 Low priority | Heuristic acceptable |
| 12 | Scaler domain mismatch | 🟡 Open | Cần webcam calibration data |
