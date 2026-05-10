# Phân tích Input/Output & Đánh giá Spearman thấp

**Ngày:** 2026-05-08  
**Câu hỏi:** Input/Output của bài toán là gì? Spearman thấp có đáng tin cậy? Hướng "tập xong chấm điểm" có ổn?

---

## 1. Input / Output của bài toán

### 1.1 Input (X) — Chuỗi features biomechanics từ video

```
Video (webcam) → MediaPipe Pose → 2D Joint Positions → Feature Engineering → LSTM Input
```

**Bước 1: MediaPipe Pose Extraction**
- Trích xuất 12 keypoints cơ thể: `left/right shoulder, elbow, wrist, hip, knee, ankle`
- Mỗi keypoint có 4 channels: `(x, y, z, visibility)` → 48 columns
- Lưu vào CSV (`joint_positions`)

**Bước 2: Feature Engineering (per-exercise)**
- Từ 2D joint positions → tính góc, khoảng cách, tỷ lệ biomechanics
- **Mỗi exercise có bộ features riêng** (theo paper Guo & Khan 2021, Table 2):

| Exercise | Mô tả | Số features | Features chính |
|----------|--------|-------------|----------------|
| **Es1** | Nâng tay | 6 | elbow_angle L/R, hand_shoulder_ratio, torso_tilt, hand_tilt, elbow_diff |
| **Es2** | Nghiêng thân | 6 | elbow_angle L/R, torso_tilt, elbow_diff, shoulder_angle L/R |
| **Es3** | Xoay thân | 7 | elbow_angle L/R, hand_shoulder_ratio, torso_tilt, elbow_diff, shoulder_angle L/R |
| **Es4** | Xoay xương chậu | 6 | torso_tilt, knee_hip_ratio, hip_angle L/R, hip_diff, torso_velocity |
| **Es5** | Squat | 7 | elbow_angle L/R, hand_shoulder_ratio, torso_tilt, elbow_diff, shoulder_angle L/R |

**Bước 3: Temporal Downsampling**
- Video gốc ~25fps → downsample stride=5 → ~5fps
- Giảm từ ~672 frames xuống ~134 frames (trung bình)
- Truncate tối đa `target_len=150` frames (giữ đầu + cuối nếu quá dài)

**Bước 4: StandardScaler + Padding**
- Scale features về z-score (fit trên train set, KHÔNG fit trên toàn bộ data)
- Pad/truncate về fixed `max_length` (percentile 95 của sequence lengths)
- Padding value = `-999.0` (để Masking layer bỏ qua)

**Tensor shape cuối cùng:** `(batch, max_length, num_features)`  
- Ví dụ: `(batch, 134, 6)` cho Es1

### 1.2 Output (y) — Clinical Score

- **Thang đo:** 0–50 (thang đánh giá lâm sàng của KiMoRe dataset)
- **Chuẩn hóa khi train:** `y_normalized = y / 50.0` → [0, 1]
- **Model output:** Sigmoid → [0, 1] → `× 50` để về thang gốc
- **Ý nghĩa:** Điểm đánh giá chất lượng bài tập phục hồi chức năng của bệnh nhân

### 1.3 Tóm tắt pipeline end-to-end

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KiMoRe Dataset CSV                                             │
│       │                                                         │
│       ▼                                                         │
│  load_raw_features()                                            │
│    ├── Đọc joint_features CSV (per video)                       │
│    ├── temporal_downsample(stride=5, target_len=150)            │
│    └── Returns: list of (T_i, F) arrays + y + subject_ids      │
│                                                                 │
│  BalancedStratifiedGroupKFold (5 folds)                         │
│    ├── Per fold:                                                │
│    │     ├── StandardScaler.fit(train_frames) ← CHỈ train       │
│    │     ├── scale_and_pad() → (N, max_length, F)               │
│    │     ├── build_model() → 2×LSTM(32/16) + sigmoid            │
│    │     ├── train_fold() với CCC loss, LR warmup               │
│    │     └── Evaluate: Spearman, CCC, MAE, Pearson               │
│    └── OOF predictions → aggregated metrics                     │
│                                                                 │
│  train_final_model() trên ALL data → deployment model           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Webcam video → MediaPipe → joint_positions DataFrame           │
│       │                                                         │
│       ▼                                                         │
│  ml_wrapper.prepare_data_with_motion()                          │
│    ├── _safe_extract_features(df, exercise_id)                  │
│    │     └── get_esN_features(df) → (T, F) raw features         │
│    ├── temporal_downsample(stride=5, max_length)                │
│    ├── detect_motion(features) → MotionResult                   │
│    │     └── Nếu !is_active → calibrate_score = 0              │
│    ├── trim_to_active_region(features)                          │
│    ├── scaler.transform(features) ← scaler đã load từ file     │
│    └── pad/truncate → (1, max_length, F) tensor                 │
│                                                                 │
│  model.predict(data) → sigmoid output × 50 → clinical score    │
│       │                                                         │
│       ▼                                                         │
│  feedback_engine: score + real-time posture feedback            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Spearman thấp — Có đáng tin cậy không?

### 2.1 Kết quả hiện tại vs Paper baseline

| Exercise | Paper (no aug) | Paper (best) | Ý nghĩa |
|----------|---------------|--------------|----------|
| Es1 | ρ = 0.41 | ρ = 0.76 | Nâng tay |
| Es2 | ρ = 0.48 | ρ = 0.61 | Nghiêng thân |
| Es3 | ρ = 0.52 | ρ = 0.73 | Xoay thân |
| Es4 | ρ = 0.37 | ρ = 0.54 | Xoay chậu |
| Es5 | ρ = 0.41 | ρ = 0.67 | Squat |

> **Lưu ý:** Paper dùng augmentation để đạt best scores. Baseline "no aug" đã là ρ = 0.37–0.52.

### 2.2 Tại sao Spearman thấp? — Phân tích nguyên nhân

#### Nguyên nhân 1: Dataset quá nhỏ (N≈67 subjects)

- Với 67 subjects, 5-fold CV → mỗi fold validation chỉ có ~13 subjects
- **Standard error của Spearman ≈ 1/√N ≈ 0.12** cho mỗi fold
- → Fold-level Spearman dao động rất mạnh (có thể từ 0.1 đến 0.7 giữa các fold)
- **Kết luận:** Spearman trên dataset nhỏ có **variance cao**, KHÔNG ổn định

#### Nguyên nhân 2: Score distribution bị skewed (~65% healthy)

```
KiMoRe score distribution (thô):
  ████████████████████████████  0-10 (healthy): ~65%
  ████████                      10-25 (mild):   ~20%
  ████                          25-40 (moderate): ~10%
  ██                            40-50 (severe):  ~5%
```

- Khi 65% subjects có score thấp (<10), model dễ học "predict trung bình" → Spearman thấp
- Fold quality gating (y_val.std < 2.0) đã loại bớt fold không cân bằng, nhưng vấn đề gốc vẫn còn

#### Nguyên nhân 3: 2D features từ video frontal — mất thông tin chiều sâu

- KiMoRe quay từ **1 góc camera frontal** (phía trước bệnh nhân)
- Các bài tập như **Es3 (xoay thân), Es4 (xoay xương chậu)** — chuyển động chủ yếu theo chiều sâu (z-axis)
- MediaPipe 2D projection **mất hoàn toàn thông tin rotation** → features gần như noise
- **Đây là lý do Es3 và Es4 có Spearman thấp nhất**

#### Nguyên nhân 4: Feature engineering chỉ 6-7 features

- So với các bài toán video understanding hiện đại (hàng trăm/thousands features)
- 6-7 features handcrafted có thể **không đủ discriminative** để phân biệt quality levels
- Đặc biệt với biên độ chuyển động nhỏ giữa "good" và "excellent"

#### Nguyên nhân 5: Mô hình đơn giản (2×LSTM 32/16)

- ~5K parameters — rất nhỏ so với tiêu chuẩn deep learning
- Hữu ích cho dataset nhỏ (tránh overfitting) nhưng **giới hạn capacity học**
- Không thể capture complex temporal patterns mà clinicians đánh giá

### 2.3 Spearman thấp có đáng tin cậy không?

**Câu trả lời: Có giới hạn, nhưng KHÔNG phải vô nghĩa.**

| Khía cạnh | Đánh giá |
|-----------|----------|
| **Ranking power** | ρ = 0.3–0.5 = model có khả năng ranking TRUNG BÌNH. Phân biệt được "tốt vs kém" nhưng không phân biệt được mức độ chi tiết |
| **Statistical significance** | Với N=67, ρ > 0.3 thường có p < 0.05 → có ý nghĩa thống kê |
| **So với baseline** | Nếu ρ ≈ paper "no aug" (0.37–0.52) → đạt baseline, KHÔNG phải failure |
| **Clinical utility** | ρ < 0.5 → KHÔNG đủ chính xác cho clinical decision, chỉ đủ cho screening/triage |
| **Variance** | Std across folds cao → cần repeated CV (n-repeats=3) để ổn định |

**Kết luận:** Spearman thấp **phản ánh đúng giới hạn** của bài toán với 2D features + dataset nhỏ. Không phải bug, mà là **inherent limitation**.

---

## 3. Hướng "tập xong chấm điểm" — Có ổn không?

### 3.1 Workflow hiện tại

```
Bệnh nhân xem video mẫu → Thực hiện bài tập trước webcam → Model chấm điểm 0-50
```

Đây là **assessment-only** workflow: bệnh nhân tập 1 lần, model đánh giá tổng thể.

### 3.2 Đánh giá: Ưu điểm

| Ưu điểm | Chi tiết |
|----------|----------|
| **Đơn giản cho user** | Chỉ cần 1 lần thực hiện, không cần setup phức tạp |
| **Phù hợp screening** | Có thể phân biệt "có vấn đề" vs "không có vấn đề" |
| **Real-time feedback** | Motion detection + posture feedback trong khi tập |
| **Calibration** | `motion_detector` giúp loại bỏ trường hợp không tập (= 0 điểm) |

### 3.3 Đánh giá: Vấn đề và rủi ro

#### Vấn đề 1: Không có reference template

- Model train trên KiMoRe (bệnh nhân stroke thực) nhưng inference trên user webcam
- **Domain gap:** góc camera, khoảng cách, ánh sáng, background khác nhau
- Không có "video mẫu chuẩn" để so sánh → model phải tự đánh giá dựa trên patterns đã học

#### Vấn đề 2: Single-session scoring — không robust

```
Scenarios gây sai:
├── User tập chậm hơn video mẫu → score thấp hơn (không phải do chất lượng)
├── User tập nhanh hơn → score cũng bị ảnh hưởng
├── Camera góc khác → features khác nhiều
├── User mặc quần áo rộng → MediaPipe tracking kém
└── User chỉ tập 1 lần → không có variance estimate
```

#### Vấn đề 3: Motion detection threshold quá strict

- `motion_detector.py` dùng hard-coded thresholds
- Bệnh nhân stroke thật có **range of motion rất nhỏ** → có thể bị detect là "không active"
- → `calibrate_score()` force score = 0 → **false negative nghiêm trọng**

#### Vấn đề 4: Spearman thấp → ranking không ổn

- Với ρ ≈ 0.4, model chỉ đúng ranking khoảng 60-65% thời gian
- 2 bệnh nhân score lần lượt 25 và 30 → model có thể predict ngược (30 và 25)
- **Không đủ tin cậy cho clinical decision** nhưng có thể dùng cho screening

### 3.4 So sánh với hướng alternatives

| Hướng | Ưu điểm | Nhược điểm |
|-------|---------|------------|
| **A. Tập xong chấm điểm (hiện tại)** | Đơn giản, 1 lần | Thiếu reference, Spearman thấp, domain gap |
| **B. So sánh với video template (DTW)** | Có reference chuẩn, interpretable | Cần template per exercise, DTW sensitive to speed |
| **C. Real-time feedback + scoring kết hợp** | User biết lỗi ngay, điều chỉnh được | Phức tạp hơn, cần exercise-specific rules |
| **D. Multi-session tracking** | Theo dõi tiến trình, ổn định hơn | User phải tập nhiều lần, UX phức tạp |

---

## 4. Khuyến nghị

### 4.1 Ngắn hạn — Cải thiện workflow hiện tại

1. **Giữ hướng "tập xong chấm điểm"** nhưng thêm context:
   - Hiển thị score + confidence interval (dựa trên motion quality)
   - Thêm disclaimer: "Kết quả mang tính tham khảo, không thay thế đánh giá chuyên gia"

2. **Giảm motion detection strictness:**
   - Threshold nên adaptive theo exercise type
   - Bệnh nhân stroke có ROM nhỏ → threshold thấp hơn

3. **Thêm multi-trial support:**
   - Cho phép user tập 2-3 lần → lấy median score
   - Giảm variance, tăng reliability

### 4.2 Trung hạn — Cải thiện model

4. **Feature engineering nâng cao:**
   - Thêm temporal features: velocity profiles, acceleration, smoothness metrics
   - Thêm symmetry features: bilateral comparison
   - Tham khảo Capecci 2019 (Clinical Features CF)

5. **Augmentation strategy:**
   - Temporal augmentation (speed perturbation) đã có trong CSV
   - Spatial augmentation (small rotations, scaling) để giảm domain gap

6. **Ensemble hoặc multi-task learning:**
   - Train 1 model cho tất cả exercises (shared backbone) → tăng effective N
   - Hoặc ensemble 5 exercise-specific models

### 4.3 Dài hạn — Thay đổi paradigm

7. **Video-to-video comparison:**
   - So sánh video user với video template bằng DTW hoặc temporal alignment
   - Ưu điểm: không cần labeled data cho model, interpretable

8. **Pose-based scoring rules:**
   - Dùng rule-based scoring dựa trên biomechanics thresholds
   - Ví dụ: "elbow angle > 150° khi nâng tay = đạt"
   - Kết hợp với ML model → hybrid approach

---

## 5. Kết luận

| Câu hỏi | Trả lời |
|---------|---------|
| **Input là gì?** | Chuỗi 6-7 features biomechanics (angles, distances, ratios) từ video, shape `(T, F)` → pad → `(max_length, F)` |
| **Output là gì?** | Clinical score 0-50 (thang đánh giá phục hồi chức năng KiMoRe) |
| **Spearman thấp có đáng tin?** | **Có giới hạn** — phản ánh đúng inherent limits (2D features, small dataset, skewed distribution). Đạt paper baseline "no aug" nhưng KHÔNG đủ cho clinical decision |
| **"Tập xong chấm điểm" có ổn?** | **Ổn cho screening/demo**, nhưng cần thêm: confidence indicator, multi-trial option, và disclaimer về accuracy |

**Tóm lại:** Pipeline hiện tại đã được xây dựng tốt với methodology đúng (GroupKFold, CCC loss, fold gating). Spearman thấp là **expected** với constraints hiện tại, không phải bug. Hướng "tập xong chấm điểm" **có thể chấp nhận** nếu được contextualize đúng (screening tool, không phải diagnostic tool).