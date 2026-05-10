# C3. Feature Extraction Divergence — Phân tích chi tiết & Hướng giải quyết

**Ngày:** 2026-05-10
**Tác giả:** Senior Backend Engineer review
**Severity:** 🔴 CRITICAL
**Status:** Phân tích hoàn tất, chờ review trước khi implement

---

## 1. Tổng quan vấn đề

Dự án tồn tại **3 pipeline feature extraction KHÁC NHAU** cho cùng một tác vụ (predict clinical score từ skeleton data). Khi model được train bằng pipeline A nhưng inference dùng pipeline B, **features đầu vào hoàn toàn khác nhau** → model predict garbage.

---

## 2. Chi tiết 3 Pipeline

### Pipeline #1 — `03_extract_joint_features.py` + `backend/joint_features.py` ✅ ĐÚNG

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Input** | Raw MediaPipe/Kinect keypoints (x, y, z) |
| **Xử lý** | Aspect ratio correction (`x * 1.7778`), Median filter (window=5) |
| **Features** | 2D biomechanical: `elbow_angle`, `shoulder_angle`, `torso_tilt`, `hand_shoulder_ratio`, `knee_angle`, velocity features, v.v. |
| **Output** | CSV files chứa 6-10 features/exercise (tùy exercise) |
| **Số features** | Es1=10, Es2=8, Es3=9, Es4=6, Es5=8 |
| **Dùng bởi** | `clinical_score_prediction_model.py` (load từ CSV column `joint_features`) |
| **Backend sync** | `backend/joint_features.py` — **ĐÃ ĐỒNG BỘ** |

**Data flow:**
```
KiMoRe videos → MediaPipe/Kinect → raw keypoints
    → 03_extract_joint_features.py → CSV files (biomechanical features)
    → clinical_score_prediction_model.py reads column "joint_features" → load CSV → train LSTM
```

### Pipeline #2 — `evaluate_models.py` ❌ HOÀN TOÀN KHÁC

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Input** | CSV column `joint_positions` (raw keypoints, KHÔNG PHẢI `joint_features`) |
| **Xử lý** | Drop face columns (nose, eyes, ears = 15 cols), mid-shoulder normalization, downsample 6x |
| **Features** | **Raw joint coordinates** (y, x, confidence) — KHÔNG có feature engineering |
| **Output** | (N, max_len, ~36 raw features) — coordinate arrays |
| **Số features** | ~36 raw coordinates (12 body joints × 3 channels - face) |
| **Model** | Bidirectional LSTM(64/32) hoặc Transformer — **KHÁC architecture với Pipeline #1** |
| **Masking** | `mask_value=0.0` — **SAI** (Pipeline #1 dùng `-999.0`) |

**Data flow:**
```
KiMoRe CSV → column "joint_positions" → load raw keypoints CSV
    → drop face → mid-shoulder normalization → pad with zeros
    → train BiLSTM/Transformer trên raw coordinates
```

### Pipeline #3 — `backend/ml_wrapper.py` (Inference) ✅ KHỚP VỚI #1

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Input** | MediaPipe landmarks từ webcam/video |
| **Xử lý** | `joint_features.py` → aspect ratio correction, median filter |
| **Features** | 2D biomechanical (giống Pipeline #1) |
| **Scaler** | Load `scaler_{exercise}.joblib` (fit bởi Pipeline #1) |
| **Masking** | `MASK_VALUE = -999.0` ✅ |

---

## 3. Bảng so sánh chi tiết sự khác biệt

| Aspect | Pipeline #1 (training) | Pipeline #2 (evaluate_models) | Pipeline #3 (inference) |
|--------|:---:|:---:|:---:|
| **Data column** | `joint_features` | `joint_positions` | Realtime landmarks |
| **Feature type** | Biomechanical (angles, ratios) | Raw coordinates (x, y, conf) | Biomechanical ✅ |
| **# Features/exercise** | 6-10 | ~36 (all joints) | 6-10 ✅ |
| **Aspect ratio** | `x * 1.7778` | Không có | `x * 1.7778` ✅ |
| **Median filter** | `window=5` | Không có | `window=5` ✅ |
| **Normalization** | StandardScaler per-fold | Mid-shoulder subtraction | StandardScaler ✅ |
| **Downsample** | `stride=5` | `stride=6` | `stride=5/6` ✅ |
| **Mask value** | `-999.0` | `0.0` ❌ | `-999.0` ✅ |
| **CV strategy** | BalancedStratifiedGroupKFold | KFold (no group!) ❌ | N/A |
| **Loss** | CCC loss | CCC loss | N/A |
| **Architecture** | LSTM(32/16) | BiLSTM(64/32) / Transformer ❌ | Load trained model |

---

## 4. Root Cause Analysis — Tại sao C3 gây low Spearman ρ = 0.11–0.32?

### Kịch bản gây lỗi:

```
1. Bạn chạy evaluate_models.py để đánh giá model
2. evaluate_models.py dùng column "joint_positions" → raw coordinates (36 features)
3. Nhưng model được train bởi clinical_score_prediction_model.py 
   dùng column "joint_features" → biomechanical features (6-10 features)
4. Feature dimensions KHÁC NHAU → model không thể load
5. evaluate_models.py BUILD MODEL MỚI (BiLSTM 64/32) → train từ đầu trên raw data
6. Raw coordinates + KFold (no group) + mask=0.0 → kết quả kém
7. Metrics báo ρ = 0.11–0.32 → bạn tưởng model kém nhưng thực ra là ĐÁNH GIÁ SAI
```

### Tóm lại:

> **`evaluate_models.py` đánh giá trên một pipeline HOÀN TOÀN KHÁC** — nó build model mới, train trên data khác, dùng architecture khác. Kết quả ρ = 0.11–0.32 KHÔNG phản ánh chất lượng thực sự của model được train bởi `clinical_score_prediction_model.py`.

---

## 5. Mức độ nghiêm trọng THỰC TẾ — Có phải C3 vẫn đang gây hại không?

### Tin TỐT ✅

| Cặp file | Status | Giải thích |
|----------|--------|------------|
| `03_extract_joint_features.py` ↔ `backend/joint_features.py` | ✅ **ĐỒNG BỘ** | Cùng features, cùng logic, cùng order. Vừa update Es1 (10 features) và Es5 (8 features). |
| `clinical_score_prediction_model.py` ↔ `backend/joint_features.py` | ✅ **TƯƠNG THÍCH** | Training script load CSV files (`joint_features` column) mà `03_extract_joint_features.py` đã generate → features giống hệt backend. |
| `ml_wrapper.py` pipeline | ✅ **ĐÚNG** | Dùng `joint_features.py` functions → giống training. Scaler, mask value, downsample đều match. |

### Tin XẤU ❌

| Cặp file | Status | Giải thích |
|----------|--------|------------|
| `evaluate_models.py` | ❌ **ORPHANED** | Pipeline hoàn toàn tách biệt. Dùng raw coords thay vì biomechanical features. KFold thay vì GroupKFold. Masking sai (0.0 vs -999.0). **Metrics từ file này KHÔNG CÓ GIÁ TRỊ** cho model hiện tại. |

---

## 6. Hướng giải quyết

### Option A: Xóa `evaluate_models.py` (Nhanh nhất — Recommended nếu đang gấp)

**Lý do:**
- File này là di tích từ giai đoạn prototype ban đầu
- `clinical_score_prediction_model.py` **đã có OOF evaluation** tích hợp (Spearman, CCC, MAE per fold)
- Không cần script evaluation riêng biệt nữa

**Thực hiện:**
```
1. Xóa hoặc rename evaluate_models.py → evaluate_models.py.deprecated
2. Update audit report: C3 → RESOLVED
```

### Option B: Rewrite `evaluate_models.py` để dùng Pipeline #1 (Proper fix)

**Thực hiện:**
1. Thay `prepare_data()` bằng `load_raw_features()` từ training script
2. Thay `build_lstm()` bằng `build_model()` từ training script
3. Thay `KFold` bằng `BalancedStratifiedGroupKFold`
4. Thay `mask_value=0.0` bằng `MASK_VALUE=-999.0`
5. Load existing trained models thay vì train model mới

### Option C: Import trực tiếp từ `clinical_score_prediction_model.py`

**Thực hiện:**
- `from clinical_score_prediction_model import load_raw_features, build_model, BalancedStratifiedGroupKFold`
- Tuy nhiên, 2 scripts nằm cùng thư mục nên import trực tiếp ok

---

## 7. Kết luận

| Câu hỏi | Trả lời |
|---------|---------|
| **Pipeline training → inference có bị lệch không?** | ❌ **KHÔNG.** Pipeline #1 (training) và Pipeline #3 (inference) đã đồng bộ. |
| **C3 có đang gây model performance kém không?** | ❌ **KHÔNG trực tiếp.** Model hiện tại train và infer đúng pipeline. |
| **C3 có gây metrics sai không?** | ✅ **CÓ.** `evaluate_models.py` report ρ = 0.11–0.32 là sai lệch, KHÔNG phản ánh model thực tế. Metrics đúng nằm trong output của `clinical_score_prediction_model.py` (OOF ρ = 0.44–0.64). |
| **Cần fix gấp không?** | ⚠️ **Option A (deprecate file) là đủ** để tránh nhầm lẫn metrics. Option B làm khi rảnh. |

---

## 8. Checklist verification

```
1. [x] 03_extract_joint_features.py ↔ backend/joint_features.py: ĐỒNG BỘ
2. [x] clinical_score_prediction_model.py reads "joint_features" CSV: ĐÚNG
3. [x] ml_wrapper.py dùng joint_features.py: ĐÚNG
4. [x] Mask value: -999.0 cả train lẫn inference: ĐÚNG
5. [x] Scaler: fit trên train fold, load khi inference: ĐÚNG
6. [ ] evaluate_models.py: CẦN DEPRECATE hoặc REWRITE
```
