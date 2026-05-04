# RehabAI — Pipeline Architecture & File Reference

> Tài liệu mô tả luồng dữ liệu end-to-end từ webcam → model prediction → clinical score.
> Cập nhật: 2026-05-04

---

## Mục lục

1. [Tổng quan Pipeline](#1-tổng-quan-pipeline)
2. [Data Preparation (Offline)](#2-data-preparation-offline)
3. [Training Pipeline](#3-training-pipeline)
4. [Backend (Inference)](#4-backend-inference)
5. [Model Artifacts](#5-model-artifacts)
6. [Consistency Checklist](#6-consistency-checklist)
7. [Known Issues & Fixes](#7-known-issues--fixes)

---

## 1. Tổng quan Pipeline

```
┌─────────────┐    CSV (51 cols)    ┌──────────────┐
│  Frontend    │ ──────────────────► │   Backend    │
│  (Webcam +   │  WebSocket/REST    │   (FastAPI)  │
│   MoveNet)   │                    │              │
└─────────────┘                    └──────┬───────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  ml_wrapper.py         │
                              │  1. reorder_dataframe  │
                              │  2. feature_extraction │
                              │  3. temporal_downsample│
                              │  4. StandardScaler     │
                              │  5. pad/truncate       │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  LSTM Model (.keras)   │
                              │  sigmoid output [0,1]  │
                              │  → × 50 → × 2         │
                              │  → score [0, 100]      │
                              └───────────────────────┘
```

### Scoring Pipeline (Summary)

| Step | Training | Inference | Match? |
|------|----------|-----------|--------|
| Raw input | CSV (x, y per joint) | WebSocket CSV (x, y per joint) | ✅ |
| Feature extraction | `03_extract_joint_features.py` | `joint_features.py` | ✅ Identical functions |
| Temporal downsample | `stride=5` | `DOWNSAMPLE_STRIDE=5` | ✅ |
| Scaler | `StandardScaler` fit per-fold | `scaler_{ExN}.joblib` from best fold | ✅ |
| Padding | `MASK_VALUE=-999.0` | `MASK_VALUE=-999.0` | ✅ |
| Model output | sigmoid → [0, 1] | auto-detect: `× 50 × 2` → [0, 100] | ✅ |

---

## 2. Data Preparation (Offline)

Chạy một lần để tạo dataset từ video gốc KiMoRe.

### `data-preparation/01_extract_joint_positions.py`
- **Chức năng**: Chạy MoveNet trên video KiMoRe, trích xuất tọa độ 17 keypoints (x, y, confidence) cho mỗi frame.
- **Input**: Video `.mp4` từ KiMoRe dataset
- **Output**: CSV files chứa keypoints per frame per video

### `data-preparation/02_prepare_dataset.py`
- **Chức năng**: Gộp tất cả CSV keypoints thành 1 master CSV, kết hợp với clinical scores (Total Score 0–50).
- **Input**: CSV keypoints + KiMoRe metadata
- **Output**: `KiMoRe_data_movenet_features.csv` — chứa path tới feature files + clinical_score

### `data-preparation/03_extract_joint_features.py`
- **Chức năng**: Trích xuất **biomechanical features** (2D angles, distances, ratios) từ raw keypoints cho từng exercise.
- **Input**: Raw keypoint CSV files
- **Output**: Feature CSV files per sample (6–9 features tùy exercise)
- **⚠ CRITICAL**: Các hàm `get_esX_features()` **PHẢI khớp hoàn toàn** với `backend/joint_features.py`

#### Feature Set per Exercise (Paper 2, Table 2):

| Exercise | # Features | Features |
|----------|-----------|----------|
| Es1 | 6 | left/right elbow angle, hand/shoulder ratio, torso tilt, hand tilt, elbow diff |
| Es2 | 6 | left/right elbow angle, torso tilt, elbow diff, left/right shoulder angle |
| Es3 | 9 | Es2 + hand/shoulder ratio, left/right arm-torso angle |
| Es4 | 2 | torso tilt, knee/hip ratio |
| Es5 | 7 | left/right elbow angle, hand/shoulder ratio, torso tilt, elbow diff, left/right shoulder angle |

---

## 3. Training Pipeline

### `training_models/clinical_score_prediction_model.py`
- **Chức năng**: Train LSTM model v6 với 5-fold cross-validation.
- **Architecture**: `Input → Masking(-999) → LSTM(32) → Dropout(0.3) → LSTM(16) → Dropout(0.3) → GlobalAveragePooling1D → Dense(8, relu) → Dense(1, sigmoid)`
- **Loss**: Huber (robust với outliers)
- **Y normalization**: `y / 50.0` → target range [0, 1] khớp với sigmoid output
- **Scaler**: `StandardScaler` fit PER-FOLD (chống data leakage)
- **Output**: `ml_model_{ExN}.keras` + `scaler_{ExN}.joblib`

#### Key Training Parameters:
```
--epochs 200
--batch-size 16
--lr 0.001 (with ReduceLROnPlateau + warmup)
--downsample 5 (25fps → 5fps)
--target-len 150
```

### `training_models/evaluate_models.py`
- **Chức năng**: Đánh giá models đã train, tính MAE / Spearman / Pearson trên dataset.

### `training_models/hyperparameter_tuning.py`
- **Chức năng**: Tìm kiếm hyperparameters tối ưu.

### `training_models/lstm_model.py`
- **Chức năng**: Phiên bản LSTM cũ (trước v6). Không sử dụng.

### `training_models/export_scalers_colab.py`
- **Chức năng**: Export scalers từ Colab training environment.

---

## 4. Backend (Inference)

### `backend/main.py` — FastAPI Application
- **Chức năng**: Entry point, quản lý routes, WebSocket, model loading.
- **Startup**: Load tất cả 5 models + 5 scalers vào `ml_models_cache`.
- **Endpoints chính**:
  - `POST /api/clinical_score/{exercise_id}` — REST endpoint, nhận CSV, trả score
  - `WS /ws/session/{exercise_id}` — WebSocket real-time, nhận CSV mỗi 5s, trả progressive score
  - `GET /api/diagnostic/{exercise_id}` — Test model với synthetic data
  - `GET /api/health` — Health check
- **Score pipeline**: `_predict_clinical_score_from_csv()`
  1. Parse CSV → DataFrame
  2. `reorder_dataframe()` — map frontend columns → backend columns
  3. `prepare_data_with_motion()` — feature extraction + scaling + padding
  4. `model.predict()` → raw output [0, 1]
  5. Auto-detect: `raw × 50 × 2` → score [0, 100]

### `backend/ml_wrapper.py` — ML Preprocessing Pipeline
- **Chức năng**: Cầu nối giữa raw CSV data và model input tensor.
- **Constants**:
  - `MASK_VALUE = -999.0` — padding sentinel (khớp training)
  - `DOWNSAMPLE_STRIDE = 5` — temporal downsampling (khớp training)
  - `MEDIAPIPE_BODY_KEYPOINTS` — 12 body joints
- **Functions**:
  - `get_dataframe_cols()` → 48 columns (12 joints × 4: x, y, z, v)
  - `reorder_dataframe(df)` → reindex CSV sang expected columns, fillna(0.0)
  - `_safe_extract_features(df, exercise_id)` → gọi `joint_features.get_esX_features()`, xử lý NaN
  - `temporal_downsample(feat, stride, target)` → giảm frame rate, giữ đầu+cuối
  - `prepare_data(df, max_length, exercise_id)` → full pipeline cho REST
  - `prepare_data_with_motion(df, max_length, exercise_id)` → pipeline + motion detection
  - `load_scaler(exercise_id)` → load `scaler_{ExN}.joblib`, cached
  - `test_model_inference(model, exercise_id, max_length)` → synthetic test cho diagnostic endpoint

### `backend/joint_features.py` — 2D Feature Extraction
- **Chức năng**: Tính biomechanical features từ 2D keypoints.
- **⚠ CRITICAL**: Phải khớp hoàn toàn với `data-preparation/03_extract_joint_features.py`
- **Geometry functions**:
  - `get_joint_2d(df, name)` → (N, 2) array, aspect ratio corrected (x × 1.7778)
  - `calculate_angle_2d(first, middle, end)` → angle tại vertex middle [0°, 180°]
  - `calculate_distance_2d(p1, p2)` → Euclidean distance
  - `get_torso_vector_2d(df)` → vector từ vai xuống hông
  - `calculate_vector_angle_2d(v1, v2)` → angle giữa 2 vectors
- **Per-exercise**: `get_es1_features()` ... `get_es5_features()`

### `backend/motion_detector.py` — Motion Analysis
- **Chức năng**: Phân tích xem người dùng có thực sự tập hay không.
- **Input**: RAW features (trước scaler), shape (n_frames, n_features)
- **3 metrics**:
  1. **Temporal Variance**: features thay đổi qua thời gian
  2. **Range of Motion (ROM)**: max - min cho mỗi feature
  3. **Frame Displacement**: thay đổi giữa frames liên tiếp
- **Output**: `MotionResult` với `is_active`, `motion_energy`, `quality_factor`
- **`calibrate_score(raw, motion)`**: nhân score × quality_factor (hiện KHÔNG sử dụng — raw score trả trực tiếp)

### `backend/feedback_engine.py` — Real-time Feedback
- **Chức năng**: So sánh pose hiện tại với reference pose, tạo feedback tiếng Việt.
- **Input**: 51 floats (17 keypoints × 3 [y, x, confidence]) cho cả reference và current
- **Logic**: Tính angle tại 6 joints (vai, khuỷu tay, đầu gối) cho cả 2 bên
- **Output**: Feedback text + accuracy + severity cho mỗi joint

### `backend/database.py` — SQLite Configuration
- **Chức năng**: Cấu hình SQLAlchemy engine + session factory.
- **DB**: SQLite tại `/app/data/RehabAI.db` (Docker) hoặc `./RehabAI.db` (local)

### `backend/models.py` — Database Models
- **Tables**: `users`, `exercises`, `assigned_exercises`, `progress_tracker`
- `progress_tracker` lưu `score` (clinical score) cho mỗi lần tập

### `backend/data_wrapper.py` — Database Initialization
- **Chức năng**: Seed 5 exercises (Es1–Es5) vào DB nếu chưa tồn tại.
- **Data**: exercise_id, name, description, image/video/csv URLs

### `backend/utils.py` — Helpers
- `is_exercise_assigned_to_user(db, user_id, exercise_id)` — kiểm tra quyền

### `backend/hashing.py` — Password Hashing
- Bcrypt hashing cho user passwords.

### `backend/diagnostic_check.py` — Model Health Check
- **Chức năng**: Script kiểm tra model có bị collapse không.

### `backend/test_deep_diagnostic.py` — Deep Diagnostic
- **Chức năng**: Test tất cả 5 exercises với 3 test cases (zeros, standing, arm_raise).
- **⚠ ĐÂY LÀ TOOL QUAN TRỌNG NHẤT** để verify model hoạt động đúng.

### `backend/test_pipeline_gap.py` — Pipeline Gap Test
- **Chức năng**: Simulate frontend CSV → backend pipeline, so sánh với diagnostic.

---

## 5. Model Artifacts

### `models/` Directory

| File | Mô tả | Size |
|------|--------|------|
| `ml_model_Es1.keras` | LSTM v6 cho Es1 (6 features, 150 timesteps) | ~147KB |
| `ml_model_Es2.keras` | LSTM v6 cho Es2 (6 features, 150 timesteps) | ~147KB |
| `ml_model_Es3.keras` | LSTM v5 cho Es3 (**CẦN RETRAIN**) | ~154KB |
| `ml_model_Es4.keras` | LSTM v5 cho Es4 (**CẦN RETRAIN**) | ~156KB |
| `ml_model_Es5.keras` | LSTM v5 cho Es5 (**CẦN RETRAIN**) | ~160KB |
| `scaler_Es1.joblib` | StandardScaler cho Es1 (6 features) | ~759B |
| `scaler_Es2.joblib` | StandardScaler cho Es2 (6 features) | ~759B |
| `scaler_Es3.joblib` | StandardScaler cho Es3 (9 features) | ~815B |
| `scaler_Es4.joblib` | StandardScaler cho Es4 (2 features) | ~663B |
| `scaler_Es5.joblib` | StandardScaler cho Es5 (7 features) | ~767B |

> ⚠ **Es3, Es4, Es5 đang dùng model v5 cũ → bị mean collapse. Cần retrain với v6.**

### `scripts/` Directory

| File | Mô tả |
|------|--------|
| `export_scalers.py` | Export scalers từ training environment |
| `plot_results.py` | Vẽ biểu đồ kết quả training |
| `test_movenet_models_colab.py` | Test MoveNet models trên Colab |

---

## 6. Consistency Checklist

Pipeline chỉ hoạt động đúng khi TẤT CẢ các component đồng bộ:

### ✅ Đã xác nhận khớp

| Component | Training | Inference | Status |
|-----------|----------|-----------|--------|
| Feature functions | `03_extract_joint_features.py` | `joint_features.py` | ✅ Identical |
| Aspect ratio | `x * 1.7778` | `x * 1.7778` | ✅ |
| MASK_VALUE | `-999.0` | `-999.0` | ✅ |
| Downsample stride | `5` | `5` | ✅ |
| Scaler type | `StandardScaler` | `joblib.load()` | ✅ |
| Y normalization | `y / 50.0` (train) | `raw × 50.0` (inference) | ✅ |
| Score range | `[0, 50] → [0, 100]` | `clip(0,50) × 2.0` | ✅ |
| Model output | sigmoid [0, 1] | auto-detect ≤ 1.5 → × 50 | ✅ |

### ⚠ Cần lưu ý

| Issue | Chi tiết |
|-------|----------|
| Column format mismatch | Frontend gửi `_confidence`, backend expect `_z` + `_v`. Features chỉ dùng `_x`, `_y` nên **không ảnh hưởng**. |
| Es3/Es4/Es5 models | Chưa retrain v6 → mean collapse |
| Motion calibration | `calibrate_score()` hiện **KHÔNG được sử dụng** — `calibrated_score = raw_score` |

---

## 7. Known Issues & Fixes

### 🔴 BUG: Frontend trả Infinity cho tất cả keypoints (ĐÃ SỬA)

- **Nguyên nhân**: `keypointsToNormalizedKeypoints(keypoints, video)` chia cho `video.width` — nếu video element chưa set width attribute → `width = 0` → `Infinity`
- **Triệu chứng**: Mọi score luôn ~46-47 (model predict mean vì features toàn NaN/zero)
- **Fix**: Dùng `{ width: video.videoWidth, height: video.videoHeight }` thay vì `video`
- **File**: `frontend/src/pages/Exercise/Exercise.jsx`, line 133

### 🟡 Es3, Es4, Es5 bị Mean Collapse

- **Nguyên nhân**: Chưa retrain với kiến trúc v6 (sigmoid output)
- **Fix**: Retrain trên Colab với `clinical_score_prediction_model.py`
- **Verify**: Chạy `python backend/test_deep_diagnostic.py` → range > 5

### 🟢 WebSocket Connection Closed Error

- **Triệu chứng**: `ConnectionClosedOK: received 1005` trong logs
- **Nguyên nhân**: Frontend disconnect WebSocket khi exercise kết thúc, nhưng backend vẫn đang gửi response
- **Impact**: Cosmetic — không ảnh hưởng scoring
- **Fix tiềm năng**: Wrap `websocket.send_json()` trong try/except `WebSocketDisconnect`

---

## Docker Configuration

### `docker-compose.yml`

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models    # Model artifacts
      - ./backend:/app          # Source code (hot-reload)
      - backend-data:/app/data  # SQLite persistence
    environment:
      - PYTHONUNBUFFERED=1

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    volumes:
      - ./frontend/src:/app/src # Source code (Vite hot-reload)
```

> **Lưu ý**: `./backend:/app` mount toàn bộ source code → thay đổi `.py` files có hiệu lực ngay sau restart (không cần rebuild).

---

## Quick Commands

```bash
# Start
docker compose up -d

# Restart backend (sau khi sửa code Python)
docker compose restart backend

# Xem logs
docker compose logs backend --tail=50

# Test model health
docker compose exec backend python test_deep_diagnostic.py

# Full rebuild (sau khi thay đổi Dockerfile hoặc requirements.txt)
docker compose down && docker compose build && docker compose up -d
```
