# Deep Analysis: Keras Models vs Backend Compatibility

Date: 2026-04-25
Scope: Khả năng dùng cac model trong `models/` de cham diem bai tap qua endpoint `/api/clinical_score/{exercise_id}`.

## 1) Executive Summary

Ket luan: **Chua san sang de cham diem trong production**.

Hai blocker nghiem trong:
- **Blocker A (version mismatch):** Toan bo model `.keras` hien tai **khong load duoc** trong runtime backend Docker (TensorFlow 2.15.0).
- **Blocker B (input mismatch):** Pipeline preprocess `prepare_data()` hien tai tao shape dau vao **khong trung** voi input shape model (ca 5 bai Es1-Es5).

Do do, ngay ca khi endpoint ton tai, backend hien tai khong du dieu kien de infernce dung cho clinical score.

## 2) Nguon Kiem Chung

Da doi chieu code va runtime theo cac nhom sau:

1. Runtime backend:
- `backend/Dockerfile` (Python 3.11)
- `backend/requirements.txt` (tensorflow==2.15.0, keras==2.15.0)

2. Model files:
- `models/ml_model_Es1_best.keras`
- `models/ml_model_Es2_best.keras`
- `models/ml_model_Es3_best.keras`
- `models/ml_model_Es4_best.keras`
- `models/ml_model_Es5_best.keras`

3. Inference pipeline backend:
- `backend/main.py` (startup model loading + endpoint clinical score)
- `backend/ml_wrapper.py` (`prepare_data`, `reorder_dataframe`)

4. Training/data-prep pipeline:
- `training_models/clinical_score_prediction_model.py`
- `data-preparation/03_extract_joint_features.py`

## 3) Findings (Ordered by Severity)

### F1 - CRITICAL: Model `.keras` khong tuong thich voi runtime backend hien tai

**Observed**
- Runtime backend container: Python 3.11.15, TensorFlow 2.15.0.
- Thu load truc tiep trong container (`tf.keras.models.load_model(..., compile=False)`) cho ca 5 model deu fail.

**Error signature**
- `TypeError: Error when deserializing class 'InputLayer' ... Unrecognized keyword arguments: ['batch_shape', 'optional']`

**Additional evidence from metadata**
- Metadata ben trong model `.keras` cho thay `keras_version = 3.13.2`.
- `InputLayer` config co key `optional` (thuong gap voi Keras 3 serialization).

**Impact**
- Tai startup, `ml_models_cache` trong backend se khong co model hop le cho Es1-Es5.
- Endpoint `/api/clinical_score/{exercise_id}` co nguy co tra `503` (`Model failed to load...`) cho toan bo bai tap.

---

### F2 - CRITICAL: Input shape mismatch giua preprocess backend va model

Da extract input shape model tu `config.json` cua tung file `.keras`:

- Es1: `[None, 301, 9]`
- Es2: `[None, 326, 9]`
- Es3: `[None, 297, 13]`
- Es4: `[None, 398, 6]`
- Es5: `[None, 204, 9]`

Da chay `prepare_data()` trong backend container voi input gia lap 600 frames:

- Es1 -> `(1, 781, 9)`
- Es2 -> `(1, 326, 36)`
- Es3 -> `(1, 297, 36)`
- Es4 -> `(1, 398, 36)`
- Es5 -> `(1, 204, 36)`

**Mismatch chi tiet**
- Es1: sai `timesteps` (`781` vs `301`).
- Es2-Es5: sai `features` (`36` vs `9/13/6/9`).

**Impact**
- Ngay ca khi fix xong loi load model, prediction van co the fail do shape mismatch (`ValueError`/graph input mismatch).

---

### F3 - HIGH: Drift preprocess giua training va inference

**Training side (clinical_score_prediction_model + data-preparation)**
- Model duoc train tren file `joint_features` theo feature engineering rieng cho tung exercise:
  - Es1: 9 features
  - Es2: 9 features
  - Es3: 13 features
  - Es4: 6 features
  - Es5: 9 features

**Inference side (backend/ml_wrapper.py)**
- Es1: dung `get_es1_features` (9 features) nhung xu ly length/downsampling khong nhat quan.
- Es2-Es5: khong dung feature engineering tuong ung; thay vao do dung raw joint positions da drop mot phan (36 features).

**Impact**
- Day la drift nghiem trong giua train va infer.
- Neu co fix shape tam thoi ma khong dong bo preprocessing, score se thieu do tin cay lam sang.

---

### F4 - MEDIUM: Runtime khong dong nhat giua local va Docker

**Observed**
- Local environment hien tai: Python 3.13 + TensorFlow 2.21 + Keras 3.13.
- Backend Docker: Python 3.11 + TensorFlow/Keras 2.15.

**Impact**
- Co nguy co "chay duoc o may nay, fail o may khac" neu bypass Docker.
- Lam kho truy vet bug va khong dam bao reproducibility.

## 4) Root Cause Tom Tat

1. Models duoc save bang stack Keras 3.13, nhung backend runtime dang pin o TF/Keras 2.15.
2. Inference preprocessing (`prepare_data`) khong match voi training preprocessing cho Es2-Es5, va logic length Es1 co van de.
3. Chua co bo kiem tra compatibility tu dong ngay startup (load + shape smoke test + sample predict).

## 5) Readiness Assessment

Trang thai hien tai: **NOT READY** cho clinical scoring.

Muc do san sang:
- Model load compatibility: **FAIL**
- Input pipeline compatibility: **FAIL**
- End-to-end scoring reliability: **FAIL**

## 6) De Xuat Huong Xu Ly

### Huong 1 (uu tien): Dong bo backend len stack Keras 3

- Nang runtime backend len stack co the deserialize model hien tai (Keras 3-compatible TensorFlow stack).
- Re-test toan bo startup load model.

### Huong 2: Re-export/retrain model ve stack tuong thich TF 2.15

- Export lai model tu moi truong train sao cho load duoc boi TF 2.15.
- Dam bao format va config InputLayer khong dung kwargs khong duoc ho tro.

### Bat buoc cho ca 2 huong

- Refactor `backend/ml_wrapper.py` de preprocess dung theo training:
  - Implement feature engineering cho Es2-Es5 tuong ung `03_extract_joint_features.py`.
  - Dong bo downsampling/smoothing/padding/truncation dung logic train.
  - Bo sung shape assertions truoc khi predict.
- Them startup compatibility checks:
  - Load all models.
  - Verify `prepare_data(...).shape[1:] == model.input_shape[1:]` cho tung exercise.
  - Chay sample predict smoke test.

## 7) Verification Checklist Sau Khi Fix

1. `docker compose up backend` khong co `model_load_errors`.
2. `/api/clinical_score/Es1..Es5` tra ket qua hop le (khong 4xx/5xx do model).
3. Shape check pass cho tung exercise.
4. So sanh score giua backend va notebook benchmark tren cung CSV mau.
5. Them test tu dong cho regression compatibility (version + shape + predict).
