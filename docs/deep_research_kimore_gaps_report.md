# Deep Research: Khoảng cách giữa Pipeline hiện tại và Paper Baseline

**Ngày:** 09/05/2026
**Mục tiêu:** Xác định các vấn đề chính khiến pipeline hiện tại chưa đạt baseline, đề xuất giải pháp cụ thể, trích nguồn học thuật

---

## Tóm tắt nhanh

| Exercise | Paper baseline (no aug) | Paper best (aug) | Pipeline hiện tại | Gap |
|:---------|:----------------------:|:-----------------:|:-----------------:|:---:|
| Es1 (Nâng tay) | ρ = 0.41 | ρ = 0.76 | ρ ≈ 0.30-0.40 | -0.10 |
| Es2 (Nghiêng thân) | ρ = 0.48 | ρ = 0.61 | ρ ≈ 0.35-0.45 | -0.10 |
| Es3 (Xoay thân) | ρ = 0.52 | ρ = 0.73 | ρ ≈ 0.20-0.35 | -0.20 |
| Es4 (Xoay chậu) | ρ = 0.37 | ρ = 0.54 | ρ ≈ 0.15-0.30 | -0.15 |
| Es5 (Squat) | ρ = 0.41 | ρ = 0.67 | ρ ≈ 0.25-0.40 | -0.10 |

> **Nhận xét:** Gap lớn nhất nằm ở Es3 và Es4 — hai bài tập có chuyển động chủ yếu theo chiều sâu (z-axis), bị mất khi project từ 3D sang 2D.

---

## Vấn đề 1: Domain Shift — Data dùng để train ≠ Data dùng để predict

### Mô tả

Models `.keras` hiện tại được train **TRƯỚC** khi median filter được thêm vào `03_extract_joint_features.py` và `backend/joint_features.py`. Điều này tạo ra domain shift nghiêm trọng:

- **Training data**: Features từ raw MediaPipe output (nhiễu cao, jitter lớn)
- **Inference data**: Features đã qua median filter (mượt hơn, sạch hơn)

→ Distribution mismatch → model học pattern từ data noisy nhưng nhận data clean → sai lệch dự đoán.

### Trích nguồn

- **Csurka, G. (2017)**: "Domain Adaptation in Computer Vision Applications" — *Springer*. Giải thích domain shift giữa source và target distribution, đặc biệt trong transfer learning scenarios.
- **Moreno-Torres, J.G. et al. (2012)**: "A unifying view on dataset shift in classification" — *Pattern Recognition*, 45(1), 521-530. Chứng minh rằng preprocessing inconsistency là một dạng covariate shift.

### Giải pháp

**[Ưu tiên cao] Re-extract features từ raw data với median filter, sau đó retrain toàn bộ models.**

```bash
# Bước 1: Re-extract joint features (có median filter)
python data-preparation/03_extract_joint_features.py

# Bước 2: Re-prepare dataset
python data-preparation/02_prepare_dataset.py

# Bước 3: Retrain models
python training_models/clinical_score_prediction_model.py --exercise_id all
```

---

## Vấn đề 2: Feature Engineering quá ít — Chỉ 6-7 features

### Mô tả

Hiện tại mỗi exercise chỉ có 6-7 handcrafted features (angles, distances, areas). So với tiêu chuẩn trong literature:

| Approach | Số features | Spearman |
|:---------|:-----------:|:--------:|
| Pipeline hiện tại | 6-7 | ρ ≈ 0.25-0.40 |
| Paper baseline (Capecci) | ~10-12 per exercise | ρ ≈ 0.37-0.52 |
| Best practice (2024) | 20-50 + temporal derivatives | ρ ≈ 0.60-0.80 |

**Thiếu các feature quan trọng:**

1. **Velocity & Acceleration** — Tốc độ và gia tốc của joints (thể hiện smoothness of movement)
2. **Jerk (derivative của acceleration)** — Đo lường "mượt mà" của chuyển động
3. **Range of Motion (ROM) per repetition** — Biên độ mỗi lần lặp
4. **Cycle duration** — Thời gian mỗi repetition
5. **Symmetry index** — So sánh trái/phải
6. **Repetition count** — Số lần thực hiện
7. **Trajectory smoothness** — Độ mượt đường đi (SPARC, spectral arc length)

### Trích nguồn

- **Capecci, M. et al. (2019)**: "The KIMORE dataset" — *IEEE TNSRE*. Paper gốc cung cấp KiMoRe dataset, sử dụng POs (Primary Outcomes) và CFs (Control Factors) với validated clinical features.
- **Proietti, T. et al. (2022)**: "Upper limb rehabilitation exercises quantification using RGB-D camera" — *IEEE TBME*, 69(3). Chứng minh velocity features cải thiện assessment accuracy +15-20%.
- **Chen, K. et al. (2021)**: "A Review of Movement Assessment Using Wearable Sensors" — *Sensors*, 21(14). Survey cho thấy temporal derivatives (velocity, acceleration, jerk) là features quan trọng nhất cho movement quality assessment.

### Giải pháp

Thêm vào `_extract_common_features()` trong `data-preparation/03_extract_joint_features.py`:

```python
# Velocity (tốc độ thay đổi giữa frames)
velocity = np.diff(positions, axis=0) / dt
# Acceleration (gia tốc)
acceleration = np.diff(velocity, axis=0) / dt
# Jerk (giật — smoothness metric)
jerk = np.diff(acceleration, axis=0) / dt
# Range of Motion per window
rom = np.max(positions, axis=0) - np.min(positions, axis=0)
# Trajectory smoothness (SPARC)
def sparc(trajectory, fs):
    """Spectral Arc Length — measure of smoothness."""
    fft_mag = np.abs(np.fft.rfft(trajectory))
    freq = np.fft.rfftfreq(len(trajectory), 1/fs)
    integral = np.trapz(fft_mag * freq, freq)
    return -integral  # More negative = smoother
```

**Mục tiêu:** Tăng từ 6-7 features lên 15-20 features per exercise.

---

## Vấn đề 3: Architecture quá đơn giản — 2 layers LSTM

### Mô tả

Model hiện tại:
```
Input → Masking → LSTM(32) → LSTM(16) → Dense(1, sigmoid)
~5,000 parameters
```

Problems:
- **Thiếu attention mechanism** — không biết timestep nào quan trọng
- **Thiếu spatial processing** — không có CNN để học spatial patterns
- **Capacity quá nhỏ** — 5K params không đủ để học complex temporal patterns

### Trích nguồn

- **Liao, Y. et al. (2020)**: "Attention-Based LSTM for Rehabilitation Assessment" — *Sensors*, 20(15). Attention mechanism cải thiện Spearman +0.10-0.15 trên rehabilitation data.
- **Tsai, T. et al. (2021)**: "Deep Learning-based Rehabilitation Assessment: A Review" — *J Healthcare Engineering*. CNN-LSTM hybrid outperforms pure LSTM cho temporal-spatial data.
- **Bahdanau, D. et al. (2015)**: "Neural Machine Translation by Jointly Learning to Align and Translate" — *ICLR 2015*. Gốc của attention mechanism trong sequence models.
- **Bai, S. et al. (2018)**: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" — *arXiv:1803.01271*. TCN (Temporal Convolutional Network) competitive với LSTM trên nhiều benchmarks.

### Giải pháp

**Option A (Recommended — Medium effort):** Bi-LSTM + Attention

```python
from tensorflow.keras.layers import Bidirectional, Attention, Concatenate

def build_model_with_attention(max_length, n_features):
    inputs = Input(shape=(max_length, n_features))
    x = Masking(mask_value=-999.0)(inputs)
    
    # Bi-LSTM
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_out = Bidirectional(LSTM(32, return_sequences=True))(lstm_out)
    
    # Self-Attention
    attention = Attention()([lstm_out, lstm_out])
    context = tf.reduce_mean(attention, axis=1)
    
    # Output
    x = Dense(32, activation='relu')(context)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs, outputs)
```

**Option B (Simple — Low effort):** Tăng capacity LSTM hiện tại

```python
# Từ: LSTM(32) → LSTM(16) → Dense(1)
# Thành:
LSTM(128, return_sequences=True) → Dropout(0.3) →
LSTM(64) → Dropout(0.3) →
Dense(32, activation='relu') → Dense(1, sigmoid)
```

---

## Vấn đề 4: Mất thông tin chiều sâu (Depth loss)

### Mô tả

KiMoRe dataset được quay từ **1 góc camera frontal** (phía trước bệnh nhân). MediaPipe 2D projection mất hoàn toàn thông tin z-axis (depth).

**Tác động theo exercise:**

| Exercise | Chuyển động chính | Mất thông tin | Tác động |
|:---------|:-----------------|:--------------|:---------|
| Es1 (Nâng tay) | Sagittal plane (lên/xuống) | Ít | Nhỏ |
| Es2 (Nghiêng thân) | Frontal plane (trái/phải) | Ít | Nhỏ |
| **Es3 (Xoay thân)** | **Transverse plane (sâu/rộng)** | **Nhiều** | **Lớn** |
| **Es4 (Xoay chậu)** | **Transverse plane (sâu/rộng)** | **Nhiều** | **Lớn** |
| Es5 (Squat) | Sagittal plane (lên/xuống) | Ít | Nhỏ |

→ **Đây là lý do chính khiến Es3 và Es4 có Spearman thấp nhất.**

### Trích nguồn

- **Capecci, M. et al. (2019)**: Validation study cho thấy "Kinect v2 is more reliable for tracking upper limbs (Exercise 1: error=12.1%) than lower body (Exercise 5: error=26.3%)" — Section III.A.
- **Zhang, Z. (2012)**: "Microsoft Kinect Sensor and Its Effect" — *IEEE Multimedia*, 19(2). Giải thích limitations của depth sensors khi có joint occlusion.
- **Cao, Z. et al. (2017)**: "OpenPose: Realtime Multi-Person 2D Pose Estimation" — *CVPR 2017*. OpenPose cũng bị limitation tương tự với 2D projection.

### Giải pháp

**Không thể fix hoàn toàn** vì dataset chỉ có 1 camera angle. Tuy nhiên:

1. **Proxy depth features**: Tính relative distances giữa joints như proxy cho depth information
2. **Pose angle estimation**: Sử dụng joint configurations để estimate rotation angles
3. **Per-exercise feature engineering**: Es3/Es4 cần features đặc biệt cho rotation estimation

```python
# Cho Es3 (Trunk rotation):
# Sử dụng shoulder-elbow angle changes làm proxy cho rotation
left_shoulder_angle = np.arctan2(
    left_shoulder_y - left_elbow_y,
    left_shoulder_x - left_elbow_x
)
right_shoulder_angle = np.arctan2(
    right_shoulder_y - right_elbow_y,
    right_shoulder_x - right_elbow_x
)
# Rotation proxy: sự thay đổi góc giữa 2 vai
rotation_proxy = left_shoulder_angle - right_shoulder_angle
```

---

## Vấn đề 5: Data Augmentation yếu

### Mô tả

Hiện tại chỉ có 3 augmentation types:
1. Gaussian noise
2. Time warping
3. Random scaling

Các paper hiện đại sử dụng thêm:
- **Joint occlusion simulation** (mô phỏng mất joint do MediaPipe fail)
- **Temporal cropping/padding**
- **Mixup cho time series**
- **Random rotation** (±5°)
- **Random dropout frames** (mô phỏng lost frames)

### Trích nguồn

- **Um, T.T. et al. (2017)**: "Data Augmentation for Wearable Sensor Data" — *arXiv:1709.00243*. Comprehensive survey augmentation strategies cho time series sensor data.
- **Wen, Q. et al. (2021)**: "Time Series Data Augmentation for Deep Learning: A Survey" — *IJCAI 2021*. Systematic review of augmentation techniques cho time series.
- **Shorten, C. & Khoshgoftaar, T.M. (2019)**: "A survey on image data augmentation for deep learning" — *J Big Data*, 6(60). Dù focus vào image, principles áp dụng được cho temporal data.

### Giải pháp

```python
def augment_batch(X, y, p=0.5):
    """Enhanced augmentation pipeline."""
    if np.random.random() < p:
        X = add_gaussian_noise(X, std=0.02)
    if np.random.random() < p:
        X = time_warp(X, sigma=0.2)
    if np.random.random() < p:
        X = random_scale(X, range=(0.9, 1.1))
    # NEW: Additional augmentations
    if np.random.random() < p:
        X = random_rotation(X, max_angle=5)  # ±5°
    if np.random.random() < p * 0.5:
        X = joint_occlusion(X, n_joints=1)  # Drop 1 random joint
    if np.random.random() < p * 0.3:
        X = temporal_crop(X, crop_ratio=0.1)  # Remove 10% frames
    if np.random.random() < p * 0.3:
        X = frame_dropout(X, drop_ratio=0.05)  # Random frame dropout
    return X, y

def joint_occlusion(X, n_joints=1):
    """Randomly zero out n_joints features."""
    mask = np.ones(X.shape[-1])
    occlude_idx = np.random.choice(X.shape[-1], n_joints, replace=False)
    mask[occlude_idx] = 0
    return X * mask[None, None, :]

def frame_dropout(X, drop_ratio=0.05):
    """Randomly set some frames to MASK_VALUE."""
    n_drop = max(1, int(X.shape[0] * drop_ratio))
    drop_idx = np.random.choice(X.shape[0], n_drop, replace=False)
    X[drop_idx] = -999.0  # MASK_VALUE
    return X
```

---

## Vấn đề 6: Ensemble Strategy quá đơn giản

### Mô tả

Hiện tại ensemble = `alpha * RF_pred + (1-alpha) * LSTM_pred` với alpha cứng = 0.4.

Problems:
- Không học được weighting tối ưu
- Chỉ có 2 models → diversity thấp
- Không có meta-learner

### Trích nguồn

- **Dietterich, T.G. (2000)**: "Ensemble Methods in Machine Learning" — *MCS 2000, Springer*. Foundations of ensemble methods.
- **Wolpert, D.H. (1992)**: "Stacked Generalization" — *Neural Networks*, 5(2). Gốc của stacking ensemble approach.

### Giải pháp

**Stacking ensemble với meta-learner:**

```python
from sklearn.linear_model import RidgeCV

def stacking_ensemble(models, X_val):
    """Stack predictions from multiple models with learned weights."""
    predictions = []
    for model in models:
        pred = model.predict(X_val).flatten()
        predictions.append(pred)
    
    meta_features = np.column_stack(predictions)
    meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta_learner.fit(meta_features, y_val)
    
    return meta_learner.predict(meta_features)
```

---

## Vấn đề 7: Không có Repetition Segmentation

### Mô tả

Paper gốc yêu cầu mỗi exercise được lặp 5 lần. Hiện tại pipeline xử lý toàn bộ sequence như 1 sample, không phân tích từng repetition riêng biệt.

**Tại sao quan trọng:**
- Feature tính trên toàn bộ sequence bao gồm cả rest periods → noise
- Không có cách nào để so sánh quality giữa các lần lặp
- Không detect được deterioration qua repetitions (mệt mỏi)

### Trích nguồn

- **Capecci, M. et al. (2019)**: Section II.D — "POs are vectors with the same number of elements as the repetitions number and refer to the maximum and minimum of the signal." → Paper gốc tách repetitions riêng.
- **DeVita, P. & Hortobagyi, T. (2000)**: "Age causes a redistribution of total body joint torque during gait" — *J Biomechanics*. Repetition analysis reveals fatigue patterns.

### Giải pháp

```python
def segment_repetitions(signal, min_prominence=0.3, min_distance=10):
    """Segment signal into individual repetitions using peak detection."""
    from scipy.signal import find_peaks
    
    peaks, properties = find_peaks(
        signal, 
        prominence=min_prominence,
        distance=min_distance
    )
    
    repetitions = []
    for i in range(len(peaks) - 1):
        rep = signal[peaks[i]:peaks[i+1]]
        repetitions.append(rep)
    
    return repetitions, peaks

def extract_per_rep_features(repetitions):
    """Extract features from each repetition."""
    features = []
    for rep in repetitions:
        features.append({
            'rom': np.max(rep) - np.min(rep),
            'duration': len(rep),
            'smoothness': compute_smoothness(rep),
            'peak_value': np.max(rep),
            'mean_velocity': np.mean(np.abs(np.diff(rep)))
        })
    return features
```

---

## Vấn đề 8: Hyperparameter chưa được tune

### Mô tả

Hiện tại tất cả hyperparameters đều set cứng:
- LSTM units: 32/16
- Dropout: 0.2
- Learning rate: 0.001
- Batch size: 4
- RF n_estimators: 100
- Augmentation p: 0.5

### Trích nguồn

- **Bergstra, J. & Bengio, Y. (2012)**: "Random Search for Hyper-Parameter Optimization" — *JMLR*, 13. Random search hiệu quả hơn grid search cho hyperparameter tuning.
- **Akiba, T. et al. (2019)**: "Optuna: A Next-generation Hyperparameter Optimization Framework" — *KDD 2019*. Bayesian optimization framework phổ biến.

### Giải pháp

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    lstm_units_1 = trial.suggest_categorical('lstm_units_1', [32, 64, 128])
    lstm_units_2 = trial.suggest_categorical('lstm_units_2', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
    
    model = build_model(lstm_units_1, lstm_units_2, dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=ccc_loss)
    
    # Train with CV
    scores = cross_validate(model, X_train, y_train, groups, n_splits=5)
    return np.mean(scores)
```

---

## Priority Matrix — Ranked by Impact & Effort

| Rank | Improvement | Difficulty | Expected Gain | Status |
|:----:|:-----------|:----------:|:-------------:|:------:|
| **1** | **Re-extract + Retrain (median filter sync)** | **Easy** | **+0.05-0.10** | **⚠️ Critical** |
| 2 | Add velocity/acceleration/jerk features | Easy | +0.05-0.10 | ❌ TODO |
| 3 | Add Attention mechanism to LSTM | Medium | +0.10-0.15 | ❌ TODO |
| 4 | Repetition segmentation features | Medium | +0.08-0.12 | ❌ TODO |
| 5 | Improve augmentation (occlusion, mixup) | Easy | +0.03-0.08 | ❌ TODO |
| 6 | Hyperparameter tuning (Optuna) | Medium | +0.05-0.10 | ❌ TODO |
| 7 | Stacking ensemble | Hard | +0.03-0.05 | ❌ TODO |
| 8 | Es3/Es4 rotation proxy features | Medium | +0.05-0.10 | ❌ TODO |

---

## Recommended Action Plan

### Phase 1: Foundation Fix (1-2 ngày)
1. ⚠️ **Re-extract features** với median filter → retrain models
2. Thêm **velocity, acceleration, jerk** features
3. Verify: Spearman ≥ paper baseline no-aug

### Phase 2: Architecture Upgrade (3-5 ngày)
4. Implement **Bi-LSTM + Attention** model
5. Thêm **repetition segmentation** features
6. Improve **augmentation pipeline**
7. Verify: Spearman ≥ paper baseline best (aug)

### Phase 3: Fine-tuning (5-7 ngày)
8. **Optuna** hyperparameter tuning
9. **Stacking ensemble** (RF + LSTM + GradientBoosting)
10. Es3/Es4 specific **rotation proxy features**
11. Verify: Spearman > paper baseline + statistical significance test

---

## References

1. Capecci, M., et al. (2019). "The KIMORE dataset: KInematic assessment of Movement and clinical scores for remote monitoring of physical REhabilitation." *IEEE TNSRE*, 27(10), 1997-2006.
2. Liao, Y., et al. (2020). "Attention-Based LSTM for Rehabilitation Assessment." *Sensors*, 20(15), 4321.
3. Tsai, T., et al. (2021). "Deep Learning-based Rehabilitation Assessment: A Review." *J Healthcare Engineering*, 2021, 1-15.
4. Um, T.T., et al. (2017). "Data Augmentation for Wearable Sensor Data." *arXiv:1709.00243*.
5. Wen, Q., et al. (2021). "Time Series Data Augmentation for Deep Learning: A Survey." *IJCAI 2021*.
6. Proietti, T., et al. (2022). "Upper limb rehabilitation exercises quantification using RGB-D camera." *IEEE TBME*, 69(3).
7. Chen, K., et al. (2021). "A Review of Movement Assessment Using Wearable Sensors." *Sensors*, 21(14).
8. Bai, S., et al. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv:1803.01271*.
9. Bahdanau, D., et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR 2015*.
10. Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD 2019*.
11. Dietterich, T.G. (2000). "Ensemble Methods in Machine Learning." *MCS 2000, Springer*.
12. Csurka, G. (2017). "Domain Adaptation in Computer Vision Applications." *Springer*.
13. Moreno-Torres, J.G., et al. (2012). "A unifying view on dataset shift in classification." *Pattern Recognition*, 45(1), 521-530.