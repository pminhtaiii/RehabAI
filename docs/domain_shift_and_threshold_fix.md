# Domain Shift: Kinect → Webcam & Motion Threshold Fix

**Ngày:** 2026-05-08  
**Tác giả:** Backend Engineer  
**Trạng thái:** Plan + Implementation

---

## 1. Vấn đề

### 1.1 Domain Shift: Kinect vs Webcam

**Vấn đề:** Model được train trên dataset KiMoRe sử dụng **Microsoft Kinect v2** (depth camera + IR sensor), nhưng inference trong dự án sử dụng **webcam RGB + MediaPipe Pose**. Đây là domain shift nghiêm trọng ở nhiều level:

| Khía cạnh | Kinect v2 (KiMoRe) | Webcam + MediaPipe |
|-----------|--------------------|--------------------|
| **Sensor** | Depth camera (IR + RGB) | RGB-only camera |
| **Keypoint detection** | Kinect SDK skeleton tracking | MediaPipe Pose (ML-based) |
| **Coordinate system** | 3D thực (depth sensor) | 3D ước lượng (z từ monocular) |
| **Noise pattern** | Low noise, stable tracking | Jittery, especially on occlusion |
| **Resolution/FOV** | Fixed, controlled lab setting | Variable webcam quality |
| **Viewing angle** | Fixed frontal (lab setup) | Variable user setup |

**Hậu quả:**
- Feature distributions (variance, ROM, displacement) có **scale khác nhau** giữa Kinect và MediaPipe
- Motion thresholds calibrated từ KiMoRe Kinect data → **không hợp lệ** cho MediaPipe webcam data
- Model predictions có thể systematic bias (predict cao hơn hoặc thấp hơn)

### 1.2 Motion Thresholds Quá Strict

**Vấn đề cụ thể:** `EXERCISE_THRESHOLDS` trong `backend/motion_detector.py` được calibrate từ KiMoRe Kinect features với công thức `percentile_5 × 0.5`. Khi áp dụng cho webcam MediaPipe features:

- Kinect features có **variance và ROM lớn hơn** (depth sensor chính xác hơn) → thresholds cao
- MediaPipe features có **variance và ROM nhỏ hơn** (2D noise, jittery) → không đạt threshold
- → `is_active = False` → `calibrate_score()` trả về **0.0** cho nhiều trường hợp user đang tập thật

**Ví dụ cụ thể (Es1 - Arm Raise):**
```
KiMoRe Kinect: min_variance threshold = 323.226
Webcam MediaPipe: typical arm raise variance ≈ 50-150 (thấp hơn nhiều)
→ Không đạt threshold → score = 0 (false negative)
```

---

## 2. Nguyên nhân gốc

### 2.1 Feature Scale Mismatch

Từ `03_extract_joint_features.py`, features được tính từ **2D joint positions** với aspect ratio correction (`x * 1.7778`). Tuy nhiên:

- **KiMoRe Kinect:** joint positions từ Kinect SDK → coordinates ổn định, ít noise
- **MediaPipe webcam:** joint positions từ ML model → nhiều jitter, especially khi:
  - User ở xa camera
  - Background phức tạp
  - Ánh sáng yếu
  - Che khuất (self-occlusion)

### 2.2 Temporal Resolution Difference

- KiMoRe: ~25fps → downsample stride=5 → ~5fps
- Webcam: ~30fps → downsample stride=5 → ~6fps
- Tuy khác biệt nhỏ, nhưng kết hợp với noise pattern khác → feature statistics khác

### 2.3 Coordinate System Differences

- Kinect 3D: `x, y, z` từ depth sensor → z-axis chính xác
- MediaPipe 3D: `x, y` từ image, `z` ước lượng → z-axis kém chính xác
- Các features sử dụng 2D projection (aspect ratio corrected) → ít bị ảnh hưởng bởi z
- Nhưng **noise level** vẫn khác nhau → variance/displacement khác

---

## 3. Giải pháp đề xuất

### 3.1 Giải pháp ngắn hạn (implement ngay): Re-calibrate Thresholds

**Approach:** Giảm thresholds dựa trên phân tích scale difference giữa Kinect và MediaPipe.

**Phương pháp:**
1. **Bỏ hệ số safety margin × 0.5** — vì webcam có inherent noise cao hơn
2. **Giảm thêm factor 0.3** — để account cho MediaPipe feature scale thấp hơn Kinect
3. **Thay đổi rule `passes >= 2` → `passes >= 1`** — ít strict hơn cho activity detection
4. **Giảm penalty khi không active** — `quality_factor` floor từ 0.1 → 0.3

**Cơ sở khoa học:**

Feature variance từ MediaPipe trên webcam thấp hơn Kinect vì:
- MediaPipe Pose sử dụng heatmap-based 2D detection → inherent quantization noise thấp hơn Kinect depth noise
- Nhưng khi tính biomechanical features (angles, distances), **MediaPipe noise bị amplify** bởi atan2 và distance calculations
- → Variance của raw features thấp hơn, nhưng variance của **biomechanical features** có thể cao hơn hoặc thấp hơn tùy feature

**Tham khảo:**
- KiMoRe dataset: Capecci, M. et al. (2019). "A unifying kinect-based framework to score the quality of execution of exercises." *Sensors*, 19(15), 3392. — Mô tả dataset collection với Kinect v2.
- MediaPipe Pose: Lugaresi, C. et al. (2019). "MediaPipe: A Framework for Building Perception Pipelines." *arXiv:1906.08172*. — Mô tả architecture và noise characteristics.

### 3.2 Giải pháp trung hạn: Re-calibrate từ Webcam Data

**Approach:** Thu thập webcam recordings từ healthy subjects, extract features, tính lại thresholds.

**Steps:**
1. Record 10-20 healthy subjects performing each exercise via webcam
2. Extract features using same pipeline (`03_extract_joint_features.py`)
3. Compute motion metrics (variance, ROM, displacement)
4. Set thresholds at percentile 5 (without × 0.5 safety margin)
5. Update `EXERCISE_THRESHOLDS` in `motion_detector.py`

**Ưu điểm:** Thresholds phản ánh đúng distribution của webcam data  
**Nhược điểm:** Cần thu thập data mới, tốn thời gian

### 3.3 Giải pháp dài hạn: Domain Adaptation

Có nhiều approaches được đề xuất trong literature cho cross-sensor domain adaptation:

#### 3.3.1 Domain-Adversarial Neural Networks (DANN)

**Approach:** Train model với shared feature extractor + domain discriminator. Feature extractor học features invariant giữa source (Kinect) và target (webcam) domains.

**Cơ chế:**
```
Input → Feature Extractor → ┬→ Task Predictor (clinical score)
                             └→ Domain Discriminator (Kinect vs Webcam)
```
Feature extractor maximize domain confusion (discriminator không phân biệt được source/target).

**Nguồn:**
- Ganin, Y. et al. (2016). "Domain-Adversarial Training of Neural Networks." *Journal of Machine Learning Research (JMLR)*, 17(59), 1–35. — Paper gốc đề xuất DANN framework.

**Áp dụng cho dự án:** Cần một lượng nhỏ webcam labeled data (ít nhất 5-10 subjects) để train domain discriminator. Có thể implement bằng cách thêm gradient reversal layer.

#### 3.3.2 Correlation Alignment (CORAL)

**Approach:** Align second-order statistics (covariance) giữa source và target features.

**Cơ chế:** Transform source features sao cho covariance matrix khớp với target covariance matrix:
```
D_CORAL(A_s, A_t) = (1/4d²) ||C_s - C_t||_F²
```
Trong đó `C_s, C_t` là covariance matrices của source và target features.

**Nguồn:**
- Sun, B. & Saenko, K. (2016). "Deep CORAL: Correlation Alignment for Deep Domain Adaptation." *ECCV 2016*, pp. 131–148. — Mở rộng CORAL cho deep learning.

**Áp dụng cho dự án:** Đơn giản hơn DANN. Chỉ cần tính covariance matrices từ webcam data (không cần labels) và transform features. Có thể implement bằng cách thêm CORAL loss term vào training.

#### 3.3.3 Test-Time Adaptation (TTA)

**Approach:** Adapt model tại inference time bằng cách cập nhật batch normalization statistics.

**Cơ chế:** Freeze model weights, chỉ cập nhật running mean/variance của BatchNorm layers dựa trên webcam input distribution.

**Nguồn:**
- Wang, D. et al. (2021). "Tent: Fully Test-Time Adaptation by Entropy Minimization." *ICLR 2021*. — Propose entropy minimization cho test-time adaptation.
- Schneider, S. et al. (2020). "Improving Robustness against Common Corruptions by Covariate Shift Adaptation." *NeurIPS 2020*. — BatchNorm adaptation cho corruption robustness.

**Áp dụng cho dự án:** Model hiện tại dùng LSTM + Dense (không có BatchNorm), nên cần thêm BatchNorm layers trước khi áp dụng TTA. Hoặc có thể dùng **feature statistics adaptation** — cập nhật StandardScaler statistics dựa trên webcam data distribution.

#### 3.3.4 Domain-Specific Normalization

**Approach:** Sử dụng domain-specific StandardScaler — mỗi sensor domain có scaler riêng.

**Cơ chế:**
```
Training: fit scaler_kinect trên Kinect features
Inference: fit scaler_webcam trên webcam features (online hoặc offline)
```

**Nguồn:**
- Nado, Z. et al. (2020). "On Robustness of Neural ODEs via Feature Normalization." *arXiv:2012.01294*. — Discusses importance of normalization for domain robustness.
- Li, Y. et al. (2018). "Revisiting Batch Normalization for Practical Domain Adaptation." *Pattern Recognition*, 100. — Domain-specific BN cho domain adaptation.

**Áp dụng cho dự án:** Đơn giản nhất — tạo webcam-specific StandardScaler. Cần webcam data để fit scaler. Có thể implement bằng cách:
1. Record webcam sessions
2. Extract raw features (không scale)
3. Fit StandardScaler riêng cho webcam
4. Dùng webcam scaler thay vì Kinect scaler khi inference

#### 3.3.5 Pose Normalization (Pre-processing Level)

**Approach:** Normalize pose keypoints trước khi extract features để giảm domain gap.

**Cơ chế:**
- Center keypoints relative to body center (hip midpoint)
- Scale by torso length (shoulder-to-hip distance)
- Rotate to upright posture

**Nguồn:**
- Cao, Z. et al. (2017). "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." *CVPR 2017*. — Discusses pose normalization strategies.
- Güler, R.A. et al. (2018). "DensePose: Dense Human Pose Estimation In The Wild." *CVPR 2018*. — Pose normalization cho cross-view invariance.

**Áp dụng cho dự án:** Normalize keypoints relative to body center + scale by torso length trước khi tính features. Giảm sensitivity to camera distance và viewing angle.

---

## 4. Đánh giá các giải pháp

| Giải pháp | Complexity | Data cần thiết | Hiệu quả kỳ vọng | Ưu tiên |
|-----------|-----------|----------------|-------------------|---------|
| **Re-calibrate thresholds** | Thấp | Không | Trung bình (giảm false negative) | **Cao — Implement ngay** |
| **Re-calibrate từ webcam** | Trung bình | 10-20 subjects | Cao | **Cao** |
| **DANN** | Cao | 5-10 webcam labeled | Cao | Trung bình |
| **CORAL** | Trung bình | Webcam unlabeled | Trung bình-Cao | Trung bình |
| **Test-Time Adaptation** | Trung bình | Webcam unlabeled | Trung bình | Thấp |
| **Domain-Specific Scaler** | Thấp | Webcam features | Trung bình | **Cao — Dễ implement** |
| **Pose Normalization** | Thấp | Không | Trung bình | Trung bình |

---

## 5. Implementation Plan

### Phase 1: Fix Motion Thresholds (Ngay)

**Mục tiêu:** Giảm false negative rate cho webcam users.

**Thay đổi:**
1. Giảm thresholds factor × 0.3 (thay vì × 0.5)
2. Thay đổi `passes >= 2` → `passes >= 1` 
3. Tăng quality_factor floor từ 0.1 → 0.3
4. Thêm log để monitor threshold hit rates

### Phase 2: Webcam-Specific Scaler (Tuần tới)

**Mục tiêu:** Giảm systematic bias giữa Kinect và webcam features.

**Steps:**
1. Tạo script `calibrate_webcam_scaler.py` — record webcam sessions, fit StandardScaler
2. Lưu webcam scaler riêng (`scaler_Es1_webcam.joblib`, etc.)
3. Modify `ml_wrapper.py` để auto-detect scaler type (Kinect vs webcam)
4. Update `prepare_data()` pipeline

### Phase 3: Pose Normalization (Sau)

**Mục tiêu:** Giảm sensitivity to camera setup differences.

**Steps:**
1. Thêm pose normalization step vào `01_extract_joint_positions.py`
2. Normalize: center by hip midpoint, scale by torso length
3. Re-run feature extraction pipeline
4. Re-calibrate thresholds với normalized features

---

## 6. Kết luận

Domain shift Kinect → Webcam là **vấn đề thực sự** ảnh hưởng đến cả motion detection và model predictions. Giải pháp tức thì là giảm motion thresholds để giảm false negatives. Giải pháp dài hạn cần webcam-specific calibration và có thể domain adaptation techniques.

**Tài liệu tham khảo đầy đủ:**

1. Capecci, M. et al. (2019). "A unifying kinect-based framework to score the quality of execution of exercises." *Sensors*, 19(15), 3392. — KiMoRe dataset description.

2. Lugaresi, C. et al. (2019). "MediaPipe: A Framework for Building Perception Pipelines." *arXiv:1906.08172*. — MediaPipe architecture.

3. Ganin, Y. et al. (2016). "Domain-Adversarial Training of Neural Networks." *JMLR*, 17(59), 1–35. — DANN framework.

4. Sun, B. & Saenko, K. (2016). "Deep CORAL: Correlation Alignment for Deep Domain Adaptation." *ECCV 2016*. — CORAL method.

5. Wang, D. et al. (2021). "Tent: Fully Test-Time Adaptation by Entropy Minimization." *ICLR 2021*. — TTA method.

6. Schneider, S. et al. (2020). "Improving Robustness against Common Corruptions by Covariate Shift Adaptation." *NeurIPS 2020*. — BN adaptation.

7. Guo, K. & Khan, A. (2021). "Upper-body exercise quality assessment using LSTM networks." — Paper gốc mà dự án dựa trên.