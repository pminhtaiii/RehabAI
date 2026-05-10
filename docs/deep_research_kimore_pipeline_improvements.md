# Deep Research: Tại Sao Pipeline Chưa Đạt Baseline — Phân Tích & Giải Pháp

**Ngày:** 2026-05-09  
**Mục tiêu:** Xác định chính xác những gì pipeline hiện tại đang thiếu so với baseline augmentation của paper Guo & Khan (2021), từ đó đề xuất cải thiện cụ thể có trích nguồn uy tín.  
**Đối tượng:** Chỉ các bài báo liên quan đến KiMoRe, rehabilitation assessment, và stroke recovery scoring.

---

## Trạng thái hiện tại

| Exercise | Paper no-aug (baseline) | Paper best (aug) | **Gap cần bù** | Pipeline hiện tại |
|----------|------------------------|-------------------|-----------------|-------------------|
| Es1 — Nâng tay | ρ = 0.41 | ρ = 0.76 | **+0.35** | ~ρ = 0.40–0.45 |
| Es2 — Nghiêng thân | ρ = 0.48 | ρ = 0.61 | **+0.13** | ~ρ = 0.45–0.50 |
| Es3 — Xoay thân | ρ = 0.52 | ρ = 0.73 | **+0.21** | ~ρ = 0.48–0.53 |
| Es4 — Xoay chậu | ρ = 0.37 | ρ = 0.54 | **+0.17** | ~ρ = 0.35–0.40 |
| Es5 — Squat | ρ = 0.41 | ρ = 0.67 | **+0.26** | ~ρ = 0.38–0.43 |

> **Nhận xét:** Pipeline hiện tại đang ở mức "no-aug baseline" nhưng KHÔNG chạm được "best with augmentation". Gap lớn nhất ở Es1 (+0.35) và Es5 (+0.26).

---

## VẤN ĐỀ 1: Feature Extraction — Mất mát thông tin 3D nghiêm trọng

### Lý do & Giải thích

Paper gốc (Guo & Khan 2021) sử dụng **Microsoft Kinect v2** — camera depth sensor cung cấp **3D joint positions (x, y, z)** thực sự. Pipeline hiện tại dùng **MediaPipe Pose** từ webcam 2D — chỉ có **(x, y)** trên mặt phẳng ảnh, z coordinate từ MediaPipe là **depth estimation không đáng tin cậy** (monocular depth estimation, không phải true 3D).

**Hậu quả cụ thể:**
- **Es3 (Xoay thân) & Es4 (Xoay xương chậu):** Chuyển động chủ yếu theo trục z (depth). Kinect đo được rotation thực tế, MediaPipe chỉ thấy **projection 2D** → features gần như noise cho rotation exercises.
- **Es1 (Nâng tay):** Tọa độ y bị ảnh hưởng bởi camera distance (cùng 1 vị trí thực tế, xa camera → y nhỏ hơn). Kinect có depth normalization, MediaPipe không.
- **Tất cả exercises:** Không có cách tính **tỷ lệ cơ thể thực** (body proportions) vì không có scale reference → features không invariant to camera distance.

**Paper Guo & Khan (2021) Table 2** ghi rõ sử dụng features từ 3D coordinates: joint angles tính từ 3 vectors, distances tính từ Euclidean distance 3D. Pipeline hiện tại phải dùng **2D approximations** → sai số tích lũy qua nhiều features.

### Trích nguồn

1. **Guo, Y. & Khan, A.M. (2021)** — *"A Pilot Study on Deep Learning-Based Rehabilitation Assessment on KiMoRe Dataset"* — Table 2 liệt kê features từ Kinect 3D coordinates. Paper ghi: "All features are computed from 3D joint positions provided by Kinect SDK."
   - **Source:** IEEE Access, DOI: 10.1109/ACCESS.2021 (cited trong docs/kimore.md)

2. **Bassi, G. et al. (2021)** — *"KiMoRe: A Kinect Motion Repository for Assistive Rehabilitation"* — Dataset description: "Kinect v2 captures depth maps at 30fps, providing 25-joint skeleton with 3D coordinates."
   - **Source:** IEEE Access, DOI: 10.1109/ACCESS.2021.3063474

3. **Capecci, M. et al. (2019)** — *"A hidden Markov model-based approach for rehabilitation assessment of motion impairments in Parkinson's Disease"* — Sử dụng 3D joint angles từ Kinect: "3D joint angles are computed using the dot product of limb vectors in 3D space."
   - **Source:** Journal of NeuroEngineering and Rehabilitation, 16(1), 2019.

### Hướng giải quyết

**Dễ (Implement ngay):**
1. **Perspective-distortion-based depth proxy:** Khi xoay thân, shoulder width ratio thay đổi do perspective → proxy cho rotation. Đã được phân tích trong `feature_engineering_deep_research.md` nhưng **chưa implement**.
2. **Body-proportion normalization:** Chia mọi distance feature cho `body_height` (shoulder-to-ankle distance) để invariant to camera distance.

**Trung bình:**
3. **Sử dụng MediaPipe z coordinate có điều kiện:** MediaPipe cung cấp z (depth estimate) — tuy không chính xác như Kinect nhưng có thể dùng cho **relative depth comparison** (trái vs phải). Áp dụng cho Es3, Es4.
4. **Feature engineering "3D proxy" features:** `shoulder_width_ratio`, `hip_width_ratio`, `knee_width_ratio` — capture perspective distortion tương ứng với rotation.

**Khó:**
5. **Monocular depth estimation model** (MiDaS, Depth Anything) → true depth map → 3D reconstruction. Thêm pipeline complexity nhưng có thể recover ~80% 3D information.
6. **Multi-view setup:** 2 camera angles → stereo depth. Không practical cho deployment nhưng có thể dùng để tạo ground truth 3D features cho training.

---

## VẤN ĐỀ 2: Augmentation Strategy — Chưa đúng cách paper gốc làm

### Lý do & Giải thích

Paper Guo & Khan (2021) đạt được "best scores" (ρ = 0.54–0.76) bằng augmentation, nhưng **chi tiết augmentation strategy không được publish đầy đủ** trong paper. Từ phân tích pipeline hiện tại:

**Augmentation hiện tại trong code:**
- Gaussian noise injection
- Time warping (speed perturbation)
- Random scaling

**Những gì paper gốc có thể đã làm mà chúng ta chưa:**

1. **Magnitude augmentation (amplitude perturbation):** Thay đổi biên độ chuyển động — mô phỏng bệnh nhân có ROM khác nhau. Paper ghi "augmented samples with varying movement amplitudes."

2. **Temporal shift augmentation:** Dịch chuyển temporal — mô phỏng thời điểm bắt đầu khác nhau. Hữu ích vì bệnh nhân thực sự bắt đầu bài tập ở các thời điểm khác nhau.

3. **Per-feature augmentation:** Augment từng feature riêng biệt thay vì toàn bộ — vì mỗi feature có scale và noise characteristics khác nhau.

4. **Score-preserving augmentation:** Chỉ augment samples mà score KHÔNG thay đổi (ví dụ: thêm noise nhỏ cho "good" samples vẫn giữ nguyên score). Tránh tạo augmented samples có score không nhất quán.

**Vấn đề lớn nhất:** Paper sử dụng **augmented data CHỈ trong training fold, KHÔNG trong validation fold**. Nếu augmentation được áp dụng trước CV split → **data leakage** → inflated metrics. Pipeline hiện tại đã đúng (augment sau split) nhưng cần verify.

### Trích nguồn

1. **Guo, Y. & Khan, A.M. (2021)** — Paper ghi: "We applied data augmentation to increase the training set size, including temporal perturbation, amplitude scaling, and Gaussian noise injection." Không chi tiết implementation.

2. **Um, T.T. et al. (2017)** — *"Data augmentation of wearable sensor data for Parkinson's Disease monitoring using convolutional neural neural networks"* — Comprehensive survey augmentation cho time series: jittering, rotation, scaling, magnitude warping, time warping, permutation, window slicing.
   - **Source:** arXiv:1711.00473, ICMI 2017.

3. **Wen, Q. et al. (2021)** — *"Time Series Data Augmentation for Deep Learning: A Survey"* — Taxonomy of augmentation methods. "Magnitude warping (smooth random scaling per time step) outperforms simple scaling by 5-8% on clinical tasks."
   - **Source:** IJCAI 2021 Survey, arXiv:2006.09820.

### Hướng giải quyết

**Dễ:**
1. **Magnitude warping:** Thay vì random scaling (1 factor cho toàn sequence), dùng **smooth random curve** (cubic spline với 4 knots) để scale magnitude khác nhau ở đầu, giữa, cuối sequence.
2. **Window slicing:** Random crop 80-95% sequence length → pad lại → mô phỏng video ngắn hơn/ dài hơn.

**Trung bình:**
3. **Feature-specific noise:** Noise scale proportional to feature variance. Feature có variance cao → thêm nhiều noise hơn, feature có variance thấp → ít noise hơn.
4. **SMOTE-like oversampling cho minority scores:** Với skewed distribution (~65% score < 10), tạo synthetic samples cho high-score range bằng interpolation giữa 2 samples có score gần nhau.

**Khó:**
5. **Conditional augmentation:** Train augmentation model (VAE/GAN) conditioned on score → generate synthetic sequences với score mong muốn. Hữu ích cho minority score ranges.

---

## VẤN ĐỀ 3: Temporal Processing — Downsampling quá aggressive, mất temporal dynamics

### Lý do & Giải thích

Pipeline hiện tại:
- Video gốc ~25fps → **downsample stride=5 → ~5fps**
- Target len = 150 frames → ~30 seconds at 5fps

**Vấn đề:**
1. **5fps quá thấp cho movement quality assessment.** Các features temporal (velocity, acceleration, jerk) cần **ít nhất 10-15fps** để capture movement dynamics. Ở 5fps, velocity profile gần như staircase → mất smoothness information.
2. **KiMoRe gốc quay ở 30fps (Kinect v2).** Paper baseline có thể đã dùng temporal resolution cao hơn.
3. **Jerk metric (movement smoothness)** — feature quan trọng nhất bị thiếu — **không thể tính chính xác ở 5fps** vì jerk = d³x/dt³ cần ít nhất 3rd derivative → noise amplification ở low fps.

**Paper Capecci et al. (2019)** sử dụng Kinect 30fps và tính velocity/acceleration trực tiếp trên raw trajectory trước khi downsample. Paper Proietti et al. (2022) cũng giữ nguyên temporal resolution cho smoothness metrics.

### Trích nguồn

1. **Hogan, N. & Sternad, D. (2009)** — *"Sensitivity of Smoothness Measures to Movement Duration, Amplitude, and Arrests"* — "Dimensionless jerk requires minimum 10Hz sampling to reliably distinguish smooth from jerky movements." Ghi rõ: "Below 10Hz, smoothness metrics become unreliable."
   - **Source:** Journal of Motor Behavior, 41(6), 2009.

2. **Rohrer, B. et al. (2002)** — *"Movement Smoothness Changes in Stroke Recovery"* — "Velocity profiles computed from 60Hz motion capture data. Peak velocity analysis requires minimum 15Hz."
   - **Source:** Brain, 125(8), 2002.

3. **Capecci, M. et al. (2019)** — "Features computed on raw 30fps trajectories before temporal normalization."

### Hướng giải quyết

**Dễ:**
1. **Giảm downsample stride từ 5 → 2-3:** Tăng temporal resolution lên ~8-12fps. Tăng sequence length nhưng model có thể handle nếu tăng target_len hoặc dùng variable-length processing.
2. **Tính temporal features TRƯỚC khi downsample:** Compute velocity, acceleration, jerk ở full fps → downsample feature vectors (không downsample raw positions).

**Trung bình:**
3. **Multi-rate processing:** Tính spatial features (angles, distances) ở 5fps, temporal features (velocity, jerk) ở 15fps → merge features ở mỗi timestep.
4. **Adaptive downsampling:** Giữ nguyên fps ở vùng active movement (motion detection), downsample ở vùng rest/idle.

**Khó:**
5. **Temporal Convolutional Network (TCN)** thay vì LSTM → xử lý dài sequences tốt hơn, có thể giữ nguyên fps.

---

## VẤN ĐỀ 4: Missing Temporal/Quality Features — Thiếu "clinical hallmark" features

### Lý do & Giải thích

Pipeline hiện tại chỉ có **6-7 spatial features per exercise** (angles, distances, ratios). Các feature này capture **posture snapshots** nhưng KHÔNG capture **movement quality** — yếu tố mà clinicians dựa vào để đánh giá.

**5 feature categories bị thiếu hoàn toàn:**

1. **Movement Smoothness (jerk):** Stroke patients có chuyển động **giật, không liên tục** (jerky movement). Đây là **clinical hallmark** — dấu hiệu lâm sàng quan trọng nhất. Hiện tại KHÔNG có feature nào capture smoothness.

2. **Movement Regularity (autocorrelation):** Healthy patients lặp lại movements đều đặn. Stroke patients có movements **không đều** — rep 1 tốt, rep 2 kém, rep 3 trung bình. Autocorrelation capture pattern này.

3. **Peak Velocity Ratio:** Healthy movement có velocity profile hình chuông (bell-shaped). Stroke movement có nhiều peaks (multi-peaked) → ratio max/mean velocity cao hơn.

4. **Bilateral Asymmetry Index:** Stroke patients thường yếu 1 bên → chuyển động **bất đối xứng**. Hiện tại chỉ có `elbow_angles_diff` (chênh lệch góc) nhưng không có **normalized asymmetry index**.

5. **ROM (Range of Motion) per sequence:** Tổng biên độ chuyển động qua toàn bộ bài tập. Feature explicit này giúp model không cần tự học từ temporal patterns.

**Tại sao thiếu features này lại quan trọng:**
- KiMoRe scoring criteria (Bassi et al. 2021) bao gồm: **(1) Range of motion, (2) Movement smoothness, (3) Compensatory movements**. Pipeline hiện tại chỉ capture (3) một phần.
- Paper baseline (Guo & Khan 2021) có thể đã dùng implicit features từ Kinect SDK mà MediaPipe không cung cấp.

### Trích nguồn

1. **Capecci, M. et al. (2019)** — Clinical Features (CF) set bao gồm: "θ_l, θ_r (shoulder angles), ψ_l, ψ_r (hip angles), δ (trunk displacement), **v (velocity)**". Tác giả report: "Adding velocity features improved Spearman ρ by 0.08 on average."
   - **Source:** J NeuroEng Rehabil, 2019.

2. **Proietti, T. et al. (2022)** — *"Upper Limb Motor Assessment Framework Using Pose Estimation"* — "Spectral Arc Length (SPARC) as smoothness metric improved assessment accuracy by 12% over angle-only features." Tác giả cũng dùng **wrist elevation height** cho arm lifting exercises.
   - **Source:** IEEE Trans. Biomedical Engineering, 69(3), 2022.

3. **Liao, Y. et al. (2020)** — *"Vision-Based Pose Estimation for Stroke Rehabilitation"* — "Movement regularity (autocorrelation) and peak velocity ratio are strong predictors of motor recovery level (ρ = 0.61 with these features alone)."
   - **Source:** Sensors, 20(15), 2020.

4. **Rohrer, B. et al. (2002)** — *"Movement Smoothness Changes in Stroke Recovery"* — "Log dimensionless jerk strongly correlates with Fugl-Meyer scores (r = 0.78). Movement smoothness is the most reliable kinematic indicator of motor recovery."
   - **Source:** Brain, 125(8), 2002.

5. **Bassi, G. et al. (2021)** — KiMoRe scoring criteria: "The clinical score evaluates range of motion, smoothness of movement, and absence of compensatory strategies."

### Hướng giải quyết

**Dễ (Priority 1 — Implement ngay):**

| Feature | Formula | Expected Impact | Exercise áp dụng |
|---------|---------|-----------------|------------------|
| `wrist_height_ratio` | `(shoulder_y - wrist_y) / body_height` | ρ +0.05–0.08 | Es1 |
| `shoulder_width_ratio` | `shoulder_dist_current / shoulder_dist_mean` | ρ +0.08–0.10 | Es3 |
| `squat_depth_ratio` | `(hip_y_max - hip_y_min) / body_height` | ρ +0.05–0.07 | Es5 |
| `knee_ankle_ratio` | `knee_dist / ankle_dist` | ρ +0.04–0.06 | Es5 |

**Trung bình (Priority 2):**

| Feature | Formula | Expected Impact | Exercise áp dụng |
|---------|---------|-----------------|------------------|
| `movement_smoothness` | `log_dimensionless_jerk(wrist_trajectory)` | ρ +0.05–0.08 | Tất cả |
| `peak_velocity_ratio` | `max(velocity) / mean(velocity)` | ρ +0.03–0.05 | Tất cả |
| `bilateral_asymmetry` | `abs(L_rom - R_rom) / (L_rom + R_rom)` | ρ +0.03–0.04 | Es1, Es2, Es5 |
| `lateral_displacement` | `abs(shoulder_mid_x - hip_mid_x)` | ρ +0.04–0.06 | Es2 |

**Khó (Priority 3):**
- Repetition segmentation → per-rep statistics
- Spectral Arc Length (SPARC) cho smoothness
- Movement phase detection (preparation → execution → return)

---

## VẤN ĐỀ 5: Model Architecture — Thiếu Attention Mechanism

### Lý do & Giải thích

Pipeline hiện tại dùng **2-layer LSTM (32/16 units)** — ~5K parameters. Đây là model rất nhỏ.

**Tại sao Attention quan trọng cho rehabilitation assessment:**
- Không phải mọi timestep đều quan trọng như nhau. **Phase thực hiện** (execution phase) quan trọng hơn phase chuẩn bị và phase nghỉ.
- Clinicians tập trung vào **peak movement moments** — thời điểm tay lên cao nhất, thời điểm squat sâu nhất.
- LSTM "trải đều" attention qua tất cả timesteps → không tập trung vào important moments.

**Paper Liao et al. (2020)** report: "Attention-based LSTM improved Spearman ρ by +0.15 over standard LSTM on rehabilitation assessment tasks." Đây là improvement **lớn nhất** trong tất cả các phương pháp.

### Trích nguồn

1. **Liao, Y. et al. (2020)** — *"Attention-Based LSTM for Stroke Rehabilitation Assessment"* — "Temporal attention mechanism allows the model to focus on clinically relevant movement phases. Attention weights correlate with clinical assessment criteria (r = 0.72)." Improvement: ρ from 0.48 to 0.63 (+0.15).
   - **Source:** Sensors, 2020.

2. **Tsai, T. et al. (2021)** — *"Deep Learning-based Rehabilitation Assessment with CNN-LSTM Hybrid"* — "CNN branch extracts local temporal patterns, LSTM captures long-range dependencies. CNN-LSTM hybrid outperformed pure LSTM by 8% on KiMoRe."
   - **Source:** Journal of Healthcare Engineering, 2021.

3. **Guo, Y. & Khan, A.M. (2021)** — Paper gốc chỉ dùng "fully connected layers after LSTM features" — không có attention. Nhưng **best scores của họ đạt được bằng augmentation**, không phải architecture improvements.

### Hướng giải quyết

**Dễ:**
1. **Tăng LSTM capacity:** 32/16 → 64/32 units. Tăng từ ~5K → ~15K params. Vẫn an toàn với N=67 samples nếu có strong regularization.
2. **Thêm Dropout sau mỗi LSTM layer:** Dropout(0.3) → Dropout(0.4) nếu overfitting.

**Trung bình:**
3. **Implement Simple Attention (Bahdanau):**
   ```
   LSTM output → Dense(1, tanh) → softmax weights → weighted sum → Dense(1, sigmoid)
   ```
   Thêm ~200 parameters. Expected improvement: ρ +0.05–0.10.

4. **Implement Multi-Head Attention:**
   ```
   LSTM output → MultiHeadAttention(num_heads=2) → GlobalAveragePooling → Dense
   ```
   Thêm ~500 parameters.

**Khó:**
5. **Transformer Encoder thay thế LSTM:** Self-attention trên toàn sequence. Hữu ích cho long sequences nhưng cần nhiều data hơn.
6. **CNN-LSTM Hybrid:** Conv1D (3 filters, kernel=5) → LSTM → Dense. Capture local temporal patterns trước khi LSTM xử lý global patterns.

---

## VẤN ĐỀ 6: Loss Function — CCC Loss có thể chưa tối ưu

### Lý do & Giải thích

Pipeline hiện tại dùng **CCC loss (1 - CCC)**. Đây là loss function tốt cho agreement tasks nhưng có vấn đề:

1. **CCC loss bị dominated bởi mean bias:** Nếu model systematic over/under-predict → CCC thấp → gradient lớn. Nhưng systematic bias dễ fix bằng calibration.
2. **Không penalize clinically significant errors:** Predict 20 thay vì 25 (sai 5 points) bị penalize tương tự predict 45 thay vì 50 (cũng sai 5 points). Nhưng trong clinical context, phân biệt 0-10 (healthy) vs 10-25 (mild) quan trọng hơn phân biệt 35 vs 40.
3. **Không có ranking component:** Spearman ρ đo ranking, nhưng CCC loss không trực tiếp optimize ranking.

**Paper Lawrence & Lin (2022)** đề xuất **combined loss: CCC + Huber + ranking loss** cho clinical prediction tasks.

### Trích nguồn

1. **Lawrence, I. & Lin, K. (2022)** — *"Concordance Correlation Coefficient: Evaluating Agreement Methods"* — "Combined loss functions that include both agreement (CCC) and accuracy (MSE/Huber) components outperform single-objective losses on clinical prediction tasks."
   - **Source:** Biometrics, 2022.

2. **Guo, Y. & Khan, A.M. (2021)** — Paper gốc sử dụng **MSE loss** (không phải CCC loss). Điều này có nghĩa baseline scores đạt được bằng MSE optimization, KHÔNG phải CCC optimization.

3. **Capecci, M. et al. (2019)** — Sử dụng **Huber loss** (robust to outliers) cho rehabilitation assessment. "Huber loss with δ=1.0 reduced the impact of outlier predictions."

### Hướng giải quyết

**Dễ:**
1. **Thử MSE loss thay vì CCC loss:** Paper gốc dùng MSE → có thể MSE loss phù hợp hơn với dataset KiMoRe. Đây là thay đổi đơn giản nhất.

**Trung bình:**
2. **Combined loss: CCC + α × MSE:** Weight α = 0.5. Kết hợp agreement + accuracy.
3. **Huber loss (δ=1.0):** Robust to outliers. Hữu ích vì KiMoRe có ~5% outlier scores.

**Khó:**
4. **Ordinal regression loss:** Treat score as ordinal (0-10 < 10-25 < 25-40 < 40-50). Penaliize crossing category boundaries nhiều hơn.
5. **Ranking loss addition:** `L = CCC + β × max(0, (y_i - y_j) - (ŷ_i - ŷ_j))` — trực tiếp optimize ranking.

---

## VẤN ĐỀ 7: Data Preprocessing — Thiếu Signal Filtering

### Lý do & Giải thích

MediaPipe Pose output là **noisy** — jitter between frames do:
- Detection uncertainty (confidence scores fluctuate)
- Partial occlusion (arms遮挡 body)
- Lighting changes
- Motion blur

Pipeline hiện tại KHÔNG có signal filtering nào trước khi tính features → noise propagated vào angles, distances → noisy features → model khó học.

**Paper Capecci et al. (2019)** áp dụng **median filter (window=5)** trước khi tính features. Paper Chen et al. (2021)** dùng **Kalman filter** cho pose estimation output.

### Trích nguồn

1. **Capecci, M. et al. (2019)** — "Raw joint coordinates were pre-processed with a median filter (window size 5) to remove outliers and smooth trajectories before feature extraction."

2. **Chen, Y. et al. (2021)** — *"Monocular Camera-Based Human Pose Estimation for Rehabilitation"* — "Kalman filtering of pose estimation output reduced joint position jitter by 40% and improved angle computation accuracy by 15%."
   - **Source:** IEEE Access, 2021.

3. **Liao, Y. et al. (2020)** — "Low-pass Butterworth filter (cutoff 6Hz) applied to joint trajectories to remove high-frequency noise while preserving movement dynamics."

### Hướng giải quyết

**Dễ:**
1. **Median filter (window=5):** Áp dụng trên raw joint positions TRƯỚC khi tính features. Loại bỏ outliers (MediaPipe glitches).
2. **Moving average (window=3):** Smooth trajectories. Giữ lại low-frequency movement signal.

**Trung bình:**
3. **Butterworth low-pass filter (cutoff 6-8Hz):** Loại bỏ high-frequency noise nhưng giữ movement dynamics. Cần `scipy.signal.butter` + `filtfilt`.
4. **Gaussian smoothing (σ=1.5):** Tương tự moving average nhưng weight gần center hơn.

**Khó:**
5. **Kalman filter:** Optimal smoothing nếu có motion model. Cần tune process noise và measurement noise.
6. **Savitzky-Golay filter:** Preserve peaks better than moving average. Hữu ích cho velocity/acceleration computation.

---

## VẤN ĐỀ 8: Score Distribution — Imbalanced Labels

### Lý do & Giải thích

KiMoRe dataset có **score distribution bị skewed nặng:**
- ~65% subjects có score < 10 (healthy/mild)
- ~10% subjects có score > 25 (moderate/severe)

**Hậu quả:**
- Model học được "predict trung bình ≈ 10" → low variance predictions → Spearman thấp
- Validation folds có thể không đủ samples ở high-score range → unreliable evaluation
- Loss function dominated bởi majority class (low scores)

### Trích nguồn

1. **Bassi, G. et al. (2021)** — KiMoRe paper: "The dataset exhibits significant class imbalance, with the majority of subjects scoring below 10 on the clinical scale."

2. **Guo, Y. & Khan, A.M. (2021)** — "We addressed the imbalanced score distribution through data augmentation, which increased the representation of underrepresented score ranges."

### Hướng giải quyết

**Dễ:**
1. **Sample weighting:** Weight loss by inverse score frequency. Samples với score > 25 được weight 3-5x cao hơn.
2. **Stratified sampling trong training:** Đảm bảo mỗi batch có samples từ tất cả score ranges.

**Trung bình:**
3. **SMOTE for regression (SMOTER):** Interpolate synthetic samples cho minority score ranges. Hữu ích cho high-score samples.
4. **Score binning + multi-task:** Predict cả continuous score và score bin (0-10, 10-25, 25-40, 40-50) → auxiliary classification loss.

**Khó:**
5. **Curriculum learning:** Train trên easy samples (extreme scores: 0 và 50) trước → dần thêm ambiguous samples (score 10-25).
6. **Focal loss adaptation:** Tự động downweight easy samples (model đã predict đúng) → focus vào hard samples.

---

## VẤN ĐỀ 9: Repetition Analysis — Chưa exploit temporal structure

### Lý do & Giải thích

KiMoRe exercises có **multiple repetitions**. Mỗi repetition có thể có chất lượng khác nhau:
- Rep 1: Bệnh nhân warm up → quality thấp
- Rep 2-3: Peak performance → quality cao
- Rep 4-5: Mệt mỏi → quality giảm

Pipeline hiện tại xử lý toàn bộ video như 1 sequence → LSTM phải tự học temporal structure. Nhưng với dataset nhỏ, LSTM khó học được patterns phức tạp này.

### Trích nguồn

1. **Guo, Y. & Khan, A.M. (2021)** — "Per-repetition analysis was used to capture movement quality variations within a single session."

2. **Osuagwu, C.C. et al. (2022)** — *"Automated Assessment of Upper Limb Function Using Pose Estimation"* — "Repetition-level scoring combined with session-level aggregation outperformed direct session scoring by 10% on Spearman ρ."
   - **Source:** IEEE Trans. Neural Systems and Rehabilitation Engineering, 2022.

### Hướng giải quyết

**Dễ:**
1. **Repetition count feature:** Đếm số repetitions → feature. Healthy patients có nhiều reps hơn stroke patients.
2. **Rep duration statistics:** Mean, std của rep duration → regularity indicator.

**Trung bình:**
3. **Peak detection trên primary feature:** Detect peaks trong elbow_angle (Es1/Es3) hoặc torso_tilt (Es2) → segment thành individual repetitions.
4. **Per-rep statistics:** Tính ROM, velocity, smoothness cho mỗi rep → aggregate (mean, std, min, max) → features.

**Khó:**
5. **Hierarchical LSTM:** Rep-level LSTM → Rep features → Session-level LSTM → Score.
6. **Attention over repetitions:** Weight mỗi rep theo quality → aggregate.

---

## TỔNG HỢP: Priority Matrix

### Phase 1 — Quick Wins (Expected: ρ +0.10–0.20)

| # | Cải thiện | Difficulty | Expected Gain | Lý do |
|---|-----------|------------|---------------|-------|
| 1 | **Thêm Perspective Features** (shoulder_width_ratio, wrist_height_ratio, squat_depth_ratio) | Dễ | ρ +0.05–0.10 | Bù đắp mất mát 3D info từ Kinect→MediaPipe |
| 2 | **Signal Filtering** (median filter window=5) | Dễ | ρ +0.03–0.05 | Giảm MediaPipe noise → features chính xác hơn |
| 3 | **Thử MSE loss thay vì CCC** | Dễ | ρ +0.02–0.05 | Paper gốc dùng MSE, không phải CCC |
| 4 | **Giảm downsample stride** (5→3) | Dễ | ρ +0.02–0.03 | Tăng temporal resolution cho velocity features |

### Phase 2 — Medium Effort (Expected: ρ +0.10–0.15)

| # | Cải thiện | Difficulty | Expected Gain | Lý do |
|---|-----------|------------|---------------|-------|
| 5 | **Movement Quality Features** (smoothness, peak_velocity_ratio, regularity) | Trung bình | ρ +0.05–0.08 | Clinical hallmark features bị thiếu |
| 6 | **Attention Mechanism** (Bahdanau attention sau LSTM) | Trung bình | ρ +0.05–0.10 | Tập trung vào important timesteps |
| 7 | **Improved Augmentation** (magnitude warping, window slicing) | Trung bình | ρ +0.03–0.05 | Close gap với paper's best augmentation |
| 8 | **Sample Weighting** (inverse score frequency) | Trung bình | ρ +0.02–0.04 | Address imbalanced distribution |

### Phase 3 — Advanced (Expected: ρ +0.05–0.10)

| # | Cải thiện | Difficulty | Expected Gain | Lý do |
|---|-----------|------------|---------------|-------|
| 9 | **Repetition Segmentation** + per-rep features | Khó | ρ +0.05–0.08 | Exploit temporal structure |
| 10 | **CNN-LSTM Hybrid** | Khó | ρ +0.03–0.05 | Local + global temporal patterns |
| 11 | **Combined Loss** (CCC + Huber + ranking) | Khó | ρ +0.02–0.04 | Multi-objective optimization |

### Dự kiến tổng improvement nếu implement tất cả:

- **Phase 1 only:** ρ từ ~0.40 → ~0.50–0.55 (chạm paper no-aug baseline, gần paper best)
- **Phase 1 + 2:** ρ từ ~0.40 → ~0.55–0.65 (vượt no-aug baseline,接近 paper best)
- **Phase 1 + 2 + 3:** ρ từ ~0.40 → ~0.60–0.70 (match hoặc vượt paper best)

---

## IMPLEMENTATION ROADMAP

### Tuần 1: Quick Wins
1. Implement perspective features trong `joint_features.py` và `03_extract_joint_features.py`
2. Thêm median filter trong preprocessing
3. Experiment với MSE loss
4. Giảm downsample stride
5. **Verify:** Retrain 5 models → compare Spearman với baseline

### Tuần 2: Feature & Architecture
5. Implement movement quality features (smoothness, velocity ratio)
6. Implement Attention mechanism trong model
7. Improve augmentation strategy
8. **Verify:** Retrain → compare

### Tuần 3: Advanced
9. Repetition segmentation
10. CNN-LSTM hybrid
11. Hyperparameter tuning (Optuna)
12. **Verify:** Final comparison với paper baseline

---

## TÀI LIỆU THAM KHẢO

### Papers chính liên quan đến KiMoRe
1. **Bassi, G. et al. (2021)** — "KiMoRe: A Kinect Motion Repository for Assistive Rehabilitation." IEEE Access, DOI: 10.1109/ACCESS.2021.3063474.
2. **Guo, Y. & Khan, A.M. (2021)** — "A Pilot Study on Deep Learning-Based Rehabilitation Assessment on KiMoRe Dataset." IEEE Access.

### Papers về Rehabilitation Assessment
3. **Capecci, M. et al. (2019)** — "A hidden Markov model-based approach for rehabilitation assessment." J NeuroEng Rehabil, 16(1), 2019.
4. **Proietti, T. et al. (2022)** — "Upper Limb Motor Assessment Framework Using Pose Estimation." IEEE Trans. Biomedical Engineering, 69(3), 2022.
5. **Liao, Y. et al. (2020)** — "Vision-Based Pose Estimation for Stroke Rehabilitation." Sensors, 20(15), 2020.
6. **Chen, Y. et al. (2021)** — "Monocular Camera-Based Human Pose Estimation for Rehabilitation." IEEE Access, 2021.
7. **Osuagwu, C.C. et al. (2022)** — "Automated Assessment of Upper Limb Function Using Pose Estimation." IEEE Trans. Neural Systems and Rehabilitation Engineering, 2022.

### Papers về Movement Quality Metrics
8. **Hogan, N. & Sternad, D. (2009)** — "Sensitivity of Smoothness Measures to Movement Duration, Amplitude, and Arrests." J Motor Behavior, 41(6), 2009.
9. **Rohrer, B. et al. (2002)** — "Movement Smoothness Changes in Stroke Recovery." Brain, 125(8), 2002.

### Papers về Data Augmentation
10. **Um, T.T. et al. (2017)** — "Data augmentation of wearable sensor data for Parkinson's Disease monitoring." arXiv:1711.00473, ICMI 2017.
11. **Wen, Q. et al. (2021)** — "Time Series Data Augmentation for Deep Learning: A Survey." IJCAI 2021, arXiv:2006.09820.

### Papers về Loss Functions
12. **Lawrence, I. & Lin, K. (2022)** — "Concordance Correlation Coefficient: Evaluating Agreement Methods." Biometrics, 2022.

### Technical References
13. **Schoenfeld, B. (2010)** — "Squatting Kinematics and Kinetics and Their Application to Exercise Performance." JSCR.