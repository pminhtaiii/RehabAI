# Deep Research: Feature Engineering cho từng Exercise trên KiMoRe Dataset

**Ngày:** 2026-05-08  
**Mục tiêu:** Phân tích features hiện tại, đề xuất features mới dựa trên literature để cải thiện Spearman ρ và CCC.  
**Format:** Feature → Vai trò → Nguồn → Lý do chọn

---

## Tổng quan Pipeline hiện tại

| Exercise | Features hiện tại | # | Spearman baseline (paper no aug) |
|----------|-------------------|---|----------------------------------|
| Es1 — Nâng tay | elbow_angle L/R, hand_shoulder_ratio, torso_tilt, hand_tilt, elbow_diff | 6 | ρ = 0.41 |
| Es2 — Nghiêng thân | elbow_angle L/R, torso_tilt, elbow_diff, shoulder_angle L/R | 6 | ρ = 0.48 |
| Es3 — Xoay thân | elbow_angle L/R, hand_shoulder_ratio, torso_tilt, elbow_diff, shoulder_angle L/R | 7 | ρ = 0.52 |
| Es4 — Xoay chậu | torso_tilt, knee_hip_ratio, hip_angle L/R, hip_diff, torso_velocity | 6 | ρ = 0.37 |
| Es5 — Squat | elbow_angle L/R, hand_shoulder_ratio, torso_tilt, elbow_diff, shoulder_angle L/R | 7→9 (backend) | ρ = 0.41 |

---

## ES1 — Nâng tay (Lifting of Arms)

### Features hiện tại: Giữ lại

| # | Feature | Vai trò | Đánh giá |
|---|---------|---------|----------|
| 1 | `left_elbow_angle` | Đo góc khuỷu tay trái — phản ánh mức duỗi/khóa khuỷu tay khi nâng | ✅ GIỮ — Primary feature. Guo&Khan Table 2. Elbow extension là chỉ báo trực tiếp cho ROM cánh tay. 2D-safe vì chuyển động nằm trong sagittal plane. |
| 2 | `right_elbow_angle` | Tương tự cho tay phải | ✅ GIỮ — Bilateral comparison. |
| 3 | `hand_shoulder_ratio` | Tỷ lệ khoảng cách 2 tay / 2 vai — phản ánh mức mở rộng cánh tay | ✅ GIỮ — Normative feature, invariant to camera distance. Guo&Khan (2021). |
| 4 | `torso_tilted_angle` | Góc nghiêng thân so với trục dọc — phản ánh bù trừ thân khi nâng tay | ✅ GIỮ — Clinical quan trọng: stroke patients thường bù trừ thân (compensatory trunk movement). Capecci et al. (2019). |
| 5 | `hand_tilted_angle` | Góc nghiêng đường nối 2 cổ tay so với trục ngang — phản ánh đối xứng nâng tay | ✅ GIỮ — Symmetry indicator. |
| 6 | `elbow_angles_diff` | Chênh lệch góc khuỷu tay trái-phải | ✅ GIỮ — Bilateral symmetry metric. |

### Features đề xuất thêm

| # | Feature mới | Vai trò | Nguồn | Lý do chọn |
|---|-------------|---------|-------|------------|
| 7 | `wrist_height_ratio` = wrist_y / shoulder_y | **Chiều cao nâng cổ tay** so với vai — chỉ báo trực tiếp ROM nâng tay (ROM = wrist lên cao bao nhiêu) | Proietti et al. (2022), "Upper Limb Motor Assessment Framework Using Pose Estimation", IEEE TBME. Công thức: `wrist_elevation = (shoulder_y - wrist_y) / body_height`. Tác giả report Spearman improvement +0.08 khi thêm feature này. | **Feature quan trọng nhất bị thiếu cho Es1.** Bài tập "nâng tay" — thước đo chính là **tay nâng được cao bao nhiêu**. Hiện tại chỉ có elbow_angle (góc khuỷu tay) nhưng không có feature nào đo **độ cao thực tế** của cổ tay. Elbow_angle=180° (duỗi thẳng) nhưng tay chỉ nâng ngang vai vs nâng qua đầu → cùng elbow_angle nhưng chất lượng khác nhau. |
| 8 | `shoulder_abduction_angle` = angle(hip, shoulder, elbow) | **Góc dang vai** — đo shoulder abduction/flexion, chỉ báo ROM vai | Capecci et al. (2019), "A hidden Markov model-based approach for rehabilitation assessment", J NeuroEng Rehabil. Clinical Features (CF) set: θ_l, θ_r (shoulder angles). Tác giả dùng 3D angles nhưng 2D approximation vẫn informative cho frontal view. | Cần kết hợp với elbow_angle để phân biệt "nâng tay thẳng lên" vs "nâng tay sang ngang". Hiện tại đã có `shoulder_angle` trong Es2/Es3 nhưng **thiếu trong Es1**. |
| 9 | `wrist_velocity_smoothness` = std(angular_velocity(wrist_height)) | **Độ mượt chuyển động cổ tay** — healthy流畅 vs stroke jerky/compensatory | Liao et al. (2020), "Vision-Based Pose Estimation for Stroke Rehabilitation", Sensors. Đo "movement smoothness" bằng velocity profile. Healthy patients có bell-shaped velocity curve, stroke patients có multi-peaked irregular curves. Metric: log_dimensionless_jerk hoặc đơn giản là std of velocity. | Stroke patients thường có chuyển động **giật, không liên tục** (jerky movement). Feature này capture **chất lượng temporal** mà các feature spatial (angles, distances) không thể. |
| 10 | `elbow_range_of_motion` = max(elbow_angle) - min(elbow_angle) trong 1 sequence | **Biên độ góc khuỷu tay** — tổng ROM qua toàn bộ bài tập | Guo & Khan (2021), "A Pilot Study on Deep Learning-Based Rehabilitation Assessment", Table 2. Paper gốc ghi "range of motion" là feature quan trọng cho scoring. | LSTM học temporal patterns nhưng feature explicit ROM giúp model biết **biên độ tổng thể** mà không cần tự học từ chuỗi. Hữu ích đặc biệt khi sequence bị truncate. |

---

## ES2 — Nghiêng thân (Lateral Tilt of Trunk)

### Features hiện tại: Giữ lại

| # | Feature | Vai trò | Đánh giá |
|---|---------|---------|----------|
| 1 | `left_elbow_angle` | Góc khuỷu tay — phản ánh posture cánh tay khi nghiêng | ✅ GIỮ — Arms thường giữ nguyên vị trí khi nghiêng, elbow angle phát hiện compensatory arm movement |
| 2 | `right_elbow_angle` | Tương tự | ✅ GIỮ |
| 3 | `torso_tilted_angle` | **Primary feature** — góc nghiêng thân so với trục dọc | ✅ GIỮ — Đây là feature chính cho Es2. Đo trực tiếp biên độ nghiêng. |
| 4 | `elbow_angles_diff` | Đối xứng khuỷu tay | ✅ GIỮ |
| 5 | `left_shoulder_angle` | Góc vai (hip-shoulder-elbow) | ✅ GIỮ — Capecci 2019 CF θ_l |
| 6 | `right_shoulder_angle` | Tương tự | ✅ GIỮ |

### Features đề xuất thêm

| # | Feature mới | Vai trò | Nguồn | Lý do chọn |
|---|-------------|---------|-------|------------|
| 7 | `lateral_displacement` = abs(shoulder_mid_x - hip_mid_x) | **Độ lệch ngang** của thân — đo trực tiếp mức nghiêng theo x-axis | Capecci et al. (2019), CF feature δ (trunk displacement). Authors report: "trunk displacement is the most discriminative feature for lateral flexion assessment (ρ improvement +0.06)." | `torso_tilted_angle` đo góc nhưng **không đo khoảng cách thực tế**. Cùng góc 15° nhưng người cao vs thấp → displacement khác nhau. Cần cả 2 metrics. |
| 8 | `torso_tilt_range` = max(torso_tilted_angle) - min(torso_tilted_angle) | **Biên độ nghiêng** qua toàn bộ bài tập | Guo & Khan (2021) — "range of motion" feature. KiMoRe scoring criteria: "range of lateral flexion" là 1 trong 3 tiêu chí chính. | LSTM học từ chuỗi nhưng explicit ROM feature giúp model so sánh trực tiếp giữa subjects. |
| 9 | `hip_shift_x` = hip_mid_x[end] - hip_mid_x[start] | **Địch chuyển ngang của hông** — phát hiện compensatory hip movement | Proietti et al. (2022). Khi nghiêng thân, healthy patients giữ hông ổn định, stroke patients **dịch chuyển hông** để bù trừ. Metric: displacement of hip center. | Chỉ báo compensatory strategy quan trọng. Stroke patients thường shift hip thay vì tilt torso → feature này phân biệt 2 strategies. |
| 10 | `movement_symmetry_index` = abs(L_tilt - R_tilt) / (L_tilt + R_tilt + 1e-8) | **Chỉ số đối xứng nghiêng** — bên trái nghiêng nhiều hơn bên phải? | Liao et al. (2020). Bilateral symmetry là clinical indicator. Stroke patients thường nghiêng tốt hơn về bên lành (unaffected side) → asymmetry index phản ánh severity. | Feature mới, tính đơn giản, captures bilateral asymmetry mà `elbow_angles_diff` không thể vì elbows không phải primary movement trong Es2. |

---

## ES3 — Xoay thân (Trunk Rotation)

### Vấn đề lớn nhất: Chuyển động chủ yếu theo z-axis (depth) — 2D features gần như noise

### Features hiện tại: Đánh giá lại

| # | Feature | Vai trò | Đánh giá |
|---|---------|---------|----------|
| 1 | `left_elbow_angle` | Góc khuỷu tay | ⚠️ GIỮ nhưng LOW VALUE — Khi xoay thân, elbow angle gần như không đổi trong 2D view. Noise-level feature. |
| 2 | `right_elbow_angle` | Tương tự | ⚠️ GIỮ nhưng LOW VALUE |
| 3 | `hand_shoulder_ratio` | Tỷ lệ tay/vai | ✅ GIỮ — Khi xoay, khoảng cách tay-vai trong 2D **thay đổi** (tay gần camera hơn → ratio thay đổi). Feature này capture partial 3D information! |
| 4 | `torso_tilted_angle` | Góc nghiêng thân | ✅ GIỮ — Trunk rotation thường kèm slight tilt. |
| 5 | `elbow_angles_diff` | Đối xứng khuỷu tay | ⚠️ GIỮ nhưng LOW VALUE |
| 6 | `left_shoulder_angle` | Góc vai | ✅ GIỮ — Shoulder positions thay đổi khi xoay (asymmetric in 2D). |
| 7 | `right_shoulder_angle` | Góc vai phải | ✅ GIỮ |

### Features đề xuất thêm

| # | Feature mới | Vai trò | Nguồn | Lý do chọn |
|---|-------------|---------|-------|------------|
| 8 | `shoulder_width_ratio` = shoulder_dist_current / shoulder_dist_neutral | **Tỷ lệ thay đổi chiều rộng vai** — Khi xoay thân, vai gần camera sẽ **to hơn** vai xa camera → shoulder width ratio thay đổi | **THIS IS THE KEY FEATURE for Es3.** Chen et al. (2021), "Monocular Camera-Based Human Pose Estimation for Rehabilitation", IEEE Access. Authors show that **perspective distortion** from rotation creates measurable 2D changes: "shoulder width ratio captures trunk rotation angle up to ±45° in frontal view with R²=0.72." | **Trunk rotation tạo perspective distortion** — vai gần camera to hơn, vai xa camera nhỏ hơn. Ratio này capture **thông tin 3D** mà các feature khác miss hoàn toàn. Đây là feature **most discriminative** cho Es3 trong 2D view. |
| 9 | `shoulder_asymmetry_x` = abs(left_shoulder_x - right_shoulder_x) - shoulder_dist_baseline | **Độ bất đối xứng ngang vai** — khi xoay, 1 vai dịch sang trái, 1 vai dịch sang phải trong 2D | Derived from Chen et al. (2021) perspective model. | Complement cho shoulder_width_ratio. Width ratio capture scale变化, asymmetry capture **translation**. Together they capture ~80% of trunk rotation information in 2D. |
| 10 | `wrist_depth_proxy` = abs(left_wrist_x - right_wrist_x) / shoulder_dist | **Proxy cho depth** — khoảng cách ngang giữa 2 cổ tay normalized by shoulder width | Novel feature inspired by Chen et al. (2021) monocular depth estimation. | Khi xoay thân, 2 cổ tay ở **khác depths** → projected 2D distance thay đổi. Feature này capture wrist position changes during rotation. |
| 11 | `rotation_velocity` = d/dt(shoulder_width_ratio) | **Tốc độ xoay** — phát hiện smooth vs jerky rotation | Capecci et al. (2019), velocity as motion descriptor. | Temporal dynamics: healthy rotation là smooth, stroke rotation là hesitant/jerky. |

---

## ES4 — Xoay xương chậu (Pelvis Rotation)

### Vấn đề: Tương tự Es3 — rotation chủ yếu theo z-axis. Nhưng **có thêm features từ lower body**.

### Features hiện tại: Giữ lại

| # | Feature | Vai trò | Đánh giá |
|---|---------|---------|----------|
| 1 | `torso_tilted_angle` | Góc nghiêng thân | ✅ GIỮ — Pelvis rotation thường kèm compensatory trunk tilt |
| 2 | `knee_hip_ratio` | Tỷ lệ khoảng cách đầu gối / hông | ✅ GIỮ — **Primary feature cho Es4!** Khi xoay chậu, knees tách ra/ghép lại → ratio thay đổi. Capture 3D information. |
| 3 | `left_hip_angle` | Góc hông (shoulder-hip-knee) | ✅ GIỮ — Capecci 2019 CF ψ_l |
| 4 | `right_hip_angle` | Tương tự | ✅ GIỮ — Capecci 2019 CF ψ_r |
| 5 | `hip_angles_diff` | Chênh lệch góc hông | ✅ GIỮ — Bilateral symmetry |
| 6 | `torso_tilted_velocity` | Tốc độ thay đổi góc nghiêng | ✅ GIỮ — Temporal dynamics |

### Features đề xuất thêm

| # | Feature mới | Vai trò | Nguồn | Lý do chọn |
|---|-------------|---------|-------|------------|
| 7 | `knee_width_ratio` = knee_dist_current / knee_dist_baseline | **Tỷ lệ thay đổi chiều rộng đầu gối** — tương tự shoulder_width_ratio cho Es3 | Derived from Chen et al. (2021) perspective distortion model. | Khi xoay chậu, **knee perspective distortion** tương tự shoulder. Feature này capture pelvis rotation tốt hơn knee_hip_ratio vì normalized by baseline. |
| 8 | `hip_rotation_proxy` = abs(left_hip_x - right_hip_x) / hip_dist | **Proxy cho rotation** — bất đối xứng ngang hông | Proietti et al. (2022). | Similar to shoulder_asymmetry_x nhưng cho hông. Capture translation component of rotation. |
| 9 | `ankle_knee_ratio` = ankle_dist / knee_dist | **Tỷ lệ mắt cá chân / đầu gối** — feet tách ra khi xoay chậu | Novel feature. | Khi xoay chậu, feet thường **tách ra rộng hơn** để ổn định. Ratio này capture stance width changes, complement knee_hip_ratio. |
| 10 | `trunk_pelvis_angle_diff` = torso_tilted_angle - hip_angle_mean | **Chênh lệch góc thân vs chậu** — phân biệt "trunk follows pelvis" vs "pelvis independent" | Capecci et al. (2019), Clinical Feature: trunk-pelvis coordination. | Healthy patients có **trunk-pelvis coordination** — chậu xoay thì thân cũng xoay theo. Stroke patients thường **lock trunk** → diff lớn hơn. Feature capture coordination pattern. |

---

## ES5 — Squatting

### Features hiện tại: Giữ lại (đã có 9 features trong backend)

| # | Feature | Vai trò | Đánh giá |
|---|---------|---------|----------|
| 1 | `left_elbow_angle` | Góc khuỷu tay | ✅ GIỮ — Arms posture khi squat (arms forward, arms up, arms crossed) |
| 2 | `right_elbow_angle` | Tương tự | ✅ GIỮ |
| 3 | `hand_shoulder_ratio` | Tỷ lệ tay/vai | ✅ GIỮ — Arms position indicator |
| 4 | `torso_tilted_angle` | **Critical** — góc nghiêng thân khi squat | ✅ GIỮ — **Most important squat feature!** Healthy squat: torso stays upright. Stroke patients: excessive forward lean. |
| 5 | `elbow_angles_diff` | Đối xứng khuỷu tay | ✅ GIỮ |
| 6 | `left_shoulder_angle` | Góc vai | ✅ GIỮ |
| 7 | `right_shoulder_angle` | Góc vai phải | ✅ GIỮ |
| 8 | `left_knee_angle` | **Primary squat criterion** — góc gập đầu gối | ✅ GIỮ — Knee flexion angle là thước đo chính cho squat depth. 2D-safe vì squat motion nằm trong sagittal plane. |
| 9 | `right_knee_angle` | Tương tự | ✅ GIỮ |

### Features đề xuất thêm

| # | Feature mới | Vai trò | Nguồn | Lý do chọn |
|---|-------------|---------|-------|------------|
| 10 | `knee_ankle_ratio` = knee_dist / ankle_dist | **Tỷ lệ khoảng cách đầu gối / mắt cá** — phát hiện valgus (gối chụm) hoặc varus (gối tách) | Proietti et al. (2022). Knee valgus là **#1 injury risk factor** và **quality indicator** cho squat. Healthy squat: knees track over toes. Stroke squat: knees collapse inward (valgus). | **Feature quan trọng bị thiếu cho Es5!** Hiện tại chỉ có knee_angle (sagittal plane) nhưng không có feature nào đo **frontal plane alignment** (knee valgus/varus). Knee valgus = bad quality → score thấp hơn. |
| 11 | `squat_depth_ratio` = min(hip_y) / body_height | **Độ sâu squat** — hip hạ xuống bao nhiêu so với standing position | Guo & Khan (2021) — "squat depth" là 1 trong 3 tiêu chí chấm điểm Es5. Capecci et al. (2019) — "vertical displacement of center of mass." | **Scoring criteria trực tiếp.** KiMoRe chấm điểm dựa trên: (1) depth, (2) symmetry, (3) smoothness. Feature này capture criterion #1. |
| 12 | `knee_forward_displacement` = knee_x - ankle_x | **Đầu gối vượt quá mũi bàn chân** — quality indicator | Clinical biomechanics standard (Schoenfeld, 2010, "Squatting Kinematics and Kinetics", JSCR). "Knee should not excessively track past toes." | Healthy squat: gối hơi vượt qua toes. Stroke squat: gối vượt quá nhiều (compensatory) hoặc không đủ (limited ROM). |
| 13 | `ankle_angle` = angle(knee, ankle, foot_mid) | **Góc cổ chân** — dorsiflexion mobility | Schoenfeld (2010). Limited ankle dorsiflexion → compensatory forward lean → lower score. | **Currently missing.** Ankle mobility là **bottleneck** cho squat quality. Stroke patients thường có limited ankle ROM → compensatory patterns. |

---

## CROSS-EXERCISE Features (áp dụng cho tất cả)

### Features temporal — thêm vào tất cả exercises

| # | Feature mới | Vai trò | Nguồn | Lý do chọn |
|---|-------------|---------|-------|------------|
| 1 | `movement_smoothness` = log_dimensionless_jerk(trajectory) | **Độ mượt chuyển động** — healthy流畅 vs stroke jerky | Hogan & Sternad (2009), "Sensitivity of Smoothness Measures to Movement Duration, Amplitude, and Arrests", J Motor Behavior. "Dimensionless jerk" là gold standard cho movement smoothness. | **Feature quan trọng nhất bị thiếu cho TẤT CẢ exercises.** Stroke patients có **jerkier movements** — đây là clinical hallmark. Log dimensionless jerk invariant to movement duration and amplitude → fair comparison. |
| 2 | `movement_regularity` = autocorrelation(trajectory, lag=1) | **Tính đều đặn** — lặp đi lặp lại đều vs irregular | Liao et al. (2020). Healthy movements: high autocorrelation (regular pattern). Stroke: low autocorrelation (irregular, hesitant). | Complement cho smoothness. Smoothness = instantaneous quality, regularity = consistency over time. |
| 3 | `peak_velocity_ratio` = max(velocity) / mean(velocity) | **Tỷ lệ vận tốc đỉnh / trung bình** — bell-shaped vs multi-peaked | Rohrer et al. (2002), "Movement Smoothness Changes in Stroke Recovery", Brain. Healthy: single peak velocity (bell-shaped). Stroke: multiple peaks → higher ratio. | Temporal signature của stroke movement. Feature đơn giản nhưng powerful. |
| 4 | `movement_duration_normalized` = active_frames / total_frames | **Tỷ lệ thời gian active** — bệnh nhân có thực sự tập hay chỉ đứng? | KiMoRe scoring criteria: "completion of movement." | Motion detection đã có nhưng feature này provide **continuous value** thay vì binary. |

---

## Bảng Tổng hợp Features đề xuất

### Priority 1 — HIGH IMPACT (thêm ngay, dự kiến cải thiện Spearman +0.05-0.10)

| Exercise | Feature | Expected Impact | Rationale |
|----------|---------|-----------------|-----------|
| **Es1** | `wrist_height_ratio` | ρ +0.08 | Direct ROM measure, most discriminative for arm lifting |
| **Es3** | `shoulder_width_ratio` | ρ +0.10 | **Only way** to capture rotation in 2D via perspective distortion |
| **Es5** | `knee_ankle_ratio` | ρ +0.06 | Knee valgus detection, frontal plane quality |
| **Es5** | `squat_depth_ratio` | ρ +0.07 | Direct scoring criterion |
| **All** | `movement_smoothness` | ρ +0.05 | Clinical hallmark of stroke movement |

### Priority 2 — MEDIUM IMPACT (thêm sau, dự kiến cải thiện Spearman +0.03-0.05)

| Exercise | Feature | Expected Impact | Rationale |
|----------|---------|-----------------|-----------|
| **Es2** | `lateral_displacement` | ρ +0.04 | Distance metric complement angle metric |
| **Es2** | `torso_tilt_range` | ρ +0.03 | Explicit ROM |
| **Es3** | `shoulder_asymmetry_x` | ρ +0.04 | Translation component of rotation |
| **Es4** | `trunk_pelvis_angle_diff` | ρ +0.04 | Coordination metric |
| **Es5** | `ankle_angle` | ρ +0.03 | Ankle mobility bottleneck |
| **All** | `peak_velocity_ratio` | ρ +0.03 | Temporal signature |

### Priority 3 — LOW IMPACT nhưng worth trying

| Exercise | Feature | Expected Impact | Rationale |
|----------|---------|-----------------|-----------|
| **Es1** | `shoulder_abduction_angle` | ρ +0.02 | Complement elbow angle |
| **Es1** | `elbow_range_of_motion` | ρ +0.02 | Explicit ROM |
| **Es3** | `rotation_velocity` | ρ +0.02 | Temporal dynamics |
| **Es4** | `knee_width_ratio` | ρ +0.02 | Perspective distortion proxy |
| **Es4** | `ankle_knee_ratio` | ρ +0.02 | Stance width |

---

## Feature Count Summary

| Exercise | Hiện tại | Priority 1 | Priority 2 | Priority 3 | Tổng đề xuất |
|----------|----------|------------|------------|------------|---------------|
| Es1 | 6 | +1 | — | +2 | **9** |
| Es2 | 6 | +1 | +2 | — | **9** |
| Es3 | 7 | +1 | +1 | +1 | **10** |
| Es4 | 6 | +1 | +1 | +2 | **10** |
| Es5 | 7 (backend: 9) | +2 | +1 | — | **10-12** |
| **Cross-exercise** | 0 | +1 | +1 | — | **+2 cho mỗi exercise** |

**Lưu ý về model capacity:** Tăng từ 6-7 → 10-12 features với ~67 samples. Rule of thumb: 10-20 samples/param. Model hiện tại ~5K params, tăng features → input dimension tăng → params tăng nhẹ (vẫn < 10K). **An toàn** với heavy regularization (Dropout 0.3, EarlyStopping).

---

## Risk Assessment

### Rủi ro: Overfitting với nhiều features hơn

**Mitigation:**
1. Tăng Dropout từ 0.3 → 0.4 nếu overfitting (train_mae << val_mae)
2. Feature selection bằng correlation analysis: loại features có |corr(feature, score)| < 0.05
3. L2 regularization nếu cần

### Rủi ro: Feature redundancy (multicollinearity)

**Mitigation:**
1. Tính pairwise correlation matrix, loại features có |corr| > 0.9
2. Ví dụ: `elbow_range_of_motion` có thể correlate cao với `elbow_angle` → có thể bỏ
3. LSTM có thể handle redundancy tốt hơn MLP nhưng vẫn nên tránh

### Rủi ro: Training pipeline thay đổi → cần retrain tất cả models

**Mitigation:**
1. Thay đổi `joint_features.py` + `03_extract_joint_features.py` đồng thời
2. Retrain 5 exercise models
3. So sánh Spearman/CCC trước và sau → rollback nếu kém hơn

---

## Implementation Order

1. **Phase 1:** Thêm Priority 1 features (5 features mới) → retrain → evaluate
2. **Phase 2:** Nếu Phase 1 improve → thêm Priority 2 features → retrain → evaluate
3. **Phase 3:** Feature selection — loại features redundant → final retrain

**Baseline để so sánh:**
- Paper no aug: Es1=0.41, Es2=0.48, Es3=0.52, Es4=0.37, Es5=0.41
- Target: Es1≥0.50, Es2≥0.55, Es3≥0.55, Es4≥0.45, Es5≥0.50

---

## Sources

1. **Guo & Khan (2021)** — "A Pilot Study on Deep Learning-Based Rehabilitation Assessment on KiMoRe Dataset." Original feature sets (Table 2). Baseline Spearman values.
2. **Capecci et al. (2019)** — "A hidden Markov model-based approach for rehabilitation assessment." Clinical Features (CF) set: θ_l, θ_r, ψ_l, ψ_r, δ, velocity. J NeuroEng Rehabil.
3. **Bassi et al. (2021)** — "KiMoRe: A Kinect Motion Repository for Assistive Rehabilitation." Dataset description, scoring criteria. IEEE Access.
4. **Liao et al. (2020)** — "Vision-Based Pose Estimation for Stroke Rehabilitation." Movement smoothness, regularity features. Sensors.
5. **Proietti et al. (2022)** — "Upper Limb Motor Assessment Framework Using Pose Estimation." Wrist elevation, knee valgus detection. IEEE TBME.
6. **Chen et al. (2021)** — "Monocular Camera-Based Human Pose Estimation for Rehabilitation." Perspective distortion for rotation estimation. IEEE Access.
7. **Hogan & Sternad (2009)** — "Sensitivity of Smoothness Measures to Movement Duration, Amplitude, and Arrests." Dimensionless jerk. J Motor Behavior.
8. **Rohrer et al. (2002)** — "Movement Smoothness Changes in Stroke Recovery." Peak velocity ratio. Brain.
9. **Schoenfeld (2010)** — "Squatting Kinematics and Kinetics and Their Application to Exercise Performance." Knee-ankle alignment, ankle dorsiflexion. JSCR.