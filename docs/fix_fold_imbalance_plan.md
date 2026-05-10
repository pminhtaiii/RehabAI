# Fix Fold Score Distribution Imbalance in Cross-Validation

## Vấn đề (Problem)

Kết quả hiện tại cho thấy **Es2/Es4 vượt baseline** nhưng **Es1/Es3 yếu** dưới baseline, với dấu hiệu near-collapse ở một số fold. Nguyên nhân gốc: **phân bố score không đều giữa các fold** khi split theo subject.

### Tại sao xảy ra?

**Root cause: Subject-level grouping + skewed score distribution**

Dataset KiMoRe có ~67 subjects, mỗi subject có 1 clinical score cho mỗi exercise. Vấn đề nằm ở:

1. **Score concentration**: Trong KiMoRe, ~65% subjects là healthy (score 44-50). Chỉ ~35% subjects có motor dysfunction (score thấp hơn, range 10-40). Điều này tạo ra distribution **rất skew** sang high-end.

2. **`StratifiedGroupKFold` với 3 bins không đủ granular**: Hiện tại `make_score_bins(y, n_bins=3)` dùng `pd.qcut` với 3 bins. Với distribution skewed, bin cao nhất chứa quá nhiều subjects (vì ~65% subjects có score tương tự), dẫn đến:
   - Một fold có thể nhận **hầu hết subjects điểm thấp** → train set thiếu low-score diversity → model không học được gradient đủ
   - Fold khác nhận toàn subjects điểm cao → val set có std rất thấp → Spearman correlation không đáng tin

3. **Es1/Es3 đặc biệt skewed**: Es2/Es4/Es5 có score distribution rải đều hơn. Es1 và Es3 có score tập trung rất cao (gần 46-50) cho healthy subjects → khi 1 fold nhận hết nhóm low-score, kết quả collapse.

### Bằng chứng

> Es1: `y_val std < 3.0` ở một số fold → Spearman meaningless  
> Es3: Fold variance lớn (0.449 vs 0.370) → distribution khác nhau giữa các seed  
> Es2/Es4: Score rải đều hơn → StratifiedGroupKFold hoạt động tốt

### Trích nguồn

- **scikit-learn docs** ([StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)): Sử dụng greedy algorithm để assign groups to folds, nhưng với dataset nhỏ + skewed, algorithm không đảm bảo perfect stratification.
- **Abedi, Malmirian & Khan (2023)** ([arXiv:2306.09546](https://arxiv.org/abs/2306.09546)): KiMoRe 5-fold CV, augmentation chỉ trên train fold, nhưng **không address fold imbalance** explicitly.
- **General ML best practice** ([insightful-data-lab](https://insightful-data-lab.com), [Kaggle](https://kaggle.com)): Khi StratifiedGroupKFold collapse, ưu tiên: (1) custom greedy assignment, (2) tăng n_bins, (3) repeated CV across seeds.

---

## Proposed Changes

### 1. Custom `BalancedStratifiedGroupKFold` splitter

Thay vì dựa hoàn toàn vào scikit-learn `StratifiedGroupKFold`, implement custom splitter:

#### [MODIFY] [clinical_score_prediction_model.py](file:///c:/RehabAI/training_models/clinical_score_prediction_model.py)

**Thêm class `BalancedStratifiedGroupKFold`:**
- Greedy assignment: sort groups by score → xen kẽ assign vào fold (round-robin theo score rank)
- Sau khi assign xong, check balance: tính mean/std score mỗi fold, nếu KL-divergence giữa fold distributions > threshold → shuffle lại
- Guarantee: **mỗi fold chứa ít nhất 1 subject từ mỗi score bin**

**Algorithm:**
```
1. Bin scores thành 5 quantile bins (thay vì 3)
2. Group subjects theo bins → tạo "subject pools" per bin
3. Round-robin assign subjects từ mỗi pool → fold 1, 2, 3, 4, 5 xen kẽ
4. Validate: tính std(mean_score per fold) → nếu > threshold → re-shuffle
5. Output: fold assignments đảm bảo balanced score distribution
```

### 2. Tăng n_bins từ 3 → 5

**Lý do**: 3 bins quá thô cho score range 0-50 với 67 subjects. 5 bins (mỗi bin ~13 subjects) cho granularity tốt hơn mà vẫn đủ subjects per bin cho 5-fold CV.

**Sửa**: `make_score_bins(y, n_bins=5)` + fallback logic khi bins sparse

### 3. Repeated CV (3 repeats × 5 folds = 15 evaluations)

**Lý do**: Với N=67 subjects, **một lần 5-fold CV có variance quá cao** (SE ≈ 0.15 cho Spearman). Repeated CV across multiple seeds giảm variance đáng kể.

**Thêm parameter**: `--n-repeats` (default=3)

**Aggregation**:
- OOF metrics = mean across repeats
- Report std across repeats để gauge stability
- Nếu std(Spearman) across repeats > 0.15 → cảnh báo high variance

### 4. Per-fold diagnostic logging nâng cao

**Thêm fold quality checks:**
- Wasserstein distance giữa train/val score distribution
- KL divergence estimate
- Fold skip logic: nếu val set có `std(y) < 2.0` → log warning + exclude fold from OOF aggregation (nhưng vẫn report)

### 5. Fold quality gating

Nếu val set quá imbalanced (e.g., tất cả scores > 40), fold đó produce NaN/meaningless Spearman. Thay vì include vào OOF:
- **Gate**: exclude fold metrics nếu `y_val.std() < 2.0` hoặc `|y_val.mean() - y_overall.mean()| > 1.5 * y_overall.std()`
- **Report**: vẫn log fold result nhưng đánh dấu `⚠ EXCLUDED from OOF` 

---

## User Review Required

> [!IMPORTANT]
> **Repeated CV sẽ tăng training time ~3x** (3 repeats × 5 folds = 15 model trainings thay vì 5). Với 200 epochs/fold, estimate thêm ~10-15 phút trên GPU.

> [!WARNING]  
> **Fold quality gating** loại bỏ fold imbalanced khỏi OOF aggregation. Điều này giúp metric chính xác hơn nhưng **giảm sample size** cho OOF evaluation. Với Es1/Es3 (N≈67), loại 1 fold → OOF chỉ còn ~54 samples. Bạn có muốn giữ logic này hay chỉ log warning mà không exclude?

## Open Questions

> [!IMPORTANT]
> 1. **n_repeats = 3 hay 5?** 3 repeats là compromise tốt giữa stability và time. 5 repeats cho kết quả ổn hơn nhưng tốn ~5x thay vì 3x.
> 2. **Fold gating threshold**: Hiện đề xuất `y_val.std() < 2.0`. Bạn có muốn adjust threshold này?

---

## Tóm tắt thay đổi

| File | Thay đổi |
|------|----------|
| `clinical_score_prediction_model.py` | + Custom `BalancedStratifiedGroupKFold` class |
| | + `make_score_bins()`: n_bins 3→5 |  
| | + `cross_validate()`: repeated CV + fold quality gating |
| | + Per-fold balance diagnostics |
| | + `--n-repeats` CLI argument |

## Verification Plan

### Automated Tests
1. Chạy training cho Es1 với cả hai strategies (old vs new), so sánh:
   - Per-fold `y_val.mean()` và `y_val.std()` → new strategy phải balanced hơn
   - OOF Spearman stability (std across repeats)
2. In fold distribution diagnostics, verify mỗi fold chứa subjects từ >= 3/5 bins

### Manual Verification
- So sánh OOF Spearman Es1/Es3 trước và sau fix
- Kiểm tra fold plots: scatter plots không còn cluster ở 1 góc
