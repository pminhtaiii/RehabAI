# LSTM Architecture - RehabAI v6

## 📐 Model Structure

```
                    INPUT (150, F)
                         │
                    [Masking -999]
                         │
    ┌────────────────────────────────────┐
    │    LSTM Layer 1 (32 units)         │
    │    Output: (150, 32)               │
    │    → Capture raw patterns          │
    └────────────────────────────────────┘
                         │
                  [Dropout 0.3]
                         │
    ┌────────────────────────────────────┐
    │    LSTM Layer 2 (16 units)         │
    │    Output: (150, 16)               │
    │    → Funnel compression 32→16      │
    └────────────────────────────────────┘
                         │
                  [Dropout 0.3]
                         │
         [GlobalAveragePooling1D]
                   (16 features)
                         │
          [Dense 8 units, ReLU]
                         │
        [Dense 1 unit, Sigmoid]
                  Output ∈ [0, 1]
                         │
                    × 100
              CLINICAL SCORE [0-100]
```

---

## 📊 Layer Details

| # | Layer | Shape In | Shape Out | Params | Role |
|---|-------|----------|-----------|--------|------|
| 1 | Input | - | (150, F) | 0 | Raw features |
| 2 | Masking | (150, F) | (150, F) | 0 | Skip padding -999 |
| 3 | LSTM_1 | (150, F) | (150, 32) | 4,992 | Motion patterns |
| 4 | Dropout | (150, 32) | (150, 32) | 0 | Regularize |
| 5 | LSTM_2 | (150, 32) | (150, 16) | 3,072 | Compress → 16 |
| 6 | Dropout | (150, 16) | (150, 16) | 0 | Regularize |
| 7 | GAP | (150, 16) | (16,) | 0 | Average frames |
| 8 | Dense | (16,) | (8,) | 136 | Non-linear |
| 9 | Output | (8,) | (1,) | 9 | Bounded [0,1] |

**Total: 8,209 parameters** (114/sample for 72 samples)

---

## 🔄 Data Pipeline

```
Frontend
  30 frames @ 30fps (640×480)
      ↓
  MoveNet extract 17 keypoints
      ↓
Backend
  Extract features (2-9 per exercise)
      ↓
  Downsample stride=5
  (30 frames → 6 real frames)
      ↓
  StandardScaler.transform
  (per-exercise fit)
      ↓
  Pad to 150 with -999.0
      ├─ 6 real frames
      └─ 144 masked frames
      ↓
LSTM
  Process 6 real frames
  LSTM_1 → Dropout → LSTM_2 → Dropout
      ↓
  GlobalAveragePooling
  (average 150 timesteps)
      ↓
  Dense(8, relu)
      ↓
  Sigmoid [0, 1]
      ↓
  × 100
      ↓
Clinical Score [0-100]
```

---

## ⚙️ Configuration

**Architecture:**
- Funnel: 32 → 16 (hierarchical learning)
- Dropout: 0.3 (strong regularization for small dataset)
- Masking value: -999.0 (skip padded frames)

**Training:**
- Loss: Huber (robust to outliers)
- Optimizer: Adam (lr=0.001, clipvalue=0.5)
- Batch size: 16
- Epochs: 200 + early stopping
- K-fold: 5 (72 total samples)

**Per-Exercise Features:**
- Es1: 6 (shoulder flexion)
- Es2: 6 (elbow flexion)
- Es3: 9 (knee extension)
- Es4: 2 (ankle)
- Es5: 7 (wrist)

---

## 📈 Performance

```
Cross-Validation Results (5-fold):
  MAE:       8.0 ± 1.0
  Pearson r: 0.79 ± 0.03
  
Inference:
  Per batch: ~80ms
  Acceptable for real-time feedback
```

---

**Status**: ✅ Production Ready | **Version**: v6 | **Date**: May 4, 2026
