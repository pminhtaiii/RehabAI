# LSTM Architecture - RehabAI

## 🏗️ Model Diagram

```
Input (150, F)
    ↓
Masking(mask=-999)
    ↓
LSTM_1(32, return_seq=True)
    ↓
Dropout(0.3)
    ↓
LSTM_2(16, return_seq=True)
    ↓
Dropout(0.3)
    ↓
GlobalAveragePooling1D
    ↓
Dense(8, relu)
    ↓
Output(1, sigmoid) → Clinical Score × 100
```

---

## � Parameters

| Layer | Input | Output | Params |
|-------|-------|--------|--------|
| Masking | (150, F) | (150, F) | 0 |
| LSTM_1 | (150, F) | (150, 32) | 4,992 |
| Dropout_1 | (150, 32) | (150, 32) | 0 |
| LSTM_2 | (150, 32) | (150, 16) | 3,072 |
| Dropout_2 | (150, 16) | (150, 16) | 0 |
| GlobalAvgPool | (150, 16) | (16) | 0 |
| Dense | (16) | (8) | 136 |
| Output | (8) | (1) | 9 |
| **TOTAL** | - | - | **~8,200** |

For 72 training samples: 8,200/72 = 114 params/sample (good!)

---

## � Forward Pass Example (Es1)

```
30 frames (640×480, 30fps)
    ↓ Extract 6 features (angles, ratios)
Downsample stride=5: 30 → 6 frames
    ↓ StandardScaler.transform (scaler_Es1.joblib)
Normalize features
    ↓ Pad to 150: 6 real + 144 × (-999.0)
    ↓ LSTM processes 6 real frames (144 masked out)
LSTM_1(32): Raw motion patterns
    ↓ Dropout(0.3)
LSTM_2(16): Compressed features
    ↓ Dropout(0.3)
GlobalAvgPool: Average 16 features = [f1_avg, ..., f16_avg]
    ↓ Dense(8, relu)
Non-linear combinations
    ↓ Sigmoid: output ∈ [0, 1]
    ↓ × 100
Clinical Score: [0-100]
```

---

**Model Summary**: 8,200 params, Huber loss, Adam optimizer, dropout 0.3
**Created**: May 4, 2026
