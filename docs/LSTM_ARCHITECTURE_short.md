# LSTM Architecture - RehabAI

## 🏗️ Model

```
Input → Masking(-999) → LSTM(32) → Dropout(0.3) → LSTM(16) → Dropout(0.3) → GAP → Dense(8, relu) → Output(Sigmoid)
```

## 📊 Layers

| Layer | Config | Params |
|-------|--------|--------|
| Masking | mask=-999.0 | 0 |
| LSTM_1 | 32 units, return_seq=True | 4,992 |
| Dropout_1 | rate=0.3 | 0 |
| LSTM_2 | 16 units, return_seq=True | 3,072 |
| Dropout_2 | rate=0.3 | 0 |
| GlobalAvgPool | - | 0 |
| Dense | 8 units, relu | 136 |
| Output | 1 unit, sigmoid | 9 |
| **TOTAL** | - | **8,209** |

## ⚙️ Training

```
Loss:      Huber (robust to outliers)
Optimizer: Adam (lr=0.001, clipvalue=0.5)
Batch:     16
Epochs:    200 with early stopping
K-fold:    5 (72 samples)
Dropout:   0.3 (strong regularization)
Padding:   -999.0
Downsample: stride=5 (30 frames → 6 real)
```

## 🔄 Forward Pass

```
30 frames (30fps)
  ↓ Extract features (2-9 per exercise)
Downsample stride=5 → 6 frames
  ↓ StandardScaler.transform
Pad to 150 with -999.0
  ↓ LSTM(32) processes 6 real, skips 144 masked
Dropout(0.3)
  ↓ LSTM(16) compression
Dropout(0.3)
  ↓ GlobalAvgPool: average 16 features
Dense(8, relu)
  ↓ Non-linear combinations
Sigmoid [0,1]
  ↓ × 100
Clinical Score [0-100]
```

## 🎯 Why These Choices?

**32→16 Funnel**
- Large enough to capture patterns
- Small enough to prevent overfitting (72 samples)
- Hierarchical: raw patterns → filtered features

**Dropout 0.3**
- Balanced regularization for small dataset
- Too low (0.1): overfitting
- Too high (0.5): underfitting

**GlobalAveragePooling**
- Considers full sequence (not just last frame)
- Smooth motion = consistent high average
- Jerky motion = inconsistent average

**Sigmoid + ×100**
- Sigmoid: bounded [0,1]
- ×100: natural clinical score [0-100]
- ReLU/Linear: unbounded (impossible scores)

## 📈 Performance

```
K-fold CV:
  MAE:        8.0 ± 1.0
  Pearson r:  0.79 ± 0.03
  
Acceptable for 72 samples
Error margin ±8 on [0-100] = reasonable
```

---

**Status**: ✅ Production Ready
**Parameters**: 8,209 | **Sample Size**: 72
**Loss**: Huber | **Inference Speed**: ~80ms
