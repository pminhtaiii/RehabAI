"""
Clinical Score Prediction Model v6 — Practical LSTM (Sigmoid + Huber)
=======================================================================
Evolved from v5 (paper-aligned) to practical scoring pipeline.

CHANGES FROM v5:
──────────────────────────────────────────────────────────────────────────
1. Architecture: 4×LSTM(16)+GAP → 2×LSTM(32/16)+Dropout(0.3)+Dense(8,relu)+Dense(1,sigmoid)
   - Sigmoid output bounds predictions to (0,1) → prevents collapse
   - Funnel shape (32→16) compresses temporal info naturally
   - Dropout 0.3 prevents overfitting on small dataset
   - return_sequences=False → uses last hidden state (better than GAP)

2. Loss: MAE+diversity → Huber (smooth near 0, robust at extremes)
   - Diversity penalty no longer needed (sigmoid prevents collapse)

3. LR: 0.0001 → 0.001 with ReduceLROnPlateau (patience=10, factor=0.5)
   - Higher LR enables faster convergence with Huber loss
   - Warmup still used but shorter (20 epochs)

4. y normalization: y/50 → output sigmoid (0,1) → ×50×2 = [0,100] at inference

Usage:
  python clinical_score_prediction_model.py --env local --exercise Es1 --base-dir C:/RehabAI
  python clinical_score_prediction_model.py --env colab --exercise Es1
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Masking, LSTM, Input, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='RehabAI Practical LSTM v6')
    p.add_argument('--env', choices=['colab', 'local'], default='colab')
    p.add_argument('--exercise', default='Es1',
                   choices=['Es1', 'Es2', 'Es3', 'Es4', 'Es5'])
    p.add_argument('--base-dir', default=None)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=16,
                   help='Batch size (default=16)')
    p.add_argument('--k-folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.001,
                   help='Learning rate (default=0.001, with ReduceLROnPlateau)')
    p.add_argument('--downsample', type=int, default=5,
                   help='Giữ 1 frame mỗi N frames (default=5: 25fps→5fps).')
    p.add_argument('--target-len', type=int, default=150,
                   help='Max sequence length sau khi downsample + truncate.')
    args, _ = p.parse_known_args()
    return args


# ── Paths ────────────────────────────────────────────────────────────────────

def get_paths(args):
    if args.base_dir:
        base = args.base_dir
    elif args.env == 'colab':
        base = '/content/drive/MyDrive/RehabAI'
    else:
        base = 'C:/RehabAI'
    return {
        'csv': f'{base}/05_final_datasets/KiMoRe_data_movenet_features.csv',
        'save': f'{base}/models/best_models',
        'plots': f'{base}/models/plots_v6',
    }


# ── Data Loading ──────────────────────────────────────────────────────────────
# MASK_VALUE dùng để pad sequences — phải là giá trị không xuất hiện trong data
# thực sau khi scale. Dùng -999.0 (an toàn vì StandardScaler cho data thực
# thường nằm trong [-5, 5]).
MASK_VALUE = -999.0


# ── Temporal Downsampling ─────────────────────────────────────────────────────

def temporal_downsample(feat_arr, stride, target_len):
    """Downsample + truncate một sequence (T, F) → (T', F).

    Lý do cần downsample:
      - Video 25fps, exercise ~27s → ~672 frames trung bình
      - LSTM 4×16 với 72 samples không thể học qua 672 timesteps
      - Frame liền kề gần như giống hệt nhau → redundant information
      - stride=5: 25fps → 5fps, capture được động tác đủ mượt

    Truncate strategy: giữ đầu + cuối, bỏ giữa nếu quá dài.
    Lý do: phần đầu (bắt đầu động tác) và cuối (kết thúc) thường
    quan trọng hơn phần giữa để đánh giá chất lượng.

    Args:
        feat_arr: (T, F) ndarray
        stride: int, giữ 1 frame mỗi `stride` frames
        target_len: int, max length sau khi downsample

    Returns:
        (T', F) với T' <= target_len
    """
    # Bước 1: stride downsampling
    downsampled = feat_arr[::stride]          # (T/stride, F)

    T = len(downsampled)
    if T <= target_len:
        return downsampled

    # Bước 2: nếu vẫn còn dài hơn target_len, giữ đầu + cuối
    half = target_len // 2
    head = downsampled[:half]
    tail = downsampled[T - (target_len - half):]
    return np.concatenate([head, tail], axis=0)   # (target_len, F)


def load_raw_features(csv_path, exercise, downsample_stride=5, target_len=150):
    """Load raw (unscaled) feature arrays cho một exercise.

    Args:
        csv_path: path đến KiMoRe_data_movenet_features.csv
        exercise: 'Es1'...'Es5'
        downsample_stride: giữ 1 frame mỗi N frames (default=5)
        target_len: max sequence length sau downsampling (default=150)

    Returns:
        raw_features: list of (T_i, F) numpy arrays, đã downsampled, unscaled
        y: (N,) array of clinical scores
    """
    df = pd.read_csv(csv_path)
    if exercise != 'All':
        df = df[df['exercise'] == exercise].reset_index(drop=True)

    df = df.dropna(subset=['clinical_score', 'joint_features']).reset_index(drop=True)
    print(f"Dataset: {len(df)} samples for {exercise} (after dropping NaN scores)")

    raw_features = []
    y_labels = []
    skipped = 0
    original_lengths = []
    downsampled_lengths = []

    for _, row in df.iterrows():
        fpath = row['joint_features']
        if pd.isna(fpath) or not os.path.exists(str(fpath)):
            skipped += 1
            continue
        try:
            feat_df = pd.read_csv(fpath)
            feat_arr = feat_df.to_numpy(dtype=np.float64)
        except Exception as e:
            print(f"  Skip {fpath}: {e}")
            skipped += 1
            continue

        if feat_arr.shape[0] < 5:
            skipped += 1
            continue

        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        original_lengths.append(len(feat_arr))

        # Temporal downsampling: giảm sequence length trước khi training
        feat_arr = temporal_downsample(feat_arr, downsample_stride, target_len)
        downsampled_lengths.append(len(feat_arr))

        raw_features.append(feat_arr.astype(np.float32))
        y_labels.append(float(row['clinical_score']))

    if skipped > 0:
        print(f"  Skipped: {skipped} samples")

    print(f"Original lengths:    min={min(original_lengths)}, "
          f"max={max(original_lengths)}, mean={np.mean(original_lengths):.0f}")
    print(f"Downsampled lengths: min={min(downsampled_lengths)}, "
          f"max={max(downsampled_lengths)}, mean={np.mean(downsampled_lengths):.0f} "
          f"(stride={downsample_stride}, target_len={target_len})")

    y = np.array(y_labels, dtype=np.float32)
    print(f"Loaded: {len(raw_features)} samples")
    print(f"  Score range: [{y.min():.1f}, {y.max():.1f}], "
          f"mean={y.mean():.1f}, std={y.std():.1f}")
    return raw_features, y


def scale_and_pad(raw_features_subset, scaler, max_length):
    """Scale features với scaler đã fit, sau đó pad/truncate thành (N, max_length, F).

    FIX: Scaler được truyền vào từ bên ngoài (fit trên train set chỉ),
    không load từ file global.

    Args:
        raw_features_subset: list of (T_i, F) arrays
        scaler: fitted StandardScaler (fit trên train set của fold hiện tại)
        max_length: int, độ dài pad

    Returns:
        X: (N, max_length, F) padded array với MASK_VALUE cho padding
    """
    num_features = raw_features_subset[0].shape[1]
    N = len(raw_features_subset)
    X = np.full((N, max_length, num_features), MASK_VALUE, dtype=np.float32)

    for i, feat in enumerate(raw_features_subset):
        # Scale
        scaled = scaler.transform(feat).astype(np.float32)
        # Truncate or pad
        T = min(len(scaled), max_length)
        X[i, :T, :] = scaled[:T]

    return X


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_features, max_length):
    """Practical LSTM v6: 2×LSTM(32/16) + Dropout + Dense(8) + sigmoid output.

    Architecture:
      - Input: (batch, max_length, num_features) with Masking
      - LSTM(32, return_sequences=True) → Dropout(0.3)
      - LSTM(16, return_sequences=False) → Dropout(0.3)
      - Dense(8, relu) → Dense(1, sigmoid)
      - Huber loss (smooth + robust)

    Sigmoid output:
      - Bounds predictions to (0, 1) → prevents collapse
      - Matches y/50 normalization naturally
      - No need for diversity penalty hack
    """
    inputs = Input(shape=(max_length, num_features), name='input')

    # Masking: skip padded timesteps
    x = Masking(mask_value=MASK_VALUE)(inputs)

    # 2× LSTM funnel: 32 → 16
    x = LSTM(32, return_sequences=True, name='lstm_1')(x)
    x = Dropout(0.3, name='drop_1')(x)

    x = LSTM(16, return_sequences=True, name='lstm_2')(x)
    x = Dropout(0.3, name='drop_2')(x)

    # Lấy trung bình toàn bộ quá trình tập (thay vì chỉ lấy frame cuối)
    x = GlobalAveragePooling1D(name='gap')(x)

    # Dense bottleneck + bounded output
    x = Dense(8, activation='relu', name='dense_1')(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipvalue=0.5,
        ),
        loss='huber',
        metrics=['mae'],
    )
    return model


# ── Training Loop ─────────────────────────────────────────────────────────────

def create_callbacks(lr_init):
    """Tạo callbacks: ReduceLROnPlateau + EarlyStopping."""
    return [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def train_fold(fold_model, X_train_pad, y_train, X_val_pad, y_val, args):
    """Train một fold với Keras .fit() API (batch_size=8, đúng paper).

    FIX so với v4: dùng model.fit() thay vì train_on_batch() để:
    - Gradient được tổng hợp trên batch đầy đủ (batch_size=8)
    - ReduceLROnPlateau và EarlyStopping hoạt động đúng
    - Training ổn định hơn
    """
    # LR warmup: start very small, ramp up slowly
    initial_lr = args.lr / 100.0  # much lower start (collapse fix)
    fold_model.optimizer.learning_rate.assign(initial_lr)

    callbacks = create_callbacks(args.lr)

    # Warmup callback
    class LRWarmup(tf.keras.callbacks.Callback):
        def __init__(self, target_lr, warmup_epochs=20):
            super().__init__()
            self.target_lr = target_lr
            self.warmup_epochs = warmup_epochs

        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.warmup_epochs:
                lr = initial_lr + (self.target_lr - initial_lr) * (epoch / self.warmup_epochs)
                self.model.optimizer.learning_rate.assign(lr)

    class PrintEpochs(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 0 or (epoch + 1) % 50 == 0:
                current_lr = float(self.model.optimizer.learning_rate)
                print(f"  Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, "
                      f"val_loss={logs.get('val_loss', 0):.4f}, "
                      f"lr={current_lr:.6f}")

    history = fold_model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[LRWarmup(args.lr), *callbacks, PrintEpochs()],
        verbose=0,
        shuffle=True,
    )
    return history


# ── Cross Validation ──────────────────────────────────────────────────────────

def cross_validate(raw_features, y, args, paths):
    """5-fold CV với scaler fit per-fold (FIX data leakage).

    Workflow per fold:
      1. Split: train_idx, val_idx
      2. Fit StandardScaler trên train_raw ONLY → không leakage
      3. Scale + pad cả train và val với scaler đó
      4. Build model + train
      5. Evaluate
    """
    exercise = args.exercise
    num_features = raw_features[0].shape[1]
    kfold = KFold(n_splits=args.k_folds, random_state=args.seed, shuffle=True)

    # Tính max_length từ toàn bộ dataset (ok vì không leak label info)
    all_lengths = [f.shape[0] for f in raw_features]
    # Dùng percentile 95 để tránh padding quá nhiều do outlier dài
    max_length = int(np.percentile(all_lengths, 95))
    print(f"\nMax sequence length (95th pct): {max_length}")

    all_true, all_pred = [], []
    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"  Cross Validation: {args.k_folds} folds, {exercise}")
    print(f"  Architecture: 2×LSTM(32/16) + Dropout(0.3) + Dense(8,relu) + Dense(1,sigmoid)")
    print(f"  Loss: Huber | Batch size: {args.batch_size} | Epochs: {args.epochs} | lr: {args.lr}")
    print(f"  FIX: Scaler fit per-fold (no data leakage)")
    print(f"{'='*60}")

    indices = np.arange(len(raw_features))

    for i, (train_idx, val_idx) in enumerate(kfold.split(indices), 1):
        print(f"\n── Fold {i}/{args.k_folds} ──")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train_raw = [raw_features[j] for j in train_idx]
        X_val_raw = [raw_features[j] for j in val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # ── FIX #1: Fit scaler CHỈ trên train fold ──────────────────────────
        # Concatenate tất cả train frames để fit scaler
        all_train_frames = np.concatenate(X_train_raw, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_train_frames)
        print(f"  Scaler fit on {len(all_train_frames)} train frames "
              f"(NOT on val/test — data leakage prevented)")

        # Kiểm tra: features có variance không?
        feat_std = all_train_frames.std(axis=0)
        low_var_feats = np.sum(feat_std < 0.01)
        if low_var_feats > 0:
            print(f"  ⚠ {low_var_feats}/{len(feat_std)} features have low variance "
                  f"(std < 0.01) — may not contribute to learning")

        # ── FIX #2: Pad sequences → enable batch training ────────────────────
        X_train_pad = scale_and_pad(X_train_raw, scaler, max_length)
        X_val_pad = scale_and_pad(X_val_raw, scaler, max_length)

        print(f"  X_train_pad shape: {X_train_pad.shape}")
        print(f"  X_val_pad shape:   {X_val_pad.shape}")

        # Kiểm tra target distribution trong fold này
        print(f"  y_train: mean={y_train.mean():.1f}, std={y_train.std():.1f}, "
              f"range=[{y_train.min():.1f}, {y_train.max():.1f}]")
        print(f"  y_val:   mean={y_val.mean():.1f}, std={y_val.std():.1f}, "
              f"range=[{y_val.min():.1f}, {y_val.max():.1f}]")

        # Cảnh báo nếu val set quá nhỏ hoặc target range quá hẹp
        if y_val.std() < 3.0:
            print(f"  ⚠ Val set target std={y_val.std():.1f} rất thấp — "
                  f"Spearman có thể không đáng tin cậy cho fold này")

        # ── Build + Train ────────────────────────────────────────────────────
        tf.random.set_seed(args.seed + i)
        np.random.seed(args.seed + i)

        fold_model = build_model(num_features, max_length)

        # Chuẩn hóa y về [0, 1] để tránh lỗi lệch thang đo (Scale Mismatch)
        # Không dùng Sigmoid để giữ nguyên Linear output đúng y như paper
        y_train_norm = y_train / 50.0
        y_val_norm = y_val / 50.0

        history = train_fold(fold_model, X_train_pad, y_train_norm,
                             X_val_pad, y_val_norm, args)

        # ── Evaluate ─────────────────────────────────────────────────────────
        # Sigmoid output → predictions already in (0, 1)
        train_pred = fold_model.predict(X_train_pad, verbose=0).flatten() * 50.0
        val_pred = fold_model.predict(X_val_pad, verbose=0).flatten() * 50.0

        # Clip to valid range
        val_pred = np.clip(val_pred, 0, 50)
        train_pred = np.clip(train_pred, 0, 50)

        fold_mae = mean_absolute_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)

        # Prediction range — diagnostic
        pred_range = val_pred.max() - val_pred.min()
        pred_std = val_pred.std()

        # Spearman
        try:
            spearman_rho, spearman_p = spearmanr(y_val, val_pred)
            if np.isnan(spearman_rho):
                spearman_rho, spearman_p = 0.0, 1.0
        except Exception:
            spearman_rho, spearman_p = 0.0, 1.0

        # Pearson
        try:
            pearson_r = pearsonr(y_val, val_pred)[0]
            if np.isnan(pearson_r):
                pearson_r = 0.0
        except Exception:
            pearson_r = 0.0

        print(f"  Train MAE: {train_mae:.2f} | Val MAE: {fold_mae:.2f} | "
              f"Gap: {fold_mae - train_mae:.2f}")
        print(f"  Spearman ρ: {spearman_rho:.3f} (p={spearman_p:.4f}) | "
              f"Pearson r: {pearson_r:.3f}")
        print(f"  Pred range: {pred_range:.2f} | Pred std: {pred_std:.2f} "
              f"{'⚠ COLLAPSE!' if pred_std < 0.5 else '✓'}")

        if pred_std < 0.5:
            print(f"  !! Model vẫn đang collapse. Thử:")
            print(f"     1. Kiểm tra features có variance trong tập train không")
            print(f"     2. Tăng diversity_weight (hiện: {args.diversity_weight})")
            print(f"     3. Kiểm tra clinical_score distribution")

        all_true.extend(y_val.tolist())
        all_pred.extend(val_pred.tolist())

        fold_metrics.append({
            'fold': i,
            'train_mae': float(train_mae),
            'val_mae': float(fold_mae),
            'gap': float(fold_mae - train_mae),
            'spearman_rho': float(spearman_rho),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pred_range': float(pred_range),
            'pred_std': float(pred_std),
            'epochs_trained': len(history.history['loss']),
        })

        # Save fold plots
        _save_fold_plots(history, train_pred, y_train, val_pred, y_val,
                         exercise, i, paths['plots'])

        # Lưu scaler của fold này (optional, cho inference)
        os.makedirs(paths['save'], exist_ok=True)

    # ── OOF Summary ──────────────────────────────────────────────────────────
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    oof_mae = mean_absolute_error(all_true, all_pred)
    oof_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    try:
        oof_spearman, oof_spearman_p = spearmanr(all_true, all_pred)
    except Exception:
        oof_spearman, oof_spearman_p = 0.0, 1.0
    try:
        oof_pearson = pearsonr(all_true, all_pred)[0]
    except Exception:
        oof_pearson = 0.0

    pred_std_oof = np.std(all_pred)

    print(f"\n{'='*60}")
    print(f"  OOF Results: {exercise}")
    print(f"  MAE:      {oof_mae:.2f}")
    print(f"  RMSE:     {oof_rmse:.2f}")
    print(f"  Spearman: {oof_spearman:.3f}  ← paper's primary metric")
    print(f"  Pearson:  {oof_pearson:.3f}")
    print(f"  Pred std: {pred_std_oof:.2f} (nên >> 0)")
    print(f"{'='*60}")

    # So sánh với paper baseline
    paper_no_aug = {'Es1': 0.41, 'Es2': 0.48, 'Es3': 0.52, 'Es4': 0.37, 'Es5': 0.41}
    paper_best = {'Es1': 0.76, 'Es2': 0.61, 'Es3': 0.73, 'Es4': 0.54, 'Es5': 0.67}
    if exercise in paper_no_aug:
        print(f"\n  Paper comparison ({exercise}):")
        print(f"    Paper (no aug):  ρ = {paper_no_aug[exercise]:.2f}")
        print(f"    Paper (best):    ρ = {paper_best[exercise]:.2f}")
        print(f"    Ours:            ρ = {oof_spearman:.3f}")
        if oof_spearman >= paper_no_aug[exercise]:
            print(f"    ✅ Đạt hoặc vượt baseline của paper (no aug)")
        elif oof_spearman > 0:
            print(f"    ⚠️  Dương nhưng chưa đạt baseline của paper")
        else:
            print(f"    ❌ Spearman âm — model vẫn có vấn đề, xem log từng fold")

    # Save scatter plot
    _save_oof_scatter(all_true, all_pred, exercise, oof_spearman, paths['plots'])

    return {
        'exercise': exercise,
        'max_length': max_length,
        'loss_type': 'huber',
        'oof_mae': float(oof_mae),
        'oof_rmse': float(oof_rmse),
        'oof_spearman': float(oof_spearman),
        'oof_pearson': float(oof_pearson),
        'pred_std': float(pred_std_oof),
        'folds': fold_metrics,
    }


# ── Final Model Training ──────────────────────────────────────────────────────

def train_final_model(raw_features, y, args, paths):
    """Train final model trên toàn bộ data, lưu model + scaler."""
    exercise = args.exercise
    num_features = raw_features[0].shape[1]

    # Max length
    all_lengths = [f.shape[0] for f in raw_features]
    max_length = int(np.percentile(all_lengths, 95))

    print(f"\n{'#'*60}")
    print(f"  Final Model Training: {exercise}")
    print(f"  All {len(raw_features)} samples")
    print(f"{'#'*60}")

    # Fit scaler trên toàn bộ data (cho production — không có test set ở đây)
    all_frames = np.concatenate(raw_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_frames)

    # Scale + pad
    X_pad = scale_and_pad(raw_features, scaler, max_length)

    # Split 15% for early stopping monitoring
    n_val = max(2, int(len(raw_features) * 0.15))
    perm = np.random.permutation(len(raw_features))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = X_pad[train_idx]
    X_val = X_pad[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Clip y to valid range for training stability
    y_train = np.clip(y_train, 0, 50)
    y_val = np.clip(y_val, 0, 50)

    # Chuẩn hóa y về [0, 1] để tránh lỗi lệch thang đo (Scale Mismatch)
    y_train_norm = y_train / 50.0
    y_val_norm = y_val / 50.0

    model = build_model(num_features, max_length)
    history = train_fold(model, X_train, y_train_norm, X_val, y_val_norm, args)

    model.summary()

    # Save
    os.makedirs(paths['save'], exist_ok=True)
    model_path = f"{paths['save']}/ml_model_{exercise}.keras"
    model.save(model_path)
    print(f"✅ Model saved: {model_path}")

    import joblib
    scaler_path = f"{paths['save']}/scaler_{exercise}.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    config = {
        'exercise': exercise,
        'max_length': max_length,
        'n_features': num_features,
        'mask_value': MASK_VALUE,
        'loss_type': 'huber',
        'architecture': '2xLSTM(32/16), dropout=0.3, GAP, Dense(8,relu), Dense(1,sigmoid), Huber',
        'version': 'v6_sigmoid_huber_gap',
        'y_normalization': 'y/50',
    }
    config_path = f"{paths['save']}/model_config_{exercise}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved: {config_path}")

    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_fold_plots(history, train_pred, train_y, val_pred, val_y,
                     exercise, fold, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{exercise} — Fold {fold} (Paper LSTM v5)', fontsize=14)

    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Val')
    axes[0, 0].set_title('MAE Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')

    axes[0, 1].plot(history.history.get('mae', history.history['loss']), label='Train MAE')
    axes[0, 1].plot(history.history.get('val_mae', history.history['val_loss']), label='Val MAE')
    axes[0, 1].set_title('MAE')
    axes[0, 1].legend()

    axes[1, 0].scatter(train_y, train_pred, alpha=0.5, s=30, label='Train', c='steelblue')
    axes[1, 0].scatter(val_y, val_pred, alpha=0.8, s=50, marker='x', label='Val', c='coral')
    lims = [0, 52]
    axes[1, 0].plot(lims, lims, '--', color='gray', alpha=0.5)
    axes[1, 0].set_title('Predicted vs Actual')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Actual score (0-50)')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_xlim(lims)

    residuals = val_pred - val_y
    axes[1, 1].hist(residuals, bins=min(15, len(val_pred)), edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Residuals (Val)')
    axes[1, 1].set_xlabel('Error')

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{exercise}_fold{fold}_v5.png', dpi=150)
    plt.close()


def _save_oof_scatter(all_true, all_pred, exercise, spearman, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_true, all_pred, alpha=0.6, s=40)
    ax.set_xlabel('Actual score (0-50)')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{exercise} — OOF predictions (ρ={spearman:.3f})')
    lims = [0, 52]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5)
    ax.set_xlim(lims)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{exercise}_oof_scatter_v6.png', dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    paths = get_paths(args)

    print(f"{'='*60}")
    print(f"  RehabAI Practical LSTM v6 (Sigmoid + Huber)")
    print(f"{'='*60}")
    print(f"Environment:      {args.env}")
    print(f"Exercise:         {args.exercise}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"CSV:              {paths['csv']}")

    # Load raw features
    raw_features, y = load_raw_features(paths['csv'], args.exercise)

    if len(raw_features) < args.k_folds:
        print(f"ERROR: Chỉ có {len(raw_features)} samples, cần ít nhất {args.k_folds}")
        return

    NUM_FEATURES = raw_features[0].shape[1]
    print(f"Features per frame: {NUM_FEATURES}")
    print(f"Samples: {len(raw_features)}")

    # Cross-validate
    cv_results = cross_validate(raw_features, y, args, paths)

    # Train final model
    final_model = train_final_model(raw_features, y, args, paths)

    # Save CV results
    os.makedirs(paths['plots'], exist_ok=True)
    results_path = f"{paths['plots']}/{args.exercise}_cv_results_v6.json"
    with open(results_path, 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    print(f"\n✅ CV results saved: {results_path}")

    print(f"\n{'='*60}")
    print(f"  Finished: {args.exercise}")
    print(f"  OOF Spearman ρ = {cv_results['oof_spearman']:.3f}")
    print(f"  OOF MAE:      {cv_results['oof_mae']:.2f}")
    print(f"  Pred std:     {cv_results['pred_std']:.3f} (target: >> 1.0)")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model: {paths['save']}/ml_model_{args.exercise}.keras")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
