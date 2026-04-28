"""
Clinical Score Prediction Model v2 — Robust LSTM
=================================================
Anti-overfitting training with time-series augmentation and temporal statistics.

Works on both Colab and Local:
  Colab:  !python clinical_score_prediction_model.py --env colab --exercise Es1
  Local:  python clinical_score_prediction_model.py --env local --exercise Es1 --base-dir C:/RehabAI
"""

import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Masking, LSTM, Input, BatchNormalization
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import pearsonr
import joblib


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='RehabAI Robust LSTM Trainer')
    p.add_argument('--env', choices=['colab', 'local'], default='colab')
    p.add_argument('--exercise', default='Es1',
                   choices=['Es1', 'Es2', 'Es3', 'Es4', 'Es5'])
    p.add_argument('--base-dir', default=None,
                   help='Override base directory (default: auto from --env)')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--k-folds', type=int, default=5)
    p.add_argument('--aug-factor', type=int, default=6,
                   help='Augmented copies per original sample')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-augment', action='store_true',
                   help='Disable augmentation (for ablation)')
    return p.parse_args()


# ── Configuration ─────────────────────────────────────────────────────────────

DOWNSAMPLE_FACTOR = 5

max_length_mapping = {
    "Es1": 301, "Es2": 326, "Es3": 297, "Es4": 398, "Es5": 204,
}

NUM_TEMPORAL_STATS = 5  # Must match joint_features.compute_temporal_statistics


def get_paths(args):
    if args.base_dir:
        base = args.base_dir
    elif args.env == 'colab':
        base = '/content/drive/MyDrive/RehabAI'
    else:
        base = 'C:/RehabAI'
    return {
        'csv': f'{base}/05_final_datasets/KiMoRe_final.csv',
        'save': f'{base}/models/best_models',
        'plots': f'{base}/models/plots_v2',
        'scalers': f'{base}/models/best_models',
    }


# ── Temporal Statistics ───────────────────────────────────────────────────────

def compute_temporal_statistics(features_array):
    """5 scalar stats summarizing temporal dynamics of movement."""
    arr = np.asarray(features_array, dtype=np.float64)
    T, F = arr.shape
    if T < 2:
        return np.zeros(5, dtype=np.float64)

    per_var = np.var(arr, axis=0)
    overall_variance = float(np.mean(per_var))

    per_rom = np.ptp(arr, axis=0)
    overall_rom = float(np.mean(per_rom))

    diffs = np.abs(np.diff(arr, axis=0))
    total_displacement = float(np.mean(diffs))

    autocorrs = []
    for f in range(F):
        sig = arr[:, f]
        if np.std(sig) > 1e-8:
            c = np.corrcoef(sig[:-1], sig[1:])[0, 1]
            if not np.isnan(c):
                autocorrs.append(c)
    smoothness = float(np.mean(autocorrs)) if autocorrs else 0.0

    frame_disp = np.mean(diffs, axis=1)
    thresh = np.median(frame_disp) * 0.5
    active_ratio = float(np.mean(frame_disp > thresh))

    return np.array([overall_variance, overall_rom, total_displacement,
                     smoothness, active_ratio], dtype=np.float64)


def append_temporal_stats(features_array):
    """(T, F) -> (T, F+5) with temporal stats as constant columns."""
    arr = np.asarray(features_array, dtype=np.float64)
    T = arr.shape[0]
    stats = compute_temporal_statistics(arr)
    return np.hstack([arr, np.tile(stats, (T, 1))])


# ── Data Augmentation (Time-Series) ──────────────────────────────────────────

def augment_gaussian_noise(X, sigma=0.05):
    return X + np.random.normal(0, sigma, X.shape)


def augment_magnitude_scaling(X, lo=0.85, hi=1.15):
    return X * np.random.uniform(lo, hi)


def augment_feature_dropout(X, rate=0.15):
    mask = np.random.binomial(1, 1 - rate, X.shape[1])
    return X * mask[np.newaxis, :]


def augment_temporal_jitter(X, max_shift=2):
    T, F = X.shape
    out = np.zeros_like(X)
    for t in range(T):
        src = min(max(t + np.random.randint(-max_shift, max_shift + 1), 0), T - 1)
        out[t] = X[src]
    return out


def augment_speed_variation(X, lo=0.8, hi=1.2):
    T, F = X.shape
    factor = np.random.uniform(lo, hi)
    new_T = max(int(T * factor), 2)
    old_idx = np.arange(T)
    new_idx = np.linspace(0, T - 1, new_T)
    out = np.zeros((new_T, F))
    for f in range(F):
        out[:, f] = interp1d(old_idx, X[:, f], kind='linear',
                             fill_value='extrapolate')(new_idx)
    if new_T > T:
        out = out[:T]
    elif new_T < T:
        out = np.vstack([out, np.zeros((T - new_T, F))])
    return out


def augment_time_warp(X, sigma=0.2, knots=4):
    T, F = X.shape
    orig = np.linspace(0, 1, knots + 2)
    warp = orig + np.random.normal(0, sigma, knots + 2)
    warp[0], warp[-1] = 0, 1
    warp = np.sort(warp)
    cs = CubicSpline(orig, warp)
    warped = np.clip(cs(np.linspace(0, 1, T)), 0, 1) * (T - 1)
    out = np.zeros_like(X)
    for f in range(F):
        out[:, f] = interp1d(np.arange(T), X[:, f], kind='linear',
                             fill_value='extrapolate')(warped)
    return out


AUGMENTATIONS = [
    augment_gaussian_noise,
    augment_magnitude_scaling,
    augment_feature_dropout,
    augment_temporal_jitter,
    augment_speed_variation,
    augment_time_warp,
]


def create_augmented_dataset(X_orig, y_orig, aug_factor=6):
    """Generate augmented samples. Temporal stats recomputed per augmented copy."""
    X_aug, y_aug = list(X_orig), list(y_orig)

    for i in range(len(X_orig)):
        sample = X_orig[i]  # (T, F_orig+5)
        F_orig = sample.shape[1] - NUM_TEMPORAL_STATS
        raw_features = sample[:, :F_orig]  # strip old temporal stats

        for _ in range(aug_factor):
            # Pick 2-3 random augmentations and chain them
            n_aug = np.random.randint(2, 4)
            chosen = np.random.choice(len(AUGMENTATIONS), n_aug, replace=False)
            aug = raw_features.copy()
            for idx in chosen:
                aug = AUGMENTATIONS[idx](aug)
            # Recompute temporal stats from augmented features
            aug_with_stats = append_temporal_stats(aug)
            # Add small label noise (±2 points) for regularization
            label_noise = np.random.uniform(-2, 2)
            noisy_label = np.clip(y_orig[i] + label_noise, 0, 100)
            X_aug.append(aug_with_stats)
            y_aug.append(noisy_label)

    return X_aug, np.array(y_aug, dtype=np.float32)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_and_prepare(csv_path, exercise, max_len):
    """Load joint features CSVs, append temporal stats, pad/truncate."""
    df = pd.read_csv(csv_path)
    if exercise != 'All':
        df = df[df['exercise'] == exercise].reset_index(drop=True)
    print(f"Dataset: {len(df)} rows for {exercise}")

    X_data, y_labels = [], []
    expected_cols = None

    for _, row in df.iterrows():
        fpath = row['joint_features']
        if pd.isna(fpath):
            continue
        try:
            feat_df = pd.read_csv(fpath)
            feat_arr = feat_df.to_numpy(dtype=np.float64)
        except Exception as e:
            print(f"  Skip {fpath}: {e}")
            continue

        if feat_arr.shape[0] == 0:
            continue

        # Downsample
        feat_arr = feat_arr[::DOWNSAMPLE_FACTOR]

        # Append temporal statistics (5 extra columns)
        feat_arr = append_temporal_stats(feat_arr)

        ncols = feat_arr.shape[1]
        if expected_cols is None:
            expected_cols = ncols
        elif ncols != expected_cols:
            print(f"  Skip {fpath}: cols {ncols} != expected {expected_cols}")
            continue

        # Truncate or pad
        T = feat_arr.shape[0]
        if T > max_len:
            feat_arr = feat_arr[:max_len]
        elif T < max_len:
            feat_arr = np.pad(feat_arr, ((0, max_len - T), (0, 0)),
                              mode='constant', constant_values=0.0)

        X_data.append(feat_arr)
        y_labels.append(row['clinical_score'])

    X = np.nan_to_num(np.array(X_data, dtype=np.float32))
    y = np.nan_to_num(np.array(y_labels, dtype=np.float32))
    print(f"Data: {X.shape}, Labels: {y.shape}")
    print(f"  Score range: [{y.min():.1f}, {y.max():.1f}], "
          f"mean={y.mean():.1f}, std={y.std():.1f}")
    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────

def build_robust_lstm(max_len, num_features, lr=0.001):
    """Regularized LSTM: smaller capacity, L2, recurrent dropout, linear output."""
    model = Sequential([
        Input(shape=(max_len, num_features)),
        Masking(mask_value=0.0),

        LSTM(64, return_sequences=True,
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01),
             dropout=0.3, recurrent_dropout=0.3),
        Dropout(0.4),

        LSTM(32, return_sequences=False,
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01),
             dropout=0.3, recurrent_dropout=0.3),
        Dropout(0.4),

        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1)  # Linear output — avoids sigmoid gradient compression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss='huber',
        metrics=['mae']
    )
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_fold_plots(history, train_pred, train_y, val_pred, val_y,
                    exercise, fold, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{exercise} — Fold {fold} (Robust LSTM v2)', fontsize=16)

    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Val')
    axes[0, 0].set_title('Huber Loss'); axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')

    axes[0, 1].plot(history.history['mae'], label='Train')
    axes[0, 1].plot(history.history['val_mae'], label='Val')
    axes[0, 1].set_title('MAE'); axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MAE')

    axes[1, 0].scatter(train_y, train_pred, alpha=0.5, s=30, label='Train')
    axes[1, 0].scatter(val_y, val_pred, alpha=0.7, s=50, marker='x', label='Val')
    lims = [0, 100]
    axes[1, 0].plot(lims, lims, '--', color='gray', alpha=0.5)
    axes[1, 0].set_title('Predicted vs Actual'); axes[1, 0].legend()
    axes[1, 0].set_xlabel('Actual'); axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_xlim(lims); axes[1, 0].set_ylim(lims)

    residuals = val_pred - val_y
    axes[1, 1].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Residuals (Val)'); axes[1, 1].set_xlabel('Error')

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{exercise}_fold{fold}_v2.png', dpi=200)
    plt.close()


# ── Cross Validation ──────────────────────────────────────────────────────────

class PrintEpochs(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch in [1, 25, 50, 75, 100, 125, 150]:
            vals = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            print(f"  Epoch {epoch}: {vals}")


def cross_validate(X, y, args, paths):
    """K-fold CV with per-fold augmentation (only on training split)."""
    exercise = args.exercise
    max_len = max_length_mapping[exercise]
    num_features = X.shape[2]
    kfold = KFold(n_splits=args.k_folds, random_state=args.seed, shuffle=True)

    all_true, all_pred = [], []
    fold_metrics = []
    print(f"\n{'='*60}")
    print(f"  Cross Validation: {args.k_folds} folds, {exercise}")
    print(f"  Augmentation: {'OFF' if args.no_augment else f'{args.aug_factor}x'}")
    print(f"{'='*60}")

    for i, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n── Fold {i}/{args.k_folds} ──")
        X_train_raw, X_val = X[train_idx], X[val_idx]
        y_train_raw, y_val = y[train_idx], y[val_idx]

        # Augment training data only
        if not args.no_augment:
            X_train_list, y_train = create_augmented_dataset(
                X_train_raw, y_train_raw, args.aug_factor)
            # Pad augmented samples to max_len (speed variation may change length)
            X_train_padded = []
            for s in X_train_list:
                s = np.asarray(s, dtype=np.float32)
                T = s.shape[0]
                if T > max_len:
                    s = s[:max_len]
                elif T < max_len:
                    s = np.pad(s, ((0, max_len - T), (0, 0)),
                               mode='constant', constant_values=0.0)
                X_train_padded.append(s)
            X_train = np.array(X_train_padded, dtype=np.float32)
            print(f"  Augmented: {len(X_train_raw)} → {len(X_train)} samples")
        else:
            X_train = X_train_raw
            y_train = y_train_raw

        # Normalize targets to [0,1]
        y_train_norm = y_train / 100.0
        y_val_norm = y_val / 100.0

        fold_model = build_robust_lstm(max_len, num_features, lr=0.001)

        history = fold_model.fit(
            X_train, y_train_norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val_norm),
            verbose=0,
            callbacks=[
                PrintEpochs(),
                EarlyStopping(monitor='val_mae', patience=20,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_mae', factor=0.5,
                                  patience=12, min_lr=1e-6, verbose=1),
            ]
        )

        train_pred = np.clip(fold_model.predict(X_train, verbose=0).flatten() * 100, 0, 100)
        val_pred = np.clip(fold_model.predict(X_val, verbose=0).flatten() * 100, 0, 100)

        fold_mae = mean_absolute_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        pred_range = val_pred.max() - val_pred.min()

        print(f"  Train MAE: {train_mae:.2f} | Val MAE: {fold_mae:.2f} | "
              f"Gap: {fold_mae - train_mae:.2f} | Pred range: {pred_range:.1f}")

        all_true.extend(y_val)
        all_pred.extend(val_pred)
        fold_metrics.append({
            'fold': i, 'train_mae': train_mae, 'val_mae': fold_mae,
            'gap': fold_mae - train_mae, 'pred_range': pred_range,
            'epochs': len(history.history['loss']),
        })

        save_fold_plots(history, train_pred, y_train, val_pred, y_val,
                        exercise, i, paths['plots'])

    # Overall metrics
    oof_mae = mean_absolute_error(all_true, all_pred)
    oof_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    try:
        oof_pearson = pearsonr(all_true, all_pred)[0]
    except Exception:
        oof_pearson = 0.0
    pred_std = np.std(all_pred)

    print(f"\n{'='*60}")
    print(f"  OOF Results: {exercise}")
    print(f"  MAE:     {oof_mae:.2f}")
    print(f"  RMSE:    {oof_rmse:.2f}")
    print(f"  Pearson: {oof_pearson:.3f}")
    print(f"  Pred std: {pred_std:.1f} (should be >> 0)")
    print(f"{'='*60}")

    # Discrimination check
    if pred_std < 3.0:
        print("  ⚠️  WARNING: Low prediction variance — model may still be collapsing to mean")
    elif oof_pearson < 0.3:
        print("  ⚠️  WARNING: Low correlation — predictions weakly related to actual scores")
    else:
        print("  ✅ Model shows meaningful discrimination")

    return {
        'exercise': exercise, 'oof_mae': oof_mae, 'oof_rmse': oof_rmse,
        'oof_pearson': oof_pearson, 'pred_std': pred_std,
        'folds': fold_metrics,
    }


# ── Final Training ────────────────────────────────────────────────────────────

def train_final_model(X, y, args, paths):
    """Train on ALL data with augmentation, save best model."""
    exercise = args.exercise
    max_len = max_length_mapping[exercise]
    num_features = X.shape[2]

    print(f"\n{'#'*60}")
    print(f"  Final Model Training: {exercise}")
    print(f"{'#'*60}")

    # Augment full dataset
    if not args.no_augment:
        X_list, y_aug = create_augmented_dataset(X, y, args.aug_factor)
        X_padded = []
        for s in X_list:
            s = np.asarray(s, dtype=np.float32)
            T = s.shape[0]
            if T > max_len:
                s = s[:max_len]
            elif T < max_len:
                s = np.pad(s, ((0, max_len - T), (0, 0)),
                           mode='constant', constant_values=0.0)
            X_padded.append(s)
        X_full = np.array(X_padded, dtype=np.float32)
        y_full = y_aug / 100.0
        print(f"  Augmented: {len(X)} → {len(X_full)} samples")
    else:
        X_full = X
        y_full = y / 100.0

    os.makedirs(paths['save'], exist_ok=True)
    checkpoint_path = f"{paths['save']}/ml_model_{exercise}_best.keras"

    model = build_robust_lstm(max_len, num_features, lr=0.001)

    history = model.fit(
        X_full, y_full,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_mae',
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=12,
                              min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_mae', patience=25,
                          restore_best_weights=True),
        ],
        verbose=1
    )

    model.summary()

    # Save final model
    save_path = f"{paths['save']}/ml_model_{exercise}.keras"
    model.save(save_path)
    print(f"✅ Model saved: {save_path}")

    # Plot final training
    os.makedirs(paths['plots'], exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{exercise} — Final Training (Robust LSTM v2)')
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(history.history['mae'], label='Train')
    axes[1].plot(history.history['val_mae'], label='Val')
    axes[1].set_title('MAE'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{paths['plots']}/{exercise}_final_v2.png", dpi=200)
    plt.close()

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    paths = get_paths(args)
    max_len = max_length_mapping[args.exercise]

    print(f"Environment: {args.env}")
    print(f"Exercise: {args.exercise} | Max length: {max_len}")
    print(f"CSV: {paths['csv']}")

    # Load data
    X, y = load_and_prepare(paths['csv'], args.exercise, max_len)
    NUM_FEATURES = X.shape[2]
    print(f"Features: {NUM_FEATURES} "
          f"(original + {NUM_TEMPORAL_STATS} temporal stats)")

    # Cross-validate
    cv_results = cross_validate(X, y, args, paths)

    # Train final model
    final_model = train_final_model(X, y, args, paths)

    # Save CV results
    os.makedirs(paths['plots'], exist_ok=True)
    with open(f"{paths['plots']}/{args.exercise}_cv_results.json", 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    print(f"\n✅ All done for {args.exercise}!")


if __name__ == '__main__':
    main()
