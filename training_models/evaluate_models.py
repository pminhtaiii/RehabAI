"""
RehabAI Model Evaluation: Transformer vs LSTM
==============================================
Compares existing Transformer architecture against an LSTM baseline
using identical data splits, preprocessing, and evaluation metrics.

Usage (Colab):
    1. Mount Google Drive
    2. Set DATA_CSV_PATH to your KiMoRe CSV
    3. Run all cells

Usage (Local):
    python evaluate_models.py --exercise Es1 --data_csv /path/to/KiMoRe_data_movenet_features.csv
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MAX_LENGTH_MAPPING = {
    "Es1": 253, "Es2": 278, "Es3": 253, "Es4": 332, "Es5": 171,
}
TEMPORAL_WINDOWS_MAPPING = {
    "Es1": [5, 303], "Es2": [3, 556], "Es3": [11, 138],
    "Es4": [4, 497], "Es5": [7, 146],
}
SEED = 42
K_FOLDS = 5
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.001


# ──────────────────────────────────────────────
# Data Preparation (reused from notebook logic)
# ──────────────────────────────────────────────
def get_dataframe_cols():
    KEYPOINT_DICT = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    cols = []
    for name in KEYPOINT_DICT:
        cols.extend([f"{name}_y", f"{name}_x", f"{name}_confidence"])
    return cols


FACE_COLS = get_dataframe_cols()[:15]  # Drop nose, eyes, ears


def prepare_data(df, exercise, max_len):
    """Prepare data matching the final notebook pipeline."""
    data, labels, masks = [], [], []

    for _, row in df.iterrows():
        jp_path = row['joint_positions']
        if jp_path is np.nan or pd.isna(jp_path):
            continue

        clinical_score = row['clinical_score']
        video_length = math.ceil(row['#frames'] / 6.0)

        jp_data = pd.read_csv(jp_path)
        jp_data = jp_data.drop(FACE_COLS, axis=1, errors='ignore')
        jp_data = jp_data.to_numpy()[::6]  # Downsample 6x

        # Mid-shoulder normalization
        mid_y = (jp_data[:, 0:1] + jp_data[:, 3:4]) / 2.0
        mid_x = (jp_data[:, 1:2] + jp_data[:, 4:5]) / 2.0
        for i in range(0, jp_data.shape[1], 3):
            jp_data[:, i] -= mid_y[:, 0]
        for i in range(1, jp_data.shape[1], 3):
            jp_data[:, i] -= mid_x[:, 0]

        # Pad / truncate
        pad_len = max_len - video_length
        if pad_len < 0:
            jp_data = jp_data[:max_len]
            pad_len = 0
            video_length = max_len

        mask = np.ones(video_length + pad_len)
        if pad_len > 0:
            mask[-pad_len:] = 0

        padded = np.pad(jp_data, ((0, pad_len), (0, 0)), mode='constant')
        data.append(padded)
        labels.append(clinical_score)
        masks.append(mask)

    data = np.nan_to_num(np.array(data))
    labels = np.nan_to_num(np.array(labels))
    masks = np.array(masks)
    return data, labels, masks


# ──────────────────────────────────────────────
# Model Builders
# ──────────────────────────────────────────────
def build_transformer(input_shape, mask_shape, num_windows, window_size, d_model=9, num_heads=3):
    """Reproduce the final notebook Transformer architecture.
    Note: changed num_heads to 3 so d_model=9 is divisible (original used 4 which is invalid).
    """
    import tensorflow as tf
    import keras_nlp

    inputs = tf.keras.Input(shape=input_shape, name='original_data')
    mask_input = tf.keras.Input(shape=(mask_shape,), name='padding_masks')

    windows = tf.split(inputs, num_windows, axis=1)
    w_masks = tf.split(mask_input, num_windows, axis=1)

    emb1 = tf.keras.layers.Dense(d_model * 2, activation='relu')
    emb2 = tf.keras.layers.Dense(d_model, activation='relu')
    pos_emb = tf.keras.layers.Embedding(input_dim=window_size, output_dim=d_model)

    embeddings = []
    for w in windows:
        e = emb2(emb1(w))
        e = e + pos_emb(tf.range(window_size))
        embeddings.append(e)

    encoder = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=d_model, num_heads=num_heads
    )
    encoded = [encoder(emb, wm) for emb, wm in zip(embeddings, w_masks)]

    concat = tf.concat(encoded, axis=1)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(concat)

    from tensorflow.keras.regularizers import l2
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(pooled)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=[inputs, mask_input], outputs=output)
    return model


def build_lstm(input_shape):
    """LSTM baseline: simpler architecture with built-in sequential bias."""
    import tensorflow as tf
    from tensorflow.keras.regularizers import l2

    inputs = tf.keras.Input(shape=input_shape)
    # Masking layer handles padding (zeros) automatically
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)
    )(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


# ──────────────────────────────────────────────
# Evaluation Engine
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute all evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    pearson_r, pearson_p = pearsonr(y_true, y_pred) if len(y_true) > 2 else (float('nan'), 1.0)
    spearman_r, spearman_p = spearmanr(y_true, y_pred) if len(y_true) > 2 else (float('nan'), 1.0)
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'Pearson_r': round(pearson_r, 4),
        'Pearson_p': round(pearson_p, 6),
        'Spearman_r': round(spearman_r, 4),
        'Spearman_p': round(spearman_p, 6),
    }


def evaluate_model_cv(model_builder, data, labels, masks, model_type, exercise,
                       output_dir, k=K_FOLDS):
    """Run K-Fold CV and collect comprehensive metrics."""
    import tensorflow as tf

    kfold = KFold(n_splits=k, random_state=SEED, shuffle=True)
    fold_results = []
    all_val_true, all_val_pred = [], []
    all_train_true, all_train_pred = [], []

    for fold_i, (train_idx, val_idx) in enumerate(kfold.split(data), 1):
        print(f"\n{'='*50}")
        print(f"  {model_type} | {exercise} | Fold {fold_i}/{k}")
        print(f"{'='*50}")

        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        m_train, m_val = masks[train_idx], masks[val_idx]

        # Build fresh model each fold
        if model_type == 'Transformer':
            tw = TEMPORAL_WINDOWS_MAPPING[exercise]
            base = model_builder(
                input_shape=(data.shape[1], data.shape[2]),
                mask_shape=masks.shape[1],
                num_windows=tw[0], window_size=tw[1],
            )
            base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                         loss='mse', metrics=['mae'])
            train_inputs = [X_train, m_train]
            val_inputs = [X_val, m_val]
        else:  # LSTM
            base = model_builder(input_shape=(data.shape[1], data.shape[2]))
            base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                         loss='mse', metrics=['mae'])
            train_inputs = X_train
            val_inputs = X_val

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', patience=PATIENCE, restore_best_weights=True
        )

        history = base.fit(
            train_inputs, y_train,
            epochs=EPOCHS,
            validation_data=(val_inputs, y_val),
            verbose=0,
            callbacks=[early_stop]
        )

        # Predictions
        train_pred = base.predict(train_inputs, verbose=0).flatten()
        val_pred = base.predict(val_inputs, verbose=0).flatten()

        # Metrics
        train_metrics = compute_metrics(y_train, train_pred)
        val_metrics = compute_metrics(y_val, val_pred)

        gap = val_metrics['MAE'] - train_metrics['MAE']
        ratio = val_metrics['MAE'] / max(train_metrics['MAE'], 1e-8)

        fold_result = {
            'fold': fold_i,
            'train': train_metrics,
            'val': val_metrics,
            'generalization_gap': round(gap, 4),
            'overfitting_ratio': round(ratio, 4),
            'epochs_trained': len(history.history['loss']),
        }
        fold_results.append(fold_result)

        all_val_true.extend(y_val)
        all_val_pred.extend(val_pred)
        all_train_true.extend(y_train)
        all_train_pred.extend(train_pred)

        # Plot learning curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'{model_type} | {exercise} | Fold {fold_i}')

        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
        axes[0].legend(); axes[0].set_title('Loss')

        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
        axes[1].legend(); axes[1].set_title('MAE')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type}_{exercise}_fold{fold_i}_curves.png'), dpi=150)
        plt.close()

        print(f"  Train MAE: {train_metrics['MAE']:.4f} | Val MAE: {val_metrics['MAE']:.4f} | Gap: {gap:.4f} | Ratio: {ratio:.2f}")

    # Overall OOF metrics
    oof_metrics = compute_metrics(np.array(all_val_true), np.array(all_val_pred))
    oof_train_metrics = compute_metrics(np.array(all_train_true), np.array(all_train_pred))

    # Fold stability
    val_maes = [f['val']['MAE'] for f in fold_results]
    train_maes = [f['train']['MAE'] for f in fold_results]

    summary = {
        'model_type': model_type,
        'exercise': exercise,
        'n_samples': len(labels),
        'k_folds': k,
        'oof_val_metrics': oof_metrics,
        'oof_train_metrics': oof_train_metrics,
        'mean_val_mae': round(np.mean(val_maes), 4),
        'std_val_mae': round(np.std(val_maes), 4),
        'mean_train_mae': round(np.mean(train_maes), 4),
        'mean_generalization_gap': round(np.mean([f['generalization_gap'] for f in fold_results]), 4),
        'mean_overfitting_ratio': round(np.mean([f['overfitting_ratio'] for f in fold_results]), 4),
        'fold_results': fold_results,
    }

    # Overfitting diagnosis
    if summary['mean_overfitting_ratio'] > 2.0:
        summary['overfitting_diagnosis'] = 'SEVERE — model is heavily overfitting'
    elif summary['mean_overfitting_ratio'] > 1.5:
        summary['overfitting_diagnosis'] = 'MODERATE — noticeable overfitting'
    elif summary['mean_generalization_gap'] > 3.0:
        summary['overfitting_diagnosis'] = 'MILD — some overfitting detected'
    else:
        summary['overfitting_diagnosis'] = 'OK — generalization looks reasonable'

    return summary


def print_comparison(transformer_results, lstm_results):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON: Transformer vs LSTM")
    print("=" * 70)

    metrics = ['MAE', 'RMSE', 'R2', 'Pearson_r', 'Spearman_r']
    header = f"{'Metric':<20} {'Transformer':>15} {'LSTM':>15} {'Winner':>12}"
    print(header)
    print("-" * 62)

    for m in metrics:
        t_val = transformer_results['oof_val_metrics'][m]
        l_val = lstm_results['oof_val_metrics'][m]
        # For MAE/RMSE lower is better, for R2/correlations higher is better
        if m in ('MAE', 'RMSE'):
            winner = 'Transformer' if t_val < l_val else 'LSTM'
        else:
            winner = 'Transformer' if t_val > l_val else 'LSTM'
        print(f"{m:<20} {t_val:>15.4f} {l_val:>15.4f} {winner:>12}")

    print("-" * 62)
    print(f"{'Overfit Ratio':<20} {transformer_results['mean_overfitting_ratio']:>15.2f} {lstm_results['mean_overfitting_ratio']:>15.2f}")
    print(f"{'Gen. Gap (MAE)':<20} {transformer_results['mean_generalization_gap']:>15.4f} {lstm_results['mean_generalization_gap']:>15.4f}")
    print(f"{'Val MAE Std':<20} {transformer_results['std_val_mae']:>15.4f} {lstm_results['std_val_mae']:>15.4f}")
    print(f"\nTransformer diagnosis: {transformer_results['overfitting_diagnosis']}")
    print(f"LSTM diagnosis:       {lstm_results['overfitting_diagnosis']}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(exercise='Es1', data_csv=None):
    import tensorflow as tf
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if data_csv is None:
        # Default Colab path
        data_csv = "/content/drive/MyDrive/rehab-ai-data/KiMoRe_final/KiMoRe_data_movenet_features.csv"

    print(f"Loading data from: {data_csv}")
    df = pd.read_csv(data_csv)
    if exercise != "All":
        df = df[df['exercise'] == exercise]

    max_len = MAX_LENGTH_MAPPING[exercise]
    print(f"Exercise: {exercise} | Max length: {max_len}")

    data, labels, masks = prepare_data(df, exercise, max_len)
    print(f"Data shape: {data.shape} | Labels: {labels.shape} | Masks: {masks.shape}")

    if len(labels) < K_FOLDS * 2:
        print(f"ERROR: Only {len(labels)} samples — too few for {K_FOLDS}-fold CV")
        return

    # Evaluate Transformer
    print("\n\n" + "#" * 60)
    print("  EVALUATING: Transformer")
    print("#" * 60)
    transformer_results = evaluate_model_cv(
        build_transformer, data, labels, masks,
        'Transformer', exercise, output_dir
    )

    # Evaluate LSTM
    print("\n\n" + "#" * 60)
    print("  EVALUATING: LSTM")
    print("#" * 60)
    lstm_results = evaluate_model_cv(
        build_lstm, data, labels, masks,
        'LSTM', exercise, output_dir
    )

    # Comparison
    print_comparison(transformer_results, lstm_results)

    # Save results
    results = {
        'transformer': transformer_results,
        'lstm': lstm_results,
    }
    results_path = os.path.join(output_dir, f'comparison_{exercise}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RehabAI Model Evaluation')
    parser.add_argument('--exercise', type=str, default='Es1', choices=['Es1','Es2','Es3','Es4','Es5'])
    parser.add_argument('--data_csv', type=str, default=None)
    args = parser.parse_args()
    main(exercise=args.exercise, data_csv=args.data_csv)
