"""
LSTM Model — Quick Training Script
===================================
Converted from lstm_model.ipynb (user's reference).

Fixed bug: Original crashed with ValueError "inhomogeneous shape" when
EXERCISE='All' because different exercises have different feature column
counts. Fix: detect num_cols from first sample, then enforce it for all.

Usage (Colab):
    1. Mount Google Drive
    2. Set EXERCISE below
    3. Run: !python lstm_model.py
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, LSTM, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

EXERCISE = 'All'

BASE_DIR = '/content/drive/MyDrive/RehabAI'
CSV_PATH = f'{BASE_DIR}/05_final_datasets/KiMoRe_final.csv'
SAVE_DIR = f'{BASE_DIR}/models/best_models'

DOWNSAMPLE_FACTOR = 5  # Take every 5th frame (~6fps from 30fps video)

# Max frame counts per exercise (after 5x downsampling)
max_length_mapping = {
    "Es1": 301,   # ceil(1505/5)
    "Es2": 326,   # ceil(1629/5)
    "Es3": 297,   # ceil(1482/5)
    "Es4": 398,   # ceil(1988/5)
    "Es5": 204,   # ceil(1016/5)
}


def get_max_length(exercise):
    if exercise == 'All':
        return max(max_length_mapping.values())
    return max_length_mapping[exercise]


max_exercise_length = get_max_length(EXERCISE)


# ── Load Data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
if EXERCISE != 'All':
    df_exercise = df[df['exercise'] == EXERCISE].reset_index(drop=True)
else:
    df_exercise = df.copy()

print(f"Exercise: {EXERCISE} | Samples: {len(df_exercise)}")
print(df_exercise.head())


# ── Data Preparation ──────────────────────────────────────────────────────────

def prepare_lstm_data(df, max_len):
    """
    Load joint features, pad/truncate to max_len.

    FIX: When EXERCISE='All', different exercises may have different feature
    column counts. We detect num_cols from the first valid sample and then
    truncate or pad columns for subsequent samples to ensure a uniform shape.
    """
    X_data = []
    y_labels = []
    target_cols = None  # Will be set from first sample

    for _, row in df.iterrows():
        feature_path = row['joint_features']
        if pd.isna(feature_path):
            continue

        clinical_score = row['clinical_score']
        features_df = pd.read_csv(feature_path)
        features_array = features_df.to_numpy()

        # Detect or enforce column count
        if target_cols is None:
            target_cols = features_array.shape[1]
        elif features_array.shape[1] != target_cols:
            # Truncate extra columns or pad missing columns
            current_cols = features_array.shape[1]
            if current_cols > target_cols:
                features_array = features_array[:, :target_cols]
            else:
                col_pad = target_cols - current_cols
                features_array = np.pad(
                    features_array,
                    ((0, 0), (0, col_pad)),
                    mode='constant', constant_values=0.0
                )

        # Downsample: take every Nth frame to reduce sequence length
        features_array = features_array[::DOWNSAMPLE_FACTOR]

        # Truncate or pad rows to fixed length
        video_len = len(features_array)
        if video_len > max_len:
            features_array = features_array[:max_len]
        else:
            pad_len = max_len - video_len
            features_array = np.pad(
                features_array,
                ((0, pad_len), (0, 0)),
                mode='constant', constant_values=0.0
            )

        X_data.append(features_array)
        y_labels.append(clinical_score)

    X = np.nan_to_num(np.array(X_data))
    y = np.nan_to_num(np.array(y_labels))
    print(f"Data Shape: {X.shape} | Labels Shape: {y.shape}")
    return X, y


X, y = prepare_lstm_data(df_exercise, max_exercise_length)
NUM_FEATURES = X.shape[2]


# ── Build Model ──────────────────────────────────────────────────────────────

def build_lstm():
    model = Sequential([
        Input(shape=(max_exercise_length, NUM_FEATURES)),
        Masking(mask_value=0.0),

        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


model_lstm = build_lstm()
model_lstm.summary()


# ── Train Model ──────────────────────────────────────────────────────────────

history = model_lstm.fit(
    X, y,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    verbose=1
)


# ── Save Model ──────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
save_path = f'{SAVE_DIR}/LSTM_model_{EXERCISE}.keras'
model_lstm.save(save_path)
print(f"Model saved to: {save_path}")


# ── Train Larger Model with LR Scheduling ────────────────────────────────────

def build_lstm_v2():
    model = Sequential([
        Input(shape=(max_exercise_length, NUM_FEATURES)),
        Masking(mask_value=0.0),

        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


model_lstm_v2 = build_lstm_v2()
model_lstm_v2.summary()

lr_scheduler = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

history_v2 = model_lstm_v2.fit(
    X, y,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[lr_scheduler],
    verbose=1
)


# ── Plot Results ─────────────────────────────────────────────────────────────

hist_data = history_v2.history

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_data['mae'], label='Train MAE', color='blue', linewidth=2)
plt.plot(hist_data['val_mae'], label='Val MAE', color='red', linewidth=2)
plt.title('MAE', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(hist_data['loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(hist_data['val_loss'], label='Val Loss', color='red', linewidth=2)
plt.title('MSE Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Save v2 model
save_path_v2 = f'{SAVE_DIR}/LSTM_model_v2_{EXERCISE}.keras'
model_lstm_v2.save(save_path_v2)
print(f"Model v2 saved to: {save_path_v2}")
