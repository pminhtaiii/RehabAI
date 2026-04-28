"""
Hyperparameter Tuning (LSTM) — Reference Only
==============================================
Converted from hyperparameter_tuning.ipynb.
This was used in early phases of the project and is added for reference only.
Changed from Transformer to LSTM architecture.

Usage (Colab):
    1. Mount Google Drive
    2. Update CSV_PATH below
    3. Run: !python hyperparameter_tuning.py
"""

import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, LSTM, Input

from sklearn.model_selection import train_test_split

# pip install keras-tuner
import keras_tuner


# ── Load and Prepare Data ────────────────────────────────────────────────────

CSV_PATH = "/content/drive/MyDrive/RehabAI/05_final_datasets/KiMoRe_final.csv"

df = pd.read_csv(CSV_PATH)

EXERCISE = "Es4"
df = df[df['exercise'] == EXERCISE].reset_index(drop=True)
print(df)

DOWNSAMPLE_FACTOR = 5  # Take every 5th frame (~6fps from 30fps video)

# Max frame counts per exercise (after 5x downsampling)
max_length_mapping = {
    "Es1": 301,   # ceil(1505/5)
    "Es2": 326,   # ceil(1629/5)
    "Es3": 297,   # ceil(1482/5)
    "Es4": 398,   # ceil(1988/5)
    "Es5": 204,   # ceil(1016/5)
}
EXERCISE_VIDEO_MAX_LEN = max_length_mapping[EXERCISE]
print(f"Maximum video length: {EXERCISE_VIDEO_MAX_LEN}")


def prepare_data(df, max_len, data_type):
    """Load joint features, pad/truncate to max_len."""
    X_data = []
    y_labels = []

    for _, row in df.iterrows():
        feature_path = row['joint_features']
        if pd.isna(feature_path):
            continue

        clinical_score = row['clinical_score']
        features_df = pd.read_csv(feature_path)
        features_array = features_df.to_numpy()

        # Downsample: take every Nth frame to reduce sequence length
        features_array = features_array[::DOWNSAMPLE_FACTOR]

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

    print(f"{data_type} Data Shape: {X.shape}")
    print(f"{data_type} Labels Shape: {y.shape}")
    return X, y


train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_X, train_y = prepare_data(train_df, EXERCISE_VIDEO_MAX_LEN, "Train")
test_X, test_y = prepare_data(test_df, EXERCISE_VIDEO_MAX_LEN, "Test")

NUM_FEATURES = train_X.shape[2]


# ── Build Tunable LSTM Model ─────────────────────────────────────────────────

def build_model(hp):
    """LSTM model with tunable hyperparameters."""
    lstm_units_1 = hp.Int("lstm_units_1", min_value=32, max_value=256, step=32)
    lstm_units_2 = hp.Int("lstm_units_2", min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
    )
    dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)
    learning_rate = hp.Float(
        "lr", min_value=1e-4, max_value=1e-2, sampling="log"
    )

    model = Sequential([
        Input(shape=(EXERCISE_VIDEO_MAX_LEN, NUM_FEATURES)),
        Masking(mask_value=0.0),

        LSTM(lstm_units_1, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units_2, return_sequences=False),
        Dropout(dropout_rate),

        Dense(dense_units, activation='relu'),
    ])

    # Optional extra dense layers
    for i in range(hp.Int("num_extra_dense", 0, 2)):
        extra_units = hp.Int(
            f"extra_dense_{i}", min_value=16, max_value=64, step=16
        )
        model.add(Dense(extra_units, activation='relu'))
        if hp.Boolean(f"extra_dropout_{i}"):
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


# ── Run Tuner ─────────────────────────────────────────────────────────────────

TUNER_DIR = "/content/drive/MyDrive/RehabAI/models/hyperparameter-tuning"

tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_mae",
    max_trials=40,
    overwrite=True,
    directory=TUNER_DIR,
    project_name=f"LSTM_{EXERCISE}_tuning",
)

tuner.search_space_summary()

tuner.search(
    train_X, train_y,
    epochs=20,
    validation_data=(test_X, test_y),
    batch_size=8,
)

tuner.results_summary()
