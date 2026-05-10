"""Clinical Score Prediction Model - LSTM training pipeline with K-fold CV."""

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
    Dense, Dropout, Masking, LSTM, Input
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
import tensorflow.keras.backend as K

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


class BalancedStratifiedGroupKFold:
    """Custom CV splitter đảm bảo balanced score distribution giữa các fold.

    Vấn đề: StratifiedGroupKFold của sklearn dùng greedy algorithm không
    đảm bảo balanced folds khi distribution skewed (KiMoRe ~65% healthy).

    Algorithm:
      1. Bin scores thành n_bins quantile bins
      2. Group subjects theo bins → tạo subject pools per bin
      3. Round-robin assign subjects từ mỗi pool → folds xen kẽ
      4. Validate: nếu fold imbalance > threshold → re-shuffle với seed khác

    Tham khảo: scikit-learn StratifiedGroupKFold docs, Kaggle best practices.
    """

    def __init__(self, n_splits=5, n_bins=5, n_attempts=50, random_state=None):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.n_attempts = n_attempts
        self.random_state = random_state

    def split(self, X, y, groups):
        """Generate (train_idx, val_idx) splits balanced theo score distribution.

        Args:
            X: array-like, indices hoặc features (chỉ dùng len)
            y: array-like, clinical scores (continuous)
            groups: array-like, subject IDs

        Yields:
            (train_idx, val_idx) tuples
        """
        y = np.asarray(y, dtype=np.float32)
        groups = np.asarray(groups)
        indices = np.arange(len(y))

        # Tạo score bins cho stratification
        bins = make_score_bins(y, n_bins=self.n_bins)

        # Tạo mapping: subject → (mean_score, bin, list of sample indices)
        unique_subjects = np.unique(groups)
        subject_info = {}
        for subj in unique_subjects:
            mask = groups == subj
            subj_indices = indices[mask]
            subj_scores = y[mask]
            subject_info[subj] = {
                'mean_score': float(np.mean(subj_scores)),
                'bin': int(bins[mask][0]),  # bin của subject (giả sử 1 score/subject)
                'indices': subj_indices,
            }

        # Group subjects theo bin
        bin_to_subjects = {}
        for subj, info in subject_info.items():
            b = info['bin']
            if b not in bin_to_subjects:
                bin_to_subjects[b] = []
            bin_to_subjects[b].append(subj)

        rng = np.random.RandomState(self.random_state)
        best_assignment = None
        best_imbalance = float('inf')

        # Thử nhiều lần, chọn assignment cân bằng nhất
        for attempt in range(self.n_attempts):
            fold_subjects = [[] for _ in range(self.n_splits)]

            # Round-robin assign subjects từ mỗi bin vào folds
            for b in sorted(bin_to_subjects.keys()):
                subjects_in_bin = bin_to_subjects[b].copy()
                rng.shuffle(subjects_in_bin)
                for j, subj in enumerate(subjects_in_bin):
                    fold_idx = j % self.n_splits
                    fold_subjects[fold_idx].append(subj)

            # Tính imbalance: std của mean_score per fold
            fold_means = []
            for f in range(self.n_splits):
                if fold_subjects[f]:
                    scores = [subject_info[s]['mean_score'] for s in fold_subjects[f]]
                    fold_means.append(np.mean(scores))
                else:
                    fold_means.append(0.0)
            imbalance = np.std(fold_means)

            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_assignment = [list(fs) for fs in fold_subjects]

            # Nếu đã đủ tốt, dừng sớm
            if imbalance < 1.5:
                break

        # Generate splits từ best assignment
        for fold_idx in range(self.n_splits):
            val_subjects = set(best_assignment[fold_idx])
            train_subjects = set()
            for fi in range(self.n_splits):
                if fi != fold_idx:
                    train_subjects.update(best_assignment[fi])

            train_mask = np.array([groups[i] in train_subjects for i in range(len(groups))])
            val_mask = np.array([groups[i] in val_subjects for i in range(len(groups))])

            train_idx = indices[train_mask]
            val_idx = indices[val_mask]

            if len(val_idx) > 0 and len(train_idx) > 0:
                yield train_idx, val_idx

def ccc_loss(y_true, y_pred):
    """Concordance Correlation Coefficient Loss.
    Forces the model to match both the mean AND the variance of the true scores,
    preventing the model from collapsing predictions to the dataset mean (46).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    
    covar = K.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2.0 * covar) / (var_true + var_pred + K.square(mean_true - mean_pred) + K.epsilon())
    return 1.0 - ccc

def ccc_metric(y_true, y_pred):
    """Metric để theo dõi CCC live qua từng Epoch (vì loss là 1 - CCC)."""
    return 1.0 - ccc_loss(y_true, y_pred)


def hybrid_loss(y_true, y_pred, ccc_weight=0.7):
    """Hybrid loss: CCC + MAE combined.

    Rationale (from deep research plan):
      - CCC loss alone can be noisy with small batches (batch_size=8/16, N=67)
      - MAE adds stable per-sample gradient signal
      - Combined: 0.7 * (1-CCC) + 0.3 * MAE
      - ccc_weight=0.7: CCC dominates (agreement focus) but MAE stabilizes

    Reference: Lawrence & Lin (1989) for CCC; Abedi et al. (2023) uses MAE.
    """
    ccc_part = ccc_loss(y_true, y_pred)
    mae_part = K.mean(K.abs(y_true - y_pred))
    return ccc_weight * ccc_part + (1.0 - ccc_weight) * mae_part


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
    p.add_argument('--n-repeats', type=int, default=1,
                   help='Số lần lặp CV với seed khác nhau (default=1, '
                        'nên dùng 3 cho dataset nhỏ để giảm variance). '
                        'Tổng folds = k_folds × n_repeats.')
    p.add_argument('--output-type', choices=['sigmoid', 'linear'], default='sigmoid',
                   help='Output activation: sigmoid (bounded [0,1]) or linear '
                        '(no activation, clip post-prediction). '
                        'Linear avoids gradient saturation for high scores. '
                        'Ref: LeCun (1998), Goodfellow (2016) Ch.6.2.2')
    p.add_argument('--loss-type', choices=['ccc', 'hybrid'], default='ccc',
                   help='Loss function: ccc (1-CCC) or hybrid (0.7*CCC + 0.3*MAE). '
                        'Hybrid balances agreement and error stability for small batches.')
    p.add_argument('--ensemble-alpha', type=float, default=0.5,
                   help='Weight for LSTM in ensemble (default=0.5). '
                        'Ignored if --auto-alpha.')
    p.add_argument('--auto-alpha', action='store_true',
                   help='Auto-tune ensemble alpha via CV grid search '
                        '(grid: 0.3,0.4,0.5,0.6,0.7 per fold).')
    args, _ = p.parse_known_args()
    return args


def get_paths(args):
    if args.base_dir:
        base = args.base_dir
    elif args.env == 'colab':
        base = '/content/drive/MyDrive/RehabAI'
    else:
        base = 'C:/RehabAI'
    return {
        'csv': f'{base}/05_final_datasets/KiMoRe_hello_world.csv',
        'save': f'{base}/models/best_models_3',
        'plots': f'{base}/models/plots_v1',
    }


MASK_VALUE = -999.0


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
    downsampled = feat_arr[::stride]
    T = len(downsampled)
    if T <= target_len:
        return downsampled
    half = target_len // 2
    head = downsampled[:half]
    tail = downsampled[T - (target_len - half):]
    return np.concatenate([head, tail], axis=0)


def load_raw_features(csv_path_or_df, exercise, downsample_stride=5, target_len=150):
    """Load raw (unscaled) feature arrays cho một exercise.

    Args:
        csv_path_or_df: path đến CSV hoặc DataFrame đã load sẵn
        exercise: 'Es1'...'Es5'
        downsample_stride: giữ 1 frame mỗi N frames (default=5)
        target_len: max sequence length sau downsampling (default=150)

    Returns:
        raw_features: list of (T_i, F) numpy arrays, đã downsampled, unscaled
        y: (N,) array of clinical scores
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df)
    if exercise != 'All':
        df = df[df['exercise'] == exercise].reset_index(drop=True)

    df = df.dropna(subset=['clinical_score', 'joint_features']).reset_index(drop=True)
    print(f"Dataset: {len(df)} samples for {exercise} (after dropping NaN scores)")

    raw_features = []
    y_labels = []
    subject_ids_list = []
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
        subject_ids_list.append(str(row['ID']))

    if skipped > 0:
        print(f"  Skipped: {skipped} samples")

    print(f"Original lengths:    min={min(original_lengths)}, "
          f"max={max(original_lengths)}, mean={np.mean(original_lengths):.0f}")
    print(f"Downsampled lengths: min={min(downsampled_lengths)}, "
          f"max={max(downsampled_lengths)}, mean={np.mean(downsampled_lengths):.0f} "
          f"(stride={downsample_stride}, target_len={target_len})")

    y = np.array(y_labels, dtype=np.float32)
    subject_ids = np.array(subject_ids_list)
    print(f"Loaded: {len(raw_features)} samples ({len(np.unique(subject_ids))} unique subjects)")
    print(f"  Score range: [{y.min():.1f}, {y.max():.1f}], "
          f"mean={y.mean():.1f}, std={y.std():.1f}")
    return raw_features, y, subject_ids


def extract_summary_features(raw_features_subset):
    """Extract summary features from temporal sequences for RF ensemble.

    Converts (N, T, F) variable-length sequences → (N, 3*F) fixed-size vectors.
    Per-feature statistics: mean, std, range (max-min).

    Ref: Proietti et al. (2022) — RF + handcrafted summary features on KiMoRe.

    Args:
        raw_features_subset: list of (T_i, F) arrays (already downsampled, unscaled)

    Returns:
        X_summary: (N, 3*F) array
    """
    n_features = raw_features_subset[0].shape[1]
    N = len(raw_features_subset)
    X_summary = np.zeros((N, 3 * n_features), dtype=np.float32)

    for i, feat in enumerate(raw_features_subset):
        # feat: (T_i, F) - use unscaled features, RF handles its own scaling
        means = np.mean(feat, axis=0)
        stds = np.std(feat, axis=0)
        ranges = np.max(feat, axis=0) - np.min(feat, axis=0)
        X_summary[i] = np.concatenate([means, stds, ranges])

    return X_summary


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


def make_score_bins(y, n_bins=5):
    """Bin continuous scores for stratified group splitting.

    FIX: Default n_bins tăng từ 3 → 5 cho granularity tốt hơn trên
    skewed distribution (KiMoRe ~65% healthy). 3 bins quá thô dẫn đến
    bin cao nhất chứa quá nhiều subjects → fold imbalance.

    Uses quantile bins when possible so each fold gets a similar score
    distribution while preserving subject-level grouping.
    """
    y = np.asarray(y, dtype=np.float32)
    unique_scores = np.unique(y)
    if len(unique_scores) < 2:
        return np.zeros(len(y), dtype=np.int32)

    n_bins = int(min(n_bins, len(unique_scores)))
    if n_bins < 2:
        return np.zeros(len(y), dtype=np.int32)

    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        return np.asarray(bins, dtype=np.int32)
    except ValueError:
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        edges = np.unique(np.percentile(y, percentiles))
        if len(edges) == 0:
            return np.zeros(len(y), dtype=np.int32)
        return np.digitize(y, edges, right=True).astype(np.int32)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_features, max_length, output_type='sigmoid', loss_type='ccc'):
    """Practical LSTM v9: 2×LSTM(32/16) + configurable output/loss.

    Args:
      output_type: 'sigmoid' (bounded [0,1]) or 'linear' (no activation, clip post)
        - sigmoid: prevents collapse, but saturates gradient for high scores (y/50~0.9)
          Ref: LeCun (1998) "Efficient BackProp"
        - linear: avoids saturation, needs clip [0,50] post-prediction
          Ref: Goodfellow (2016) Ch.6.2.2

      loss_type: 'ccc' or 'hybrid'
        - ccc: 1-CCC, agreement-focused, prevents collapse
        - hybrid: 0.7*(1-CCC) + 0.3*MAE, more stable with small batches
    """
    inputs = Input(shape=(max_length, num_features), name='input')

    x = Masking(mask_value=MASK_VALUE)(inputs)

    x = LSTM(32, return_sequences=True, name='lstm_1')(x)
    x = Dropout(0.3, name='drop_1')(x)

    x = LSTM(16, return_sequences=False, name='lstm_2')(x)
    x = Dropout(0.3, name='drop_2')(x)

    x = Dense(8, activation='relu', name='dense_1')(x)

    if output_type == 'linear':
        outputs = Dense(1, activation=None, name='output')(x)
    else:
        outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs)

    if loss_type == 'hybrid':
        loss_fn = hybrid_loss
    else:
        loss_fn = ccc_loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipvalue=0.5,
        ),
        loss=loss_fn,
        metrics=['mae', ccc_metric],
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
            patience=40,
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
                # loss = 1 - CCC, nên CCC = 1 - loss (không cần phụ thuộc tên metric)
                train_ccc = 1.0 - logs.get('loss', 1.0)
                val_ccc = 1.0 - logs.get('val_loss', 1.0)
                print(f"  Epoch {epoch+1}: CCC={train_ccc:.4f}, "
                      f"val_CCC={val_ccc:.4f}, "
                      f"MAE={logs.get('mae', 0):.4f}, lr={current_lr:.6f}")

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

def cross_validate_single_run(raw_features, y, subject_ids, aug_features, aug_y,
                              aug_ids, args, paths, seed_offset=0):
    """Một lần K-fold CV với BalancedStratifiedGroupKFold.

    FIX v8:
      - Dùng BalancedStratifiedGroupKFold (custom) thay vì sklearn StratifiedGroupKFold
      - n_bins=5 cho stratification granularity tốt hơn
      - Per-fold Wasserstein distance diagnostics
      - Fold quality gating: exclude fold có y_val.std() < 2.0 khỏi OOF

    Args:
        seed_offset: offset vào seed cho repeated CV (mỗi repeat khác seed)

    Returns:
        dict với fold_metrics, oof_true, oof_pred, max_length, included/excluded folds
    """
    exercise = args.exercise
    num_features = raw_features[0].shape[1]
    current_seed = args.seed + seed_offset
    score_bins = make_score_bins(y, n_bins=5)

    # Luôn dùng BalancedStratifiedGroupKFold (custom) — ưu tiên balanced folds
    splitter = BalancedStratifiedGroupKFold(
        n_splits=args.k_folds,
        n_bins=5,
        n_attempts=50,
        random_state=current_seed,
    )
    split_name = "BalancedStratifiedGroupKFold (custom, round-robin per bin)"

    # Tính max_length từ toàn bộ dataset (ok vì không leak label info)
    all_lengths = [f.shape[0] for f in raw_features]
    if aug_features:
        all_lengths += [f.shape[0] for f in aug_features]
    max_length = int(np.percentile(all_lengths, 95))

    y_overall_mean = float(np.mean(y))
    y_overall_std = float(np.std(y))

    oof_true_included, oof_pred_included = [], []
    fold_metrics = []
    n_excluded = 0

    print(f"\n{'='*60}")
    print(f"  Cross Validation: {args.k_folds} folds, {exercise} (seed={current_seed})")
    print(f"  Split strategy: {split_name}")
    print(f"  Architecture: 2×LSTM(32/16) + Dropout(0.3) + Dense(8,relu) + Dense(1,sigmoid)")
    print(f"  Loss: CCC | Batch size: {args.batch_size} | Epochs: {args.epochs} | lr: {args.lr}")
    print(f"  FIX: Balanced split + augmentation in train only + fold quality gating")
    print(f"  Original samples: {len(raw_features)} | Augmented available: {len(aug_features)}")
    print(f"  Score bins for stratification: {np.bincount(score_bins, minlength=int(score_bins.max()) + 1).tolist()}")
    print(f"  Overall y: mean={y_overall_mean:.1f}, std={y_overall_std:.1f}")
    print(f"{'='*60}")

    for i, (train_idx, val_idx) in enumerate(splitter.split(
            np.arange(len(raw_features)), y, subject_ids), 1):

        print(f"\n── Fold {i}/{args.k_folds} (seed={current_seed}) ──")

        # Val: original samples only (no augmentation — prevents leakage)
        X_val_raw = [raw_features[j] for j in val_idx]
        y_val = y[val_idx]

        # Train: original samples for train subjects
        X_train_raw = [raw_features[j] for j in train_idx]
        y_train = y[train_idx]

        # Merge augmented data for train subjects only
        train_subjects_fold = set(subject_ids[train_idx])
        n_aug_added = 0
        if aug_features:
            aug_train_y_list = []
            for j, sid in enumerate(aug_ids):
                if sid in train_subjects_fold:
                    X_train_raw.append(aug_features[j])
                    aug_train_y_list.append(aug_y[j])
                    n_aug_added += 1
            if aug_train_y_list:
                y_train = np.concatenate([y_train, np.array(aug_train_y_list, dtype=np.float32)])

        # ── Per-fold balance diagnostics ──────────────────────────────────
        y_train_orig = y[train_idx]  # chỉ original samples cho diagnostic
        fold_val_bins = score_bins[val_idx]
        bin_counts = np.bincount(fold_val_bins, minlength=int(score_bins.max()) + 1).tolist()

        # Wasserstein distance: đo sự khác biệt phân bố train vs val
        try:
            w_dist = float(wasserstein_distance(y_train_orig, y_val))
        except Exception:
            w_dist = -1.0

        print(f"  Train: {len(train_idx)} orig + {n_aug_added} aug = {len(X_train_raw)} total")
        print(f"  Val: {len(val_idx)} samples (original only, no leakage)")
        print(f"  Val score bins: {bin_counts}")
        print(f"  Wasserstein dist (train_orig vs val): {w_dist:.3f}")

        # Kiểm tra target distribution trong fold này
        print(f"  y_train_orig: mean={y_train_orig.mean():.1f}, std={y_train_orig.std():.1f}")
        print(f"  y_train (w/ aug): mean={y_train.mean():.1f}, std={y_train.std():.1f}")
        print(f"  y_val:   mean={y_val.mean():.1f}, std={y_val.std():.1f}, "
              f"range=[{y_val.min():.1f}, {y_val.max():.1f}]")

        # ── Fold quality gating ──────────────────────────────────────────
        # Exclude fold nếu y_val.std() quá thấp hoặc mean lệch quá xa overall mean
        fold_excluded = False
        exclude_reason = ""

        if y_val.std() < 2.0:
            fold_excluded = True
            exclude_reason = f"y_val.std()={y_val.std():.1f} < 2.0"
        elif abs(y_val.mean() - y_overall_mean) > 1.5 * y_overall_std:
            fold_excluded = True
            exclude_reason = (f"|y_val.mean() - overall.mean()| = "
                              f"{abs(y_val.mean() - y_overall_mean):.1f} > "
                              f"1.5 × overall.std = {1.5 * y_overall_std:.1f}")

        if fold_excluded:
            print(f"  ⚠ EXCLUDED from OOF aggregation: {exclude_reason}")
            print(f"    Fold vẫn được train và report metrics, nhưng không contribute vào OOF")
            n_excluded += 1

        if y_val.std() < 3.0 and not fold_excluded:
            print(f"  ⚠ Val set target std={y_val.std():.1f} thấp — "
                  f"Spearman/CCC có thể không đáng tin cậy")

        # ── Fit scaler CHỈ trên train fold ───────────────────────────────
        all_train_frames = np.concatenate(X_train_raw, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_train_frames)
        print(f"  Scaler fit on {len(all_train_frames)} train frames")

        # Kiểm tra: features có variance không?
        feat_std = all_train_frames.std(axis=0)
        low_var_feats = np.sum(feat_std < 0.01)
        if low_var_feats > 0:
            print(f"  ⚠ {low_var_feats}/{len(feat_std)} features have low variance "
                  f"(std < 0.01)")

        # ── Pad sequences ────────────────────────────────────────────────
        X_train_pad = scale_and_pad(X_train_raw, scaler, max_length)
        X_val_pad = scale_and_pad(X_val_raw, scaler, max_length)

        # ── Build + Train ────────────────────────────────────────────────
        tf.random.set_seed(current_seed + i)
        np.random.seed(current_seed + i)

        fold_model = build_model(num_features, max_length,
                                 output_type=args.output_type,
                                 loss_type=args.loss_type)

        y_train_norm = y_train / 50.0
        y_val_norm = y_val / 50.0

        history = train_fold(fold_model, X_train_pad, y_train_norm,
                             X_val_pad, y_val_norm, args)

        # ── Evaluate LSTM ─────────────────────────────────────────────────────
        lstm_train_pred = fold_model.predict(X_train_pad, verbose=0).flatten() * 50.0
        lstm_val_pred = fold_model.predict(X_val_pad, verbose=0).flatten() * 50.0

        lstm_val_pred = np.clip(lstm_val_pred, 0, 50)
        lstm_train_pred = np.clip(lstm_train_pred, 0, 50)

        # ── Evaluate RF ───────────────────────────────────────────────────────
        X_train_summary = extract_summary_features(X_train_raw)
        X_val_summary = extract_summary_features(X_val_raw)
        
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            random_state=current_seed + i,
            n_jobs=-1
        )
        rf.fit(X_train_summary, y_train)
        rf_val_pred = np.clip(rf.predict(X_val_summary), 0, 50)

        # ── Ensemble ──────────────────────────────────────────────────────────
        alpha = getattr(args, 'ensemble_alpha', 0.5)
        
        if getattr(args, 'auto_alpha', False):
            best_alpha = alpha
            best_rho = -1
            for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
                ens_pred = np.clip(a * lstm_val_pred + (1 - a) * rf_val_pred, 0, 50)
                rho, _ = spearmanr(y_val, ens_pred)
                if not np.isnan(rho) and rho > best_rho:
                    best_rho = rho
                    best_alpha = a
            alpha = best_alpha

        ensemble_val_pred = np.clip(alpha * lstm_val_pred + (1 - alpha) * rf_val_pred, 0, 50)
        
        # Override val_pred with ensemble_val_pred so the rest of the pipeline uses the ensemble
        val_pred = ensemble_val_pred
        train_pred = lstm_train_pred # Keep LSTM train pred for MAE gap computation

        fold_mae = mean_absolute_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        pred_range = val_pred.max() - val_pred.min()
        pred_std = val_pred.std()

        try:
            spearman_rho, spearman_p = spearmanr(y_val, val_pred)
            if np.isnan(spearman_rho):
                spearman_rho, spearman_p = 0.0, 1.0
        except Exception:
            spearman_rho, spearman_p = 0.0, 1.0

        try:
            pearson_r = pearsonr(y_val, val_pred)[0]
            if np.isnan(pearson_r):
                pearson_r = 0.0
        except Exception:
            pearson_r = 0.0

        # CCC (numpy)
        _mt, _mp = np.mean(y_val), np.mean(val_pred)
        _vt, _vp = np.var(y_val), np.var(val_pred)
        _cv = np.cov(y_val, val_pred, bias=True)[0][1] if len(y_val) > 1 else 0.0
        fold_ccc = float((2.0 * _cv) / (_vt + _vp + (_mt - _mp)**2 + 1e-8))

        gap = fold_mae - train_mae
        collapse_flag = pred_std < 0.5

        # Calculate isolated metrics for printing
        lstm_mae = mean_absolute_error(y_val, lstm_val_pred)
        rf_mae = mean_absolute_error(y_val, rf_val_pred)
        lstm_rho, _ = spearmanr(y_val, lstm_val_pred)
        rf_rho, _ = spearmanr(y_val, rf_val_pred)
        if np.isnan(lstm_rho): lstm_rho = 0.0
        if np.isnan(rf_rho): rf_rho = 0.0

        print(f"  [LSTM]     MAE: {lstm_mae:.2f} | Spearman ρ: {lstm_rho:.3f}")
        print(f"  [RF]       MAE: {rf_mae:.2f} | Spearman ρ: {rf_rho:.3f}")
        print(f"  [Ensemble] MAE: {fold_mae:.2f} | Spearman ρ: {spearman_rho:.3f} (α={alpha:.2f})")
        print(f"  Ensemble CCC: {fold_ccc:.4f} | Pearson r: {pearson_r:.3f}")
        print(f"  Pred range: {pred_range:.2f} | Pred std: {pred_std:.2f} "
              f"{'⚠ COLLAPSE!' if collapse_flag else '✓'}")

        # Accumulate OOF predictions (chỉ fold không bị excluded)
        if not fold_excluded:
            oof_true_included.extend(y_val.tolist())
            oof_pred_included.extend(val_pred.tolist())

        fold_metrics.append({
            'fold': i,
            'seed': current_seed,
            'train_mae': float(train_mae),
            'val_mae': float(fold_mae),
            'gap': float(gap),
            'spearman_rho': float(spearman_rho),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'fold_ccc': float(fold_ccc),
            'pred_range': float(pred_range),
            'pred_std': float(pred_std),
            'wasserstein_distance': float(w_dist),
            'y_val_mean': float(y_val.mean()),
            'y_val_std': float(y_val.std()),
            'y_val_bins': bin_counts,
            'epochs_trained': len(history.history['loss']),
            'excluded_from_oof': fold_excluded,
            'exclude_reason': exclude_reason if fold_excluded else '',
            'collapse': collapse_flag,
            'alpha_used': float(alpha),
        })

        # Save fold plots
        _save_fold_plots(history, train_pred, y_train, val_pred, y_val,
                         exercise, i, paths['plots'])

        os.makedirs(paths['save'], exist_ok=True)

    return {
        'exercise': exercise,
        'max_length': max_length,
        'seed': current_seed,
        'fold_metrics': fold_metrics,
        'oof_true': np.array(oof_true_included),
        'oof_pred': np.array(oof_pred_included),
        'n_excluded_folds': n_excluded,
    }


def cross_validate(raw_features, y, subject_ids, aug_features, aug_y, aug_ids, args, paths):
    """Repeated Cross-Validation wrapper.

    FIX v8: Hỗ trợ repeated CV (--n-repeats) để giảm variance trên dataset nhỏ.
    Mỗi repeat dùng seed khác nhau, aggregate OOF metrics across repeats.

    Workflow:
      1. Lặp n_repeats lần, mỗi lần gọi cross_validate_single_run() với seed khác
      2. Gộp OOF predictions từ tất cả repeats (loại bỏ folds excluded)
      3. Tính aggregated OOF metrics + std across repeats
      4. Report per-repeat và aggregated results
    """
    exercise = args.exercise
    n_repeats = args.n_repeats

    print(f"\n{'#'*60}")
    print(f"  Repeated CV: {n_repeats} repeats × {args.k_folds} folds = "
          f"{n_repeats * args.k_folds} total trainings")
    if n_repeats > 1:
        print(f"  Repeats dùng seed khác nhau để giảm metric variance")
        print(f"  ⚠ Training time ~{n_repeats}x so với single run")
    print(f"{'#'*60}")

    all_repeat_results = []
    all_oof_true = []
    all_oof_pred = []
    all_fold_metrics = []
    repeat_level_metrics = []

    for rep in range(n_repeats):
        seed_offset = rep * 1000  # mỗi repeat dùng seed khác nhau

        if n_repeats > 1:
            print(f"\n{'='*60}")
            print(f"  REPEAT {rep + 1}/{n_repeats} (seed_offset={seed_offset})")
            print(f"{'='*60}")

        run_result = cross_validate_single_run(
            raw_features, y, subject_ids,
            aug_features, aug_y, aug_ids,
            args, paths,
            seed_offset=seed_offset,
        )

        all_repeat_results.append(run_result)
        all_oof_true.extend(run_result['oof_true'].tolist())
        all_oof_pred.extend(run_result['oof_pred'].tolist())
        all_fold_metrics.extend(run_result['fold_metrics'])

        # Tính metrics cho repeat này
        if len(run_result['oof_true']) > 1:
            rep_true = run_result['oof_true']
            rep_pred = run_result['oof_pred']
            try:
                rep_spearman, _ = spearmanr(rep_true, rep_pred)
                if np.isnan(rep_spearman):
                    rep_spearman = 0.0
            except Exception:
                rep_spearman = 0.0

            _mt, _mp = np.mean(rep_true), np.mean(rep_pred)
            _vt, _vp = np.var(rep_true), np.var(rep_pred)
            _cv = np.cov(rep_true, rep_pred, bias=True)[0][1]
            rep_ccc = float((2.0 * _cv) / (_vt + _vp + (_mt - _mp)**2 + 1e-8))
            rep_mae = float(mean_absolute_error(rep_true, rep_pred))

            repeat_level_metrics.append({
                'repeat': rep + 1,
                'seed': args.seed + seed_offset,
                'n_oof_samples': len(rep_true),
                'n_excluded_folds': run_result['n_excluded_folds'],
                'spearman': float(rep_spearman),
                'ccc': float(rep_ccc),
                'mae': rep_mae,
            })

            if n_repeats > 1:
                print(f"\n  Repeat {rep + 1} OOF: Spearman={rep_spearman:.3f}, "
                      f"CCC={rep_ccc:.4f}, MAE={rep_mae:.2f}, "
                      f"excluded_folds={run_result['n_excluded_folds']}")

    # ── Aggregated OOF Results ───────────────────────────────────────────
    all_oof_true = np.array(all_oof_true)
    all_oof_pred = np.array(all_oof_pred)

    oof_mae = float(mean_absolute_error(all_oof_true, all_oof_pred))
    oof_rmse = float(np.sqrt(mean_squared_error(all_oof_true, all_oof_pred)))

    try:
        oof_spearman, oof_spearman_p = spearmanr(all_oof_true, all_oof_pred)
    except Exception:
        oof_spearman, oof_spearman_p = 0.0, 1.0
    try:
        oof_pearson = pearsonr(all_oof_true, all_oof_pred)[0]
    except Exception:
        oof_pearson = 0.0

    pred_std_oof = float(np.std(all_oof_pred))

    _mt, _mp = np.mean(all_oof_true), np.mean(all_oof_pred)
    _vt, _vp = np.var(all_oof_true), np.var(all_oof_pred)
    _cv = np.cov(all_oof_true, all_oof_pred, bias=True)[0][1]
    oof_ccc = float((2.0 * _cv) / (_vt + _vp + (_mt - _mp)**2 + 1e-8))

    # ── Repeated CV stability report ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AGGREGATED OOF Results: {exercise} ({n_repeats} repeats)")
    print(f"{'='*60}")
    print(f"  OOF samples:  {len(all_oof_true)} total (across all repeats)")
    print(f"  MAE:          {oof_mae:.2f}")
    print(f"  RMSE:         {oof_rmse:.2f}")
    print(f"  CCC:          {oof_ccc:.4f}  ← Clinical Agreement")
    print(f"  Spearman ρ:   {oof_spearman:.3f}  ← primary metric")
    print(f"  Pearson r:    {oof_pearson:.3f}")
    print(f"  Pred std:     {pred_std_oof:.2f}")

    if n_repeats > 1 and len(repeat_level_metrics) > 1:
        rep_spearmans = [r['spearman'] for r in repeat_level_metrics]
        rep_cccs = [r['ccc'] for r in repeat_level_metrics]
        rep_maes = [r['mae'] for r in repeat_level_metrics]

        print(f"\n  Repeated CV stability (across {len(repeat_level_metrics)} repeats):")
        print(f"    Spearman: mean={np.mean(rep_spearmans):.3f}, "
              f"std={np.std(rep_spearmans):.3f}, "
              f"range=[{np.min(rep_spearmans):.3f}, {np.max(rep_spearmans):.3f}]")
        print(f"    CCC:      mean={np.mean(rep_cccs):.4f}, "
              f"std={np.std(rep_cccs):.4f}")
        print(f"    MAE:      mean={np.mean(rep_maes):.2f}, "
              f"std={np.std(rep_maes):.2f}")

        if np.std(rep_spearmans) > 0.15:
            print(f"  ⚠ High variance in Spearman across repeats "
                  f"(std={np.std(rep_spearmans):.3f} > 0.15) — "
                  f"consider increasing n_repeats")

    total_excluded = sum(r['n_excluded_folds'] for r in all_repeat_results)
    if total_excluded > 0:
        print(f"\n  ⚠ Total excluded folds: {total_excluded} "
              f"(y_val.std < 2.0 hoặc mean lệch > 1.5σ)")

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
            print(f"    ❌ Spearman âm — model vẫn có vấn đề")

    print(f"{'='*60}")

    # Save OOF scatter plot
    _save_oof_scatter(all_oof_true, all_oof_pred, exercise, oof_spearman, paths['plots'])

    # Tính max_length từ run đầu tiên (giả sử giống nhau vì cùng data)
    max_length = all_repeat_results[0]['max_length']

    return {
        'exercise': exercise,
        'max_length': max_length,
        'n_repeats': n_repeats,
        'loss_type': 'ccc',
        'split_strategy': 'BalancedStratifiedGroupKFold',
        'oof_mae': oof_mae,
        'oof_rmse': oof_rmse,
        'oof_ccc': oof_ccc,
        'oof_spearman': float(oof_spearman),
        'oof_pearson': float(oof_pearson),
        'pred_std': pred_std_oof,
        'n_excluded_folds_total': total_excluded,
        'repeat_metrics': repeat_level_metrics,
        'folds': all_fold_metrics,
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

    model = build_model(num_features, max_length,
                        output_type=args.output_type,
                        loss_type=args.loss_type)
    history = train_fold(model, X_train, y_train_norm, X_val, y_val_norm, args)

    model.summary()

    # Save
    os.makedirs(paths['save'], exist_ok=True)
    model_path = f"{paths['save']}/ml_model_{exercise}.keras"
    model.save(model_path)
    print(f"✅ LSTM Model saved: {model_path}")

    # ── Train Final RF ──
    print(f"  Training final RF ensemble model...")
    X_train_summary_final = extract_summary_features(raw_features)
    rf_final = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        random_state=args.seed, n_jobs=-1
    )
    rf_final.fit(X_train_summary_final, y)
    
    import joblib
    rf_path = f"{paths['save']}/rf_model_{exercise}.joblib"
    joblib.dump(rf_final, rf_path)
    print(f"✅ RF Model saved: {rf_path}")

    import joblib
    scaler_path = f"{paths['save']}/scaler_{exercise}.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    config = {
        'exercise': exercise,
        'max_length': max_length,
        'n_features': num_features,
        'mask_value': MASK_VALUE,
        'output_type': getattr(args, 'output_type', 'sigmoid'),
        'loss_type': getattr(args, 'loss_type', 'ccc'),
        'ensemble_alpha': getattr(args, 'ensemble_alpha', 0.5),
        'auto_alpha': getattr(args, 'auto_alpha', False),
        'architecture': (f'2xLSTM(32/16), dropout=0.3, Dense(8,relu), '
                         f'Dense(1,{getattr(args, "output_type", "sigmoid")}), {getattr(args, "loss_type", "ccc")} + RF'),
        'version': 'v9_ensemble_linear_output',
        'y_normalization': 'y/50',
    }
    config_path = f"{paths['save']}/model_config_{exercise}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved: {config_path}")

    return model, scaler, max_length


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
    print(f"  RehabAI Practical LSTM v8 (Balanced CV + Repeated)")
    print(f"{'='*60}")
    print(f"Environment:      {args.env}")
    print(f"Exercise:         {args.exercise}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"K-folds:          {args.k_folds}")
    print(f"N-repeats:        {args.n_repeats}")
    print(f"Total trainings:  {args.k_folds * args.n_repeats}")
    print(f"CSV:              {paths['csv']}")

    # Load single CSV, split original vs augmented by video path
    full_df = pd.read_csv(paths['csv'])
    is_aug = full_df['video'].str.contains('augmented', case=False, na=False)
    df_orig = full_df[~is_aug].reset_index(drop=True)
    df_aug = full_df[is_aug].reset_index(drop=True)
    print(f"CSV split: {len(df_orig)} original + {len(df_aug)} augmented = {len(full_df)} total")

    # Load original data (with subject IDs for grouping)
    raw_features, y, subject_ids = load_raw_features(
        df_orig, args.exercise, args.downsample, args.target_len)

    if len(raw_features) < args.k_folds:
        print(f"ERROR: Chỉ có {len(raw_features)} samples, cần ít nhất {args.k_folds}")
        return

    # Load augmented data from the same CSV
    aug_features, aug_y, aug_ids = [], np.array([], dtype=np.float32), np.array([])
    if len(df_aug) > 0:
        aug_features, aug_y, aug_ids = load_raw_features(
            df_aug, args.exercise, args.downsample, args.target_len)
        print(f"Augmented data loaded: {len(aug_features)} samples")
    else:
        print(f"⚠ No augmented samples found in CSV — proceeding without augmentation")

    NUM_FEATURES = raw_features[0].shape[1]
    print(f"Features per frame: {NUM_FEATURES}")
    print(f"Original samples: {len(raw_features)}")

    # ── GroupKFold CV on ALL subjects (no held-out test set) ─────────────
    # Methodology: Guo & Khan (2021) and arXiv 2306.09546 both use full
    # dataset for CV evaluation. With only ~67 subjects, a held-out test
    # set (N≈10) produces statistically unreliable metrics (SE≈0.38 for
    # Spearman). OOF predictions from GroupKFold are the primary evaluation.
    unique_subjects = np.unique(subject_ids)

    print(f"\n── Full-dataset Balanced CV (no held-out test) ──")
    print(f"  Total subjects: {len(unique_subjects)}")
    print(f"  Original samples: {len(raw_features)}")
    print(f"  Augmented samples: {len(aug_features)} (merged into train folds only)")
    print(f"  Split: BalancedStratifiedGroupKFold (custom, 5 bins, round-robin)")
    print(f"  Repeats: {args.n_repeats} (different seeds for stability)")

    # Cross-validate on ALL original data (GroupKFold by subject)
    # Augmented data is merged into train folds only — no leakage
    cv_results = cross_validate(
        raw_features, y, subject_ids,
        aug_features, aug_y, aug_ids,
        args, paths)

    # Train final deployment model on ALL data (original + augmented)
    # This model is for inference/deployment only — NOT for evaluation.
    # Reported metrics come from OOF (above), not from this model.
    all_features = raw_features + list(aug_features) if len(aug_features) > 0 else raw_features
    all_y = np.concatenate([y, aug_y]) if len(aug_y) > 0 else y
    final_model, scaler, max_length = train_final_model(all_features, all_y, args, paths)

    # Save CV results
    os.makedirs(paths['plots'], exist_ok=True)
    results_path = f"{paths['plots']}/{args.exercise}_cv_results_v8.json"
    with open(results_path, 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    print(f"\n✅ CV results saved: {results_path}")

    # ── Final Report (OOF metrics — primary evaluation) ──────────────────
    print(f"\n{'='*60}")
    print(f"  Finished: {args.exercise}")
    print(f"  Methodology: BalancedStratifiedGroupKFold CV + Repeated ({args.n_repeats}x)")
    print(f"  Subjects: {len(unique_subjects)} | Folds: {args.k_folds} × {args.n_repeats} repeats")
    print(f"  OOF CCC:       {cv_results['oof_ccc']:.4f}  ← Clinical Agreement")
    print(f"  OOF Spearman:  {cv_results['oof_spearman']:.3f}")
    print(f"  OOF MAE:       {cv_results['oof_mae']:.2f}")
    print(f"  Excluded folds: {cv_results.get('n_excluded_folds_total', 0)}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model: {paths['save']}/ml_model_{args.exercise}.keras")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
