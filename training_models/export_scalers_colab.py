"""
Export StandardScaler artifacts (v4 — Paper-aligned, no temporal stats).
========================================================================
Run on Colab after mounting Google Drive, or locally.

Imports feature functions from backend/joint_features.py (single source of truth)
to ensure training-time and inference-time features are EXACTLY the same.

Usage:
  Colab:  !python export_scalers_colab.py
  Local:  python export_scalers_colab.py --env local --base-dir C:/RehabAI
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Export per-exercise StandardScalers')
    p.add_argument('--env', choices=['colab', 'local'], default='colab')
    p.add_argument('--base-dir', default=None)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.base_dir:
        base = args.base_dir
    elif args.env == 'colab':
        base = '/content/drive/MyDrive/RehabAI'
    else:
        base = 'C:/RehabAI'

    # Import 3D feature functions from backend (single source of truth)
    backend_dir = os.path.join(base, 'backend')
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from joint_features import (
        get_es1_features, get_es2_features, get_es3_features,
        get_es4_features, get_es5_features,
    )

    FEATURE_EXTRACTORS = {
        "Es1": get_es1_features, "Es2": get_es2_features, "Es3": get_es3_features,
        "Es4": get_es4_features, "Es5": get_es5_features,
    }

    csv_path = f'{base}/05_final_datasets/KiMoRe_final.csv'
    save_dir = f'{base}/models/best_models'

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Group raw features by exercise
    exercise_groups = {}
    for _, row in df.iterrows():
        exercise = row['exercise']
        jp_path = row['joint_positions']
        if pd.isna(jp_path) or not os.path.exists(jp_path):
            continue
        extractor = FEATURE_EXTRACTORS.get(exercise)
        if not extractor:
            continue
        joint_data = pd.read_csv(jp_path)
        raw_features = extractor(joint_data)

        # Clean NaN/inf
        raw_df = raw_features.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        if exercise not in exercise_groups:
            exercise_groups[exercise] = []
        exercise_groups[exercise].append(raw_df)

    # Fit and save scalers
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nFitting scalers (pure geometry features) → {save_dir}")

    for exercise, df_list in sorted(exercise_groups.items()):
        all_frames = pd.concat(df_list, ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(all_frames)

        scaler_path = os.path.join(save_dir, f"scaler_{exercise}.joblib")
        joblib.dump(scaler, scaler_path)

        print(f"  ✓ {exercise}: {len(df_list)} videos, "
              f"{all_frames.shape[1]} features")
        print(f"    mean range: [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")
        print(f"    std range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
        print(f"    saved to: {scaler_path}")

    print(f"\n✅ Done! Copy scaler_EsX.joblib files to your project's models/ directory.")


if __name__ == '__main__':
    main()
