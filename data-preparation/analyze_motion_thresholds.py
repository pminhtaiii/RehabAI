"""Analyze KiMoRe feature data to derive motion detection thresholds.

Computes variance, ROM, and displacement statistics per exercise from the
extracted feature CSVs. Use the output to calibrate EXERCISE_THRESHOLDS
in backend/motion_detector.py.

Usage (Colab or local — wherever KiMoRe data lives):
    python analyze_motion_thresholds.py [--base-dir /path/to/RehabAI]

Output:
    Prints recommended thresholds per exercise based on percentile analysis.
    Thresholds are set at percentile 5 of valid samples (score >= 10/50)
    to ensure only truly static inputs fail the gate.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Analyze motion thresholds from KiMoRe features')
    p.add_argument('--base-dir', default='C:/RehabAI',
                   help='Base directory containing 05_final_datasets/')
    p.add_argument('--min-score', type=float, default=10.0,
                   help='Minimum clinical score to include in threshold analysis (0-50 scale)')
    p.add_argument('--percentile', type=float, default=5.0,
                   help='Percentile to use for threshold (lower = more conservative)')
    args, _ = p.parse_known_args()
    return args


def compute_motion_metrics(features: np.ndarray):
    """Compute the same 3 metrics as motion_detector.detect_motion().

    Args:
        features: (T, F) ndarray of raw features for one sample.

    Returns:
        dict with avg_variance, avg_rom, avg_displacement.
    """
    if features.shape[0] < 5:
        return None

    # Metric 1: Temporal variance per feature, averaged
    per_feature_variance = np.var(features, axis=0)
    avg_variance = float(np.mean(per_feature_variance))

    # Metric 2: Range of motion (peak-to-peak) per feature, averaged
    per_feature_rom = np.ptp(features, axis=0)
    avg_rom = float(np.mean(per_feature_rom))

    # Metric 3: Frame-to-frame displacement, averaged
    if features.shape[0] > 1:
        frame_diffs = np.abs(np.diff(features, axis=0))
        avg_displacement = float(np.mean(frame_diffs))
    else:
        avg_displacement = 0.0

    return {
        'avg_variance': avg_variance,
        'avg_rom': avg_rom,
        'avg_displacement': avg_displacement,
    }


def main():
    args = parse_args()

    csv_path = os.path.join(args.base_dir, '05_final_datasets',
                            'KiMoRe_data_movenet_features.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found: {csv_path}")
        print("  Ensure 03_extract_joint_features.py has been run.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['clinical_score', 'joint_features']).reset_index(drop=True)

    print("=" * 70)
    print("  Motion Threshold Analysis — KiMoRe Feature Data")
    print(f"  Input: {csv_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Min score filter: {args.min_score}/50")
    print(f"  Threshold percentile: {args.percentile}")
    print("=" * 70)

    exercises = sorted(df['exercise'].unique())

    # Collect metrics per exercise
    results = {}

    for exercise in exercises:
        ex_df = df[df['exercise'] == exercise]
        metrics_list = []

        for _, row in ex_df.iterrows():
            fpath = row['joint_features']
            score = row['clinical_score']

            if pd.isna(fpath) or not os.path.exists(str(fpath)):
                continue

            try:
                feat_df = pd.read_csv(fpath)
                feat_arr = feat_df.to_numpy(dtype=np.float64)
                feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e:
                print(f"  Skip {fpath}: {e}")
                continue

            metrics = compute_motion_metrics(feat_arr)
            if metrics is None:
                continue

            metrics['clinical_score'] = score
            metrics['sample_id'] = row.get('ID', '')
            metrics['n_frames'] = feat_arr.shape[0]
            metrics_list.append(metrics)

        results[exercise] = metrics_list

    # Report per exercise
    print()
    recommended = {}

    for exercise in exercises:
        metrics_list = results.get(exercise, [])
        if not metrics_list:
            print(f"\n{exercise}: No valid samples found!")
            continue

        metrics_df = pd.DataFrame(metrics_list)

        print(f"\n{'─' * 70}")
        print(f"  {exercise}: {len(metrics_df)} samples")
        print(f"{'─' * 70}")

        # All samples stats
        for metric in ['avg_variance', 'avg_rom', 'avg_displacement']:
            vals = metrics_df[metric]
            print(f"  {metric:25s}: "
                  f"min={vals.min():.4f}, "
                  f"p5={vals.quantile(0.05):.4f}, "
                  f"p25={vals.quantile(0.25):.4f}, "
                  f"median={vals.median():.4f}, "
                  f"p75={vals.quantile(0.75):.4f}, "
                  f"max={vals.max():.4f}")

        # Filter for "active" samples (score >= min_score)
        active_df = metrics_df[metrics_df['clinical_score'] >= args.min_score]
        print(f"\n  Active samples (score >= {args.min_score}): {len(active_df)}/{len(metrics_df)}")

        if len(active_df) > 0:
            pct = args.percentile / 100.0
            rec_variance = float(active_df['avg_variance'].quantile(pct))
            rec_rom = float(active_df['avg_rom'].quantile(pct))
            rec_displacement = float(active_df['avg_displacement'].quantile(pct))

            # Apply 0.5x safety margin (threshold = half of p5 → more conservative)
            rec_variance_safe = rec_variance * 0.5
            rec_rom_safe = rec_rom * 0.5
            rec_displacement_safe = rec_displacement * 0.5

            recommended[exercise] = {
                'min_variance': round(rec_variance_safe, 3),
                'min_rom': round(rec_rom_safe, 3),
                'min_displacement': round(rec_displacement_safe, 4),
            }

            print(f"  Recommended thresholds (p{args.percentile} × 0.5 safety):")
            print(f"    min_variance:     {rec_variance_safe:.3f}  (raw p{args.percentile}: {rec_variance:.3f})")
            print(f"    min_rom:          {rec_rom_safe:.3f}  (raw p{args.percentile}: {rec_rom:.3f})")
            print(f"    min_displacement: {rec_displacement_safe:.4f}  (raw p{args.percentile}: {rec_displacement:.4f})")

        # Low-score samples for comparison
        low_df = metrics_df[metrics_df['clinical_score'] < args.min_score]
        if len(low_df) > 0:
            print(f"\n  Low-score samples (score < {args.min_score}): {len(low_df)}")
            for metric in ['avg_variance', 'avg_rom', 'avg_displacement']:
                vals = low_df[metric]
                print(f"    {metric:25s}: min={vals.min():.4f}, median={vals.median():.4f}, max={vals.max():.4f}")

    # Final output: copy-paste code block
    if recommended:
        print(f"\n{'=' * 70}")
        print("  COPY-PASTE into backend/motion_detector.py:")
        print(f"{'=' * 70}")
        print("EXERCISE_THRESHOLDS = {")
        for ex in sorted(recommended.keys()):
            t = recommended[ex]
            print(f'    "{ex}": {{"min_variance": {t["min_variance"]}, '
                  f'"min_rom": {t["min_rom"]}, '
                  f'"min_displacement": {t["min_displacement"]}}},')
        print("}")

    print(f"\n{'=' * 70}")
    print("  Done. Review thresholds above before applying.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
