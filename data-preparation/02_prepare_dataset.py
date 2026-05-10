"""Build master metadata CSV from MediaPipe 3D joint positions.

Scans extracted joint CSV files, matches with clinical labels,
and writes combined dataset for feature extraction.
"""

import os
import glob
import pandas as pd
import numpy as np


BASE_DIR = 'C:/RehabAI'

JOINTS_DIR = f'{BASE_DIR}/03_raw_joints'
VIDEO_DIR = f'{BASE_DIR}/01_raw_data'
FINAL_DATASET_DIR = f'{BASE_DIR}/05_final_datasets'

# Directories to search for ClinicalAssessment_*.xlsx files
LABELS_SEARCH_DIRS = [
    f'{BASE_DIR}/02_clinical_labels',
    BASE_DIR,
]
LABELS_SEARCH_DIRS = [d for d in LABELS_SEARCH_DIRS if os.path.exists(d)]


# ── Clinical Score Reader ─────────────────────────────────────────────────────

def read_clinical_score(subject_id, exercise, clinical_group, expertise):
    """Read clinical score (TS only, range 0-50) from ClinicalAssessment Excel files.

    Uses only the Total Score (TS) column as per the KIMORE dataset definition
    (Capecci et al., 2019) and arXiv 2306.09546. TS = PO + CF, so we must NOT
    add them again (that would double-count).

    Searches in multiple possible locations following KiMoRe dataset structure:
        02_clinical_labels/{clinical_group}/{expertise}/{subject_id}/{exercise}/Label/
    """
    ex_num = exercise[-1:]
    ts_col = f'clinical TS Ex#{ex_num}'

    # Try subject_id with both underscore and space variants
    possible_subject_ids = [
        subject_id,
        subject_id.replace('_', ' '),
        subject_id.replace(' ', '_'),
    ]

    for base_dir in LABELS_SEARCH_DIRS:
        for s_id in possible_subject_ids:
            # Build search paths from most specific to least
            search_dirs = [
                os.path.join(base_dir, clinical_group, expertise, s_id, exercise, 'Label'),
                os.path.join(base_dir, clinical_group, expertise, s_id, exercise),
                os.path.join(base_dir, clinical_group, expertise, s_id),
            ]

            for s_dir in search_dirs:
                if not os.path.exists(s_dir):
                    continue
                matches = glob.glob(os.path.join(s_dir, 'ClinicalAssessment*.xlsx'))
                if matches:
                    return _parse_clinical_file(matches[0], ts_col)

        # Fallback: recursive search under subject directory
        for s_id in possible_subject_ids:
            subject_dir = os.path.join(base_dir, clinical_group, expertise, s_id)
            if not os.path.exists(subject_dir):
                continue
            for root, _, files in os.walk(subject_dir):
                for f in files:
                    if f.startswith('ClinicalAssessment') and f.endswith('.xlsx'):
                        result = _parse_clinical_file(
                            os.path.join(root, f), ts_col)
                        if not np.isnan(result):
                            return result

    return np.nan


def _parse_clinical_file(filepath, ts_col):
    """Parse a ClinicalAssessment Excel file and return TS score (0-50)."""
    try:
        score_df = pd.read_excel(filepath)
        if ts_col in score_df.columns:
            ts_val = pd.to_numeric(score_df[ts_col], errors='coerce').dropna()
            if not ts_val.empty:
                return float(ts_val.iloc[0])
    except Exception as e:
        print(f'  ⚠ Could not read score from {filepath}: {e}')
    return np.nan


# ── CSV File Selection ────────────────────────────────────────────────────────

def find_mediapipe_csv(exercise_path):
    """Find the best joint positions CSV in an exercise directory.

    Priority:
      1. *_mediapipe.csv  (output of 01_extract_joint_positions.py with MediaPipe 3D)
      2. Any .csv that is NOT a _features.csv

    Returns the path to the selected CSV, or None if nothing found.
    """
    all_csvs = [
        f for f in os.listdir(exercise_path)
        if f.lower().endswith('.csv') and not f.endswith('_features.csv')
    ]

    if not all_csvs:
        return None

    # Prefer MediaPipe CSVs
    mediapipe_csvs = [f for f in all_csvs if '_mediapipe' in f.lower()]
    if mediapipe_csvs:
        return os.path.join(exercise_path, mediapipe_csvs[0])

    # Fallback to any non-feature CSV
    return os.path.join(exercise_path, all_csvs[0])


# ── Video File Lookup ─────────────────────────────────────────────────────────

def find_video_path(clinical_group, expertise, subject_id, exercise):
    """Locate the .mp4 video file for a given subject/exercise.

    Checks both:
      - {VIDEO_DIR}/{group}/{expertise}/{subject}/{exercise}/rgb/*.mp4
      - {VIDEO_DIR}/{group}/{expertise}/{subject}/{exercise}/*.mp4
    """
    base = os.path.join(VIDEO_DIR, clinical_group, expertise, subject_id, exercise)

    # Check rgb/ subdirectory first (standard KiMoRe layout)
    rgb_dir = os.path.join(base, 'rgb')
    if os.path.isdir(rgb_dir):
        videos = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.mp4')]
        if videos:
            return os.path.join(rgb_dir, videos[0])

    # Check exercise directory directly
    if os.path.isdir(base):
        videos = [f for f in os.listdir(base) if f.lower().endswith('.mp4')]
        if videos:
            return os.path.join(base, videos[0])

    return np.nan


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  02_prepare_dataset.py — MediaPipe 3D Pipeline")
    print("=" * 60)
    print(f"  Joints dir:  {JOINTS_DIR}")
    print(f"  Video dir:   {VIDEO_DIR}")
    print(f"  Output dir:  {FINAL_DATASET_DIR}")
    print()

    if not os.path.exists(JOINTS_DIR):
        print(f"Error: Joint positions directory not found: {JOINTS_DIR}")
        print("  Run 01_extract_joint_positions.py first.")
        return

    # Traverse: clinical_group / expertise / subject_id / exercise
    metadata_list = []
    stats = {'success': 0, 'missing_video': 0, 'missing_score': 0, 'no_csv': 0}

    clinical_groups = [
        d for d in sorted(os.listdir(JOINTS_DIR))
        if os.path.isdir(os.path.join(JOINTS_DIR, d))
    ]

    for clinical_group in clinical_groups:
        group_path = os.path.join(JOINTS_DIR, clinical_group)
        expertise_levels = [
            d for d in sorted(os.listdir(group_path))
            if os.path.isdir(os.path.join(group_path, d))
        ]

        for expertise in expertise_levels:
            expertise_path = os.path.join(group_path, expertise)
            subject_ids = [
                d for d in sorted(os.listdir(expertise_path))
                if os.path.isdir(os.path.join(expertise_path, d))
            ]

            for subj_id in subject_ids:
                subject_path = os.path.join(expertise_path, subj_id)
                exercises = [
                    d for d in sorted(os.listdir(subject_path))
                    if os.path.isdir(os.path.join(subject_path, d))
                ]

                for exercise in exercises:
                    exercise_path = os.path.join(subject_path, exercise)

                    # Find MediaPipe CSV
                    csv_path = find_mediapipe_csv(exercise_path)
                    if csv_path is None:
                        stats['no_csv'] += 1
                        continue

                    # Count frames
                    try:
                        num_frames = len(pd.read_csv(csv_path))
                    except Exception as e:
                        print(f'  ⚠ Error reading {csv_path}: {e}')
                        continue

                    # Find video
                    video_path = find_video_path(
                        clinical_group, expertise, subj_id, exercise)

                    # Read clinical score
                    clinical_score = read_clinical_score(
                        subj_id, exercise, clinical_group, expertise)

                    # Track stats
                    if pd.isna(video_path):
                        stats['missing_video'] += 1
                    if pd.isna(clinical_score):
                        stats['missing_score'] += 1

                    metadata_list.append({
                        'ID': subj_id,
                        'clinical_group': clinical_group,
                        'expertise': expertise,
                        'exercise': exercise,
                        'video': video_path,
                        'joint_positions': csv_path,
                        'clinical_score': clinical_score,
                        '#frames': num_frames,
                    })
                    stats['success'] += 1

    # Build DataFrame with consistent column order
    expected_columns = [
        'ID', 'clinical_group', 'expertise', 'exercise',
        'video', 'joint_positions', 'clinical_score', '#frames',
    ]
    df_meta = pd.DataFrame(metadata_list)
    for col in expected_columns:
        if col not in df_meta.columns:
            df_meta[col] = np.nan
    df_meta = df_meta[expected_columns]

    # Save
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)
    output_path = os.path.join(FINAL_DATASET_DIR, 'KiMoRe_final.csv')
    df_meta.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary:")
    print(f"    Total entries:          {stats['success']}")
    print(f"    Missing video files:    {stats['missing_video']}")
    print(f"    Missing clinical scores:{stats['missing_score']}")
    print(f"    Dirs without CSV:       {stats['no_csv']}")
    print(f"")

    if not df_meta.empty:
        print(f"  Per-exercise breakdown:")
        for ex in sorted(df_meta['exercise'].dropna().unique()):
            ex_df = df_meta[df_meta['exercise'] == ex]
            valid_scores = ex_df['clinical_score'].dropna()
            print(f"    {ex}: {len(ex_df)} samples, "
                  f"{len(valid_scores)} with scores"
                  f" (range: {valid_scores.min():.0f}-{valid_scores.max():.0f})"
                  if len(valid_scores) > 0 else
                  f"    {ex}: {len(ex_df)} samples, 0 with scores")

    print(f"\n  Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
