import os
import pandas as pd
import numpy as np

# Configuration
MOVENET_CSV_DIR = 'C:/RehabAI/03_raw_joints'
VIDEO_BASE_DIR = 'C:/RehabAI/01_raw_data'
FINAL_DATASET_DIR = 'C:/RehabAI/05_final_datasets'

# Directories to search for ClinicalAssessment_*.xlsx files
LABELS_SEARCH_DIRS = [
    'C:/RehabAI/02_clinical_labels',
    'C:/RehabAI'
]
LABELS_SEARCH_DIRS = [d for d in LABELS_SEARCH_DIRS if os.path.exists(d)]

def read_clinical_score_simple(subject_id, exercise, clinical_group, expertise):
    import glob
    ex_num = exercise[-1:]
    ts_col = f'clinical TS Ex#{ex_num}'
    po_col = f'clinical PO Ex#{ex_num}'
    cf_col = f'clinical CF Ex#{ex_num}'

    for base_dir in LABELS_SEARCH_DIRS:
        # Construct the expected path down to the exercise folder
        # Handle cases where subject_id might use a space instead of an underscore (e.g., 'E ID1' instead of 'E_ID1')
        possible_subject_ids = [subject_id, subject_id.replace('_', ' '), subject_id.replace(' ', '_')]
        
        exercise_dirs = []
        for s_id in possible_subject_ids:
            exercise_dirs.append(os.path.join(base_dir, clinical_group, expertise, s_id, exercise))
        
        search_dirs = []
        for ex_dir in exercise_dirs:
            label_dir = os.path.join(ex_dir, 'Label')
            search_dirs.extend([label_dir, ex_dir, os.path.dirname(ex_dir)])
        
        found_file = None
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                # Look for ClinicalAssessment*.xlsx
                pattern = os.path.join(s_dir, 'ClinicalAssessment*.xlsx')
                matches = glob.glob(pattern)
                if matches:
                    found_file = matches[0]
                    break
                    
        # Fallback recursive search if not found in specific directories
        if not found_file:
            for s_id in possible_subject_ids:
                subject_dir = os.path.join(base_dir, clinical_group, expertise, s_id)
                if os.path.exists(subject_dir):
                    for root, _, files in os.walk(subject_dir):
                        for f in files:
                            if f.startswith('ClinicalAssessment') and f.endswith('.xlsx'):
                                found_file = os.path.join(root, f)
                                break
                        if found_file:
                            break
                if found_file:
                    break

        if found_file:
            try:
                score_df = pd.read_excel(found_file)
                if ts_col in score_df.columns and po_col in score_df.columns and cf_col in score_df.columns:
                    ts_val = pd.to_numeric(score_df[ts_col], errors='coerce').dropna()
                    po_val = pd.to_numeric(score_df[po_col], errors='coerce').dropna()
                    cf_val = pd.to_numeric(score_df[cf_col], errors='coerce').dropna()
                    if not ts_val.empty and not po_val.empty and not cf_val.empty:
                        return float(ts_val.iloc[0]) + float(po_val.iloc[0]) + float(cf_val.iloc[0])

                # Fallback to sum of first 3 valid numbers if columns not strictly matched
                numeric_values = pd.to_numeric(score_df.to_numpy().ravel(), errors='coerce')
                numeric_values = [v for v in numeric_values if not np.isnan(v)]
                if len(numeric_values) >= 3:
                    pass
            except Exception as e:
                print(f'Could not read score from {found_file}: {e}')
            
            # Since we found the file but maybe failed to parse or find columns, we can either break or return
            # We break to avoid searching other base_dirs if we already found the right file
            break

    return np.nan

def main():
    print("Starting dataset preparation...")
    if not os.path.exists(MOVENET_CSV_DIR):
        print(f"Error: Directory not found: {MOVENET_CSV_DIR}")
        return

    folders_to_process = [
        d for d in os.listdir(MOVENET_CSV_DIR)
        if os.path.isdir(os.path.join(MOVENET_CSV_DIR, d))
    ]

    metadata_list = []
    success_count = 0
    missing_video_count = 0
    missing_score_count = 0

    for clinical_group in folders_to_process:
        clinical_group_path = os.path.join(MOVENET_CSV_DIR, clinical_group)
        expertise_levels = [
            d for d in os.listdir(clinical_group_path)
            if os.path.isdir(os.path.join(clinical_group_path, d))
        ]

        for expertise in expertise_levels:
            expertise_path = os.path.join(clinical_group_path, expertise)
            subject_ids = [
                d for d in os.listdir(expertise_path)
                if os.path.isdir(os.path.join(expertise_path, d))
            ]

            for subj_id in subject_ids:
                subject_path = os.path.join(expertise_path, subj_id)
                exercises = [
                    d for d in os.listdir(subject_path)
                    if os.path.isdir(os.path.join(subject_path, d))
                ]

                for exercise in exercises:
                    exercise_path = os.path.join(subject_path, exercise)
                    csv_files = [
                        f for f in os.listdir(exercise_path)
                        if f.lower().endswith('.csv') and not f.endswith('_features.csv')
                    ]

                    if len(csv_files) == 0:
                        continue

                    movenet_path = os.path.join(exercise_path, csv_files[0])

                    try:
                        num_frames = len(pd.read_csv(movenet_path))
                    except Exception as e:
                        print(f'Error processing {movenet_path}: {e}')
                        continue

                    video_path = np.nan
                    rgb_folder = os.path.join(VIDEO_BASE_DIR, clinical_group, expertise, subj_id, exercise, 'rgb')
                    if os.path.isdir(rgb_folder):
                        video_files = [
                            f for f in os.listdir(rgb_folder)
                            if f.lower().endswith('.mp4')
                        ]
                        if len(video_files) > 0:
                            video_path = os.path.join(rgb_folder, video_files[0])

                    clinical_score = read_clinical_score_simple(subj_id, exercise, clinical_group, expertise)

                    if pd.isna(video_path):
                        missing_video_count += 1
                    if pd.isna(clinical_score):
                        missing_score_count += 1

                    metadata_list.append({
                        'ID': subj_id,
                        'clinical_group': clinical_group,
                        'expertise': expertise,
                        'exercise': exercise,
                        'video': video_path,
                        'joint_positions': movenet_path,
                        'clinical_score': clinical_score,
                        '#frames': num_frames
                    })
                    success_count += 1

    print(f'Processed files: {success_count}')
    print(f'Missing video files: {missing_video_count}')
    print(f'Missing clinical scores: {missing_score_count}')

    df_meta = pd.DataFrame(metadata_list)
    
    expected_columns = [
        'ID', 'clinical_group', 'expertise', 'exercise', 'video', 
        'joint_positions', 'clinical_score', '#frames'
    ]
    for col in expected_columns:
        if col not in df_meta.columns:
            df_meta[col] = np.nan
    df_meta = df_meta[expected_columns]

    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)
    output_path = os.path.join(FINAL_DATASET_DIR, 'KiMoRe_final.csv')
    df_meta.to_csv(output_path, index=False)
    print(f'Saved metadata dataset to: {output_path}')

if __name__ == "__main__":
    main()
