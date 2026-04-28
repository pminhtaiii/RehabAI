"""
Export StandardScaler artifacts for use in inference.

Run after 03_extract_joint_features.py to copy scaler_EsX.joblib files
from the training output directory to the models directory where the
backend expects them.

Usage:
    python scripts/export_scalers.py
"""

import os
import shutil
import sys

SOURCE_DIR = os.path.join("C:/RehabAI/05_final_datasets")
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

EXERCISES = ["Es1", "Es2", "Es3", "Es4", "Es5"]


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    copied = 0
    missing = []

    for ex_id in EXERCISES:
        filename = f"scaler_{ex_id}.joblib"
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(TARGET_DIR, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {filename} → {dst}")
            copied += 1
        else:
            missing.append(ex_id)
            print(f"  ✗ {filename} not found at {src}")

    print(f"\nCopied: {copied}/{len(EXERCISES)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Run 03_extract_joint_features.py first to generate scaler artifacts.")
        sys.exit(1)
    else:
        print("All scalers exported successfully.")


if __name__ == "__main__":
    main()
