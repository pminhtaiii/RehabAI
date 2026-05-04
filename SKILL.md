---
name: RehabAI Development
description: This skill should be used when the user asks to "modify the backend", "add a new exercise", "update ML models", "integrate frontend with backend", "fix authentication", or mentions "RehabAI architecture", "clinical score", or "DTW feedback".
version: 0.1.0
---

# RehabAI Development

This skill provides guidance for developing, maintaining, and extending the RehabAI application, a physical rehabilitation tracking system with machine learning-based clinical scoring and real-time feedback.

## Core Architecture

RehabAI consists of a FastAPI backend and a Vite-based frontend, connected via REST APIs and orchestrated via Docker Compose.

- **Backend**: FastAPI, SQLAlchemy (SQLite), TensorFlow/Keras for model predictions, `tslearn` for Dynamic Time Warping (DTW) feedback.
- **Frontend**: Vite (React/Vue), communicating with the backend API.
- **Machine Learning**: Keras models (`ml_model_EsX.keras`) for exercise evaluation, processing joint features over time. Models output normalized [0,1] (training uses y/50), un-normalized to [0,50] then scaled to 0-100 for UI display (×50×2). Architecture: 4×LSTM(16), aligned with arXiv 2306.09546.

## Development Workflows

### Add a New Exercise Model

To integrate a new exercise into the clinical scoring system:

1. Train and export the new Keras model as `ml_model_EsX.keras`.
2. Place the exported model file in the project root `models/` directory (mounted into Docker at `/app/models`).
3. Update `max_length_mapping` in `backend/main.py` with the appropriate sequence length for the new model.
4. Add the model path to `model_paths` in `backend/main.py`.
5. Update database seed data in `backend/data_wrapper.py` to include the new exercise in the `Exercise` table.

### Modify Database Schema

To update the database models and apply changes:

1. Edit `backend/models.py` to add or modify SQLAlchemy models.
2. Update foreign key relationships and back-populates if necessary.
3. Verify new tables are registered in `Base.metadata.create_all(bind=engine)` within `backend/main.py`.
4. Update `backend/database.py` if changing connection settings.

### Update API Endpoints

To add or modify FastAPI routes:

1. Define the endpoint in `backend/main.py` using appropriate HTTP method decorators.
2. Inject `db: db_dependency` for database access.
3. Enforce authentication by injecting `current_user: models.User = Depends(get_current_user)`.
4. Return structured JSON responses using `JSONResponse` or Pydantic models.

## Machine Learning Integration

### Pipeline Flow (v4 — Paper-Aligned)

The full ML pipeline has 3 stages that must stay consistent:

**Stage 1: Data Preparation** (offline, run once on KIMORE dataset)
1. `01_extract_joint_positions.py` — MediaPipe 3D keypoints from videos → raw CSV (48 cols)
2. `02_prepare_dataset.py` — Build metadata CSV linking keypoint paths to clinical scores
3. `03_extract_joint_features.py` — Exercise-specific features (Paper 2, Table 2) → **raw** feature CSVs + fit StandardScalers per exercise → save `scaler_EsX.joblib`

**Stage 2: Training** (Colab)
1. `clinical_score_prediction_model.py` — Load raw feature CSVs → load scaler → `scaler.transform()` → pad with `-999.0` → train 4×LSTM(16) with `Masking(-999.0)` → save model + config
2. Copy model `.keras` + `scaler_EsX.joblib` + `model_config_EsX.json` to `models/`

**Stage 3: Inference** (Backend)
1. Frontend sends live keypoints as CSV
2. `ml_wrapper.py` extracts exercise-specific features → `temporal_downsample(stride=5)` → loads scaler → `scaler.transform()` → pad with `-999.0` → `model.predict()`
3. Model outputs normalized [0,1], un-normalized ×50 to [0,50] TS score, then ×2 for 0-100 UI

**CRITICAL constraints:**
- No downsampling during data preparation — Paper 1 feeds ALL consecutive frames
- BUT temporal downsampling (stride=5) is applied during TRAINING and INFERENCE to reduce sequence length
- Feature CSVs store **raw** values; scaling happens at training/inference time
- Padding sentinel is `-999.0` (not `0.0`, which conflicts with StandardScaler)
- `joint_features.py` (backend) and `03_extract_joint_features.py` must compute identical features
- Model predicts normalized [0,1]; inference must un-normalize ×50 before ×2 scaling

### Clinical Score Prediction

The `/api/clinical_score/{exercise_id}` endpoint processes CSV data containing joint movements. To modify this pipeline:

1. Validate incoming CSV strings via the `CsvStringModel` schema.
2. Reorder dataframe features using `reorder_dataframe()` from `backend/ml_wrapper.py`.
3. Prepare sequence data using `prepare_data()` with the exercise's specific `max_length`.
   - **CRITICAL**: `prepare_data()` applies a per-exercise `StandardScaler` loaded from `models/scaler_EsX.joblib`. These scalers MUST match the ones used during training. Without them, model output will be a constant ~53/100.
   - **CRITICAL**: Padding uses `-999.0` sentinel, matched by `Masking(mask_value=-999.0)` in the model.
5. Execute inference using the pre-loaded TensorFlow model.
6. Un-normalize model output (×50, since training normalizes y/50) then scale to 0-100 (×2) and save to `ProgressTracker` in the database.

### Deploying Models

When deploying new models, you must include ALL of:
- `models/ml_model_EsX_best.keras` — the trained Keras model
- `models/scaler_EsX.joblib` — the StandardScaler fitted during `03_extract_joint_features.py`
- Update `MAX_LENGTH_MAPPING` in `backend/main.py` with values from `model_config_EsX.json`

Generate scalers by running `training_models/export_scalers_colab.py` on Colab, then copy to `models/`.

### Real-time Feedback

The `/api/feedback/{exercise_id}` endpoint provides immediate movement feedback using angle-based biomechanical analysis. To modify this logic:

1. Extract `referenceJointValues` and `currentJointValues` from the request (51 floats each: 17 keypoints × [y, x, confidence]).
2. Calculate joint angles at key vertices (shoulders, elbows, knees) via `feedback_engine.py`.
3. Compare angles against reference with exercise-specific tolerance thresholds.
4. Return structured feedback with `primary_instruction`, `overall_accuracy`, and per-joint `feedback_details`.

## Additional Resources

### Reference Files

For detailed implementation patterns and schemas, consult:
- **`references/ml-pipeline.md`** - Detailed patterns for data preparation and sequence padding in `ml_wrapper.py`.
- **`references/api-reference.md`** - Complete REST API specification and session token authentication flows.
- **`references/database-schema.md`** - SQLAlchemy relationships and database structures.

### Scripts

Working utilities for development and testing:
- **`scripts/test-api.sh`** - End-to-end API testing workflows.
- **`scripts/seed-db.py`** - Database population utilities.
