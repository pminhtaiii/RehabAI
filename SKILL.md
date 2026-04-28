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
- **Machine Learning**: Keras models (`ml_model_EsX.keras`) for exercise evaluation, processing joint features over time. Models output [0,1] scores (trained with sigmoid + y/100).

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

### Clinical Score Prediction

The `/api/clinical_score/{exercise_id}` endpoint processes CSV data containing joint movements. To modify this pipeline:

1. Validate incoming CSV strings via the `CsvStringModel` schema.
2. Reorder dataframe features using `reorder_dataframe()` from `backend/ml_wrapper.py`.
3. Prepare sequence data using `prepare_data()` with the exercise's specific `max_length`.
   - **CRITICAL**: `prepare_data()` applies a per-exercise `StandardScaler` loaded from `models/scaler_EsX.joblib`. These scalers MUST match the ones used during training. Without them, model output will be a constant ~53/100.
4. Execute inference using the pre-loaded TensorFlow model.
5. Scale the raw prediction by 100 (model outputs [0,1]) and save to `ProgressTracker` in the database.

### Deploying Models

When deploying new models, you must include BOTH:
- `models/ml_model_EsX_best.keras` — the trained Keras model
- `models/scaler_EsX.joblib` — the StandardScaler fitted during `03_extract_joint_features.py`

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
