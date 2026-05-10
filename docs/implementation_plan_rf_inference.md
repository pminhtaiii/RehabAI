# Implementation Plan: Integrate RF Ensemble into Backend Inference

## Overview

The training pipeline (`clinical_score_prediction_model.py`) already trains both an LSTM and a Random Forest (RF) model, saving them as `ml_model_{exercise}.keras` and `rf_model_{exercise}.joblib` respectively. The RF uses 14 summary statistics extracted from raw temporal features (mean, std, min, max, range, median, q25, q75, iqr, rms, mean_abs_diff, max_abs_diff, zero_crossing_rate, pct_nonzero) per feature dimension.

However, the backend inference (`ml_wrapper.py` + `main.py`) currently **only loads and uses the LSTM model**, completely ignoring the RF component. This plan integrates the RF model into the inference pipeline to match the ensemble behavior used during training/CV evaluation.

**Ensemble formula**: `final_score = α * lstm_pred + (1 - α) * rf_pred`

Where:
- `lstm_pred` is the LSTM output × 50 (denormalized from [0,1] to [0,50])
- `rf_pred` is the RF output directly (already in [0,50] scale)
- `α` is loaded from `model_config_{exercise}.json` (default: 0.5)

## Types

No new Pydantic models or type definitions needed. The existing `MotionResult` dataclass is unchanged.

Internal type additions in `ml_wrapper.py`:
- `_rf_models_cache: dict` — cache for loaded RF models (parallel to LSTM cache)
- `_model_configs_cache: dict` — cache for loaded model config JSONs (contains ensemble_alpha)

## Files

### Modified Files

1. **`backend/ml_wrapper.py`**
   - Add `extract_summary_features(feat_array)` function (port from training code)
   - Add `load_rf_model(exercise_id)` function
   - Add `load_model_config(exercise_id)` function
   - Modify `_predict_clinical_score_from_csv()` — not here, this stays in main.py
   
   Actually, `_predict_clinical_score_from_csv` is in `main.py`. Let me correct:

2. **`backend/main.py`**
   - Modify `lifespan()` — load RF models + configs at startup
   - Modify `_predict_clinical_score_from_csv()` — run RF prediction, blend with LSTM
   - Modify `diagnostic_endpoint()` — test ensemble behavior
   - Modify `health_check()` — verify RF models loaded
   - Update imports to include new ml_wrapper functions

### No New Files

All changes are within existing backend files.

### No Deleted Files

## Functions

### New Functions in `backend/ml_wrapper.py`

1. **`extract_summary_features(feat_array: np.ndarray) -> np.ndarray`**
   - Signature: `def extract_summary_features(feat_array: np.ndarray) -> np.ndarray`
   - Purpose: Extract 14 summary statistics from a single (T, F) feature array
   - Input: Raw feature array (T frames × F features), NOT scaled
   - Output: (1, F*14) summary vector
   - Must match training implementation exactly (clinical_score_prediction_model.py lines 502-535)
   - Statistics: mean, std, min, max, range, median, q25, q75, iqr, rms, mean_abs_diff, max_abs_diff, zero_crossing_rate, pct_nonzero

2. **`load_rf_model(exercise_id: str) -> RandomForestRegressor`**
   - Signature: `def load_rf_model(exercise_id: str)`
   - Purpose: Load the per-exercise RF model from `models/rf_model_{exercise_id}.joblib`
   - Caching: Use `_rf_models_cache` dict (same pattern as `load_scaler`)
   - Error handling: Raise `FileNotFoundError` if `.joblib` not found

3. **`load_model_config(exercise_id: str) -> dict`**
   - Signature: `def load_model_config(exercise_id: str) -> dict`
   - Purpose: Load `models/model_config_{exercise_id}.json` containing `ensemble_alpha`
   - Caching: Use `_model_configs_cache` dict
   - Returns: dict with at least `ensemble_alpha` key (default 0.5 if missing)

4. **`predict_ensemble(lstm_score: float, rf_score: float, alpha: float) -> float`**
   - Signature: `def predict_ensemble(lstm_score: float, rf_score: float, alpha: float) -> float`
   - Purpose: Blend LSTM and RF predictions
   - Formula: `alpha * lstm_score + (1 - alpha) * rf_score`
   - Clip result to [0, 50]

### Modified Functions in `backend/main.py`

1. **`lifespan(app: FastAPI)`**
   - Current: Loads only LSTM models + scalers
   - Change: Also load RF models (`rf_model_{exercise}.joblib`) and model configs (`model_config_{exercise}.json`)
   - Add to `ml_models_cache` or separate `rf_models_cache` / `model_configs_cache`
   - Log RF load status alongside LSTM

2. **`_predict_clinical_score_from_csv(csv_string, max_length, exercise_id, model, source)`**
   - Current: LSTM-only prediction with motion calibration
   - Change:
     1. After feature extraction (before scaling), save the raw features for RF
     2. After LSTM prediction + denormalization, extract summary features from raw features
     3. Run RF prediction on summary features
     4. Blend: `ensemble_score = alpha * lstm_score + (1 - alpha) * rf_score`
     5. Apply motion calibration on ensemble score
   - The RF input must be RAW features (before StandardScaler) — this matches training where `extract_summary_features(raw_features)` is called on unscaled data
   - Key: Need to capture the pre-scaler features in `prepare_data_with_motion()` or pass them separately

3. **`diagnostic_endpoint(exercise_id)`**
   - Current: Tests LSTM with synthetic data
   - Change: Also test RF + ensemble, report all three scores (LSTM, RF, ensemble)
   - Show alpha used

4. **`health_check()`**
   - Current: Checks LSTM model loaded
   - Change: Also check RF model loaded per exercise

### Modified Functions in `backend/ml_wrapper.py`

1. **`prepare_data_with_motion(df, max_length, exercise_id, source)`**
   - Current: Returns `(data, motion_result)` where `data` is the scaled+padded tensor
   - Change: Return `(data, motion_result, raw_features_for_rf)` — add the raw (pre-scaler, post-downsample) features as third return value
   - The raw features are needed for `extract_summary_features()` which must operate on unscaled data

2. **`prepare_data(df, max_length, exercise_id)`**
   - Current: Returns padded tensor only
   - Change: Also return raw features for RF (same pattern as above)

## Classes

No new classes. No modified classes. No removed classes.

## Dependencies

No new dependencies. `RandomForestRegressor` is already available via `scikit-learn` (already in `requirements.txt` as dependency of `sklearn`/`joblib`). `joblib` is already imported.

## Testing

1. **Unit test for `extract_summary_features()`**: Verify output shape is (1, F*14) for a given (T, F) input. Compare output against training implementation for same input.

2. **Integration test via `/api/diagnostic/{exercise_id}`**: 
   - Load backend with RF models
   - Hit diagnostic endpoint
   - Verify ensemble scores differ from LSTM-only scores
   - Verify RF scores are reasonable (not 0 or NaN)

3. **Manual verification**: 
   - Run `docker-compose up` with new code
   - Check startup logs confirm RF models loaded for all 5 exercises
   - Submit a test video CSV and verify ensemble score is returned

## Implementation Order

1. **Add `extract_summary_features()` to `ml_wrapper.py`** — Port from training code, standalone function, no dependencies
   - Verify: Function returns correct shape for test input

2. **Add `load_rf_model()` and `load_model_config()` to `ml_wrapper.py`** — New loader functions with caching
   - Verify: Functions load .joblib and .json files correctly

3. **Modify `prepare_data_with_motion()` to return raw features** — Add third return value
   - Verify: Existing callers still work (backward compatible with tuple unpacking)

4. **Modify `lifespan()` in `main.py`** — Load RF models + configs at startup
   - Verify: Startup logs show RF loaded for each exercise

5. **Modify `_predict_clinical_score_from_csv()` in `main.py`** — Implement ensemble blending
   - Verify: Endpoint returns ensemble score, raw LSTM score, raw RF score

6. **Modify `prepare_data()` in `ml_wrapper.py`** — Also return raw features (for non-motion path)
   - Verify: Shape validation still passes

7. **Update diagnostic endpoint** — Show LSTM vs RF vs ensemble comparison
   - Verify: Diagnostic response includes all three score types

8. **Update health check** — Verify RF models loaded
   - Verify: Health check reports RF status

9. **Update `test_model_inference()`** — Include RF in synthetic tests
   - Verify: Test results show ensemble behavior

10. **Test end-to-end with Docker** — Full pipeline verification
    - Verify: `docker-compose up` succeeds, clinical_score endpoint works with ensemble