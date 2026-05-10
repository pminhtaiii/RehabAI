# Importing the necessary libraries
from contextlib import asynccontextmanager
from datetime import datetime
import logging
from typing import Annotated, List, Optional
from uuid import uuid4, UUID

import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
from tslearn.metrics import dtw
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Response,
    Request,
    Cookie,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from io import StringIO

from database import SessionLocal, engine
from hashing import get_hashed_password, verify_password
from ml_wrapper import prepare_data, prepare_data_with_motion, reorder_dataframe, get_dataframe_cols, load_scaler, test_model_inference, load_rf_model, load_model_config, extract_summary_features, predict_ensemble
from motion_detector import calibrate_score
from data_wrapper import initialize_db
from feedback_engine import generate_feedback
import models
from utils import is_exercise_assigned_to_user

logger = logging.getLogger("rehabai")

# ---- Model cache: loaded once at startup, reused across all requests ----
ml_models_cache: dict = {}
model_load_errors: dict = {}

# Model extensions supported: .keras (modern) or .h5 (legacy)

MODELS_DIRECTORY = "models/"

MAX_LENGTH_MAPPING = {
    "Es1": 150,
    "Es2": 150,
    "Es3": 150,
    "Es4": 150,
    "Es5": 150,
}


def _get_model_input_tail_shape(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0] if input_shape else None
    if input_shape is None:
        return None
    return tuple(input_shape[1:])


def _validate_preprocess_shape(exercise_id: str, model):
    max_length = MAX_LENGTH_MAPPING[exercise_id]
    cols = get_dataframe_cols()
    dummy_df = pd.DataFrame(
        np.zeros((600, len(cols)), dtype=np.float32),
        columns=cols,
    )
    prepared = prepare_data(dummy_df, max_length, exercise_id)

    expected_tail = _get_model_input_tail_shape(model)
    actual_tail = tuple(prepared.shape[1:])
    if expected_tail and actual_tail != expected_tail:
        raise ValueError(
            f"Preprocess shape mismatch for {exercise_id}: "
            f"expected {expected_tail}, got {actual_tail}"
        )


def _resolve_model_path(exercise_id: str):
    """Resolve model file path for an exercise across known naming conventions."""
    candidates = [
        f"ml_model_{exercise_id}.keras",
        f"ml_model_{exercise_id}_best.keras",
        f"ml_model_{exercise_id}.h5",
        f"ml_model_{exercise_id}_best.h5",
    ]
    for filename in candidates:
        model_path = os.path.join(MODELS_DIRECTORY, filename)
        if os.path.exists(model_path):
            return model_path
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models and scalers into memory once when the server starts."""
    model_load_errors.clear()
    for exercise_id in ["Es1", "Es2", "Es3", "Es4", "Es5"]:
        # Pre-load scalers (critical for correct inference)
        try:
            load_scaler(exercise_id)
            print(f"[startup] Scaler loaded for {exercise_id}")
        except FileNotFoundError as e:
            model_load_errors[exercise_id] = f"Scaler missing: {e}"
            print(f"[startup] WARNING: {e}")
            continue  # Skip model loading if scaler is missing

        path_to_load = _resolve_model_path(exercise_id)

        if path_to_load:
            try:
                model = tf.keras.models.load_model(
                    path_to_load, compile=False
                )
                _validate_preprocess_shape(exercise_id, model)
                ml_models_cache[exercise_id] = model
                print(
                    f"[startup] Loaded model for {exercise_id} from {path_to_load} "
                    f"with input {model.input_shape}"
                )
            except Exception as exc:
                model_load_errors[exercise_id] = str(exc)
                print(
                    f"[startup] WARNING: Failed to load model for {exercise_id} "
                    f"from {path_to_load}: {exc}"
                )
        else:
            print(
                f"[startup] WARNING: No model file found for {exercise_id} "
                f"(checked default and _best .keras/.h5 names)"
            )

        load_rf_model(exercise_id)
        load_model_config(exercise_id)

    yield
    ml_models_cache.clear()
    model_load_errors.clear()
    print("[shutdown] ML models released.")


# Create an instance of the FastAPI class
app = FastAPI(lifespan=lifespan)


# Define the SessionData model
class SessionData(BaseModel):
    username: str
    sid: str


# Define the list of allowed origins for CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://frontend:5173",
]

# Add CORS middleware to the FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow the defined origins
    allow_credentials=True,  # Allow cookies to be included in the requests
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Define the UserCreateBase model
class UserCreateBase(BaseModel):
    username: str
    password: str


# Define the UserCreateModel model
class UserCreateModel(UserCreateBase):
    id: int

    class Config:
        orm_mode = True


# Define the UserLoginBase model
class UserLoginBase(BaseModel):
    username: str
    password: str


# Define the UserLoginModel model
class UserLoginModel(UserLoginBase):
    id: int

    class Config:
        orm_mode = True


# Define the CsvStringModel model
class CsvStringModel(BaseModel):
    csvString: Optional[str]


# Define the FeedbackRequest model (structured body instead of bare List params)
class FeedbackRequest(BaseModel):
    referenceJointValues: List[float]
    currentJointValues: List[float]
    includeDtw: bool = False


def _predict_clinical_score_from_csv(csv_string: str, max_length: int, exercise_id: str, model, source: str = "webcam"):
    """CPU-bound preprocessing + inference for clinical score endpoint.

    Args:
        source: "video" for offline KiMoRe (25fps), "webcam" for live (30fps).
                Affects temporal downsampling stride to match training fps.

    Returns dict with raw_score, calibrated_score, and motion metrics.
    """
    csv_string_io = StringIO(csv_string)
    raw_data = pd.read_csv(csv_string_io, sep=",")

    # --- DEBUG: CSV parsing ---
    print(f"\n[DEBUG][{exercise_id}] CSV rows={len(raw_data)}, cols={len(raw_data.columns)}")
    print(f"[DEBUG][{exercise_id}] CSV columns: {list(raw_data.columns[:8])}...")

    raw_data_ordered = reorder_dataframe(raw_data)

    # --- DEBUG: After reorder ---
    nan_after_reorder = raw_data_ordered.isna().sum().sum()
    print(f"[DEBUG][{exercise_id}] After reorder: NaN count={nan_after_reorder}")

    # Use motion-aware preparation pipeline
    prepared_data, motion_result, raw_features_for_rf = prepare_data_with_motion(
        raw_data_ordered, max_length, exercise_id, source=source
    )

    expected_tail = _get_model_input_tail_shape(model)
    actual_tail = tuple(prepared_data.shape[1:])
    if expected_tail and actual_tail != expected_tail:
        raise ValueError(
            f"Input shape mismatch for {exercise_id}: expected {expected_tail}, got {actual_tail}"
        )

    raw_prediction = model.predict(prepared_data, verbose=0)
    raw_out = float(raw_prediction.flatten()[0])

    if raw_out <= 1.5:
        lstm_score = raw_out * 50.0
    else:
        lstm_score = raw_out
    lstm_score = float(np.clip(lstm_score, 0, 50))

    rf_model = load_rf_model(exercise_id)
    model_config = load_model_config(exercise_id)
    alpha = model_config.get("ensemble_alpha", 0.5)

    if rf_model is not None:
        rf_input = extract_summary_features(raw_features_for_rf)
        rf_score = float(rf_model.predict(rf_input)[0])
        rf_score = float(np.clip(rf_score, 0, 50))
        raw_score = predict_ensemble(lstm_score, rf_score, alpha)
        print(f"[ENSEMBLE][{exercise_id}] LSTM={lstm_score:.2f}, RF={rf_score:.2f}, "
              f"alpha={alpha:.2f}, blended={raw_score:.2f}")
    else:
        raw_score = lstm_score
        print(f"[ENSEMBLE][{exercise_id}] RF not available, LSTM-only={lstm_score:.2f}")

    print(f"[DEBUG][{exercise_id}] Motion active={motion_result.is_active}, "
          f"energy={motion_result.motion_energy:.3f}, "
          f"quality_factor={motion_result.quality_factor:.3f}")

    return {
        "raw_score": round(raw_score, 2),
        "calibrated_score": calibrate_score(raw_score, motion_result),
        "ensemble_alpha": alpha,
        "rf_available": rf_model is not None,
        "lstm_score": round(lstm_score, 2),
        "rf_score": round(rf_score, 2) if rf_model is not None else None,
        "motion_detected": motion_result.is_active,
        "motion_energy": motion_result.motion_energy,
        "quality_factor": motion_result.quality_factor,
        "motion_details": {
            "variance_score": motion_result.variance_score,
            "rom_score": motion_result.rom_score,
            "displacement_score": motion_result.frame_displacement_score,
        },
    }


@app.get("/api/health")
async def health_check():
    models_status = {}
    for ex_id in ["Es1", "Es2", "Es3", "Es4", "Es5"]:
        rf_model = load_rf_model(ex_id)
        model_config = load_model_config(ex_id)
        models_status[ex_id] = {
            "model_loaded": ex_id in ml_models_cache,
            "rf_loaded": rf_model is not None,
            "ensemble_alpha": model_config.get("ensemble_alpha", 0.5),
            "error": model_load_errors.get(ex_id),
        }
    all_loaded = all(s["model_loaded"] for s in models_status.values())
    return JSONResponse(
        status_code=200 if all_loaded else 503,
        content={
            "status": "healthy" if all_loaded else "degraded",
            "models": models_status,
        },
    )


# Define a database dependency
def get_db():
    db = SessionLocal()  # Create a new database session
    try:
        yield db  # Yield the database session to the dependency
    finally:
        db.close()  # Close the database session


db_dependency = Annotated[Session, Depends(get_db)]

# Create all the tables in the database
models.Base.metadata.create_all(bind=engine)

initialize_db()


# Function to get the current user from the authentication token
async def get_current_user(request: Request, db: db_dependency):
    # Get the session token from the cookies
    session_id = request.cookies.get("session_token")

    # If there is no session token, raise an exception
    if not session_id:
        raise HTTPException(
            status_code=401, detail="Not authenticated. Session token not found."
        )
    # Retrieve the user with the matching session token
    user = (
        db.query(models.User)
        .filter(models.User.session_token == str(session_id))
        .first()
    )
    # If there is no user with the matching session token, raise an exception
    if not user:
        print(session_id)
        raise HTTPException(
            status_code=401, detail="Not authenticated. Invalid session token."
        )
    return user


# Endpoints


@app.post("/api/signup/", response_model=UserCreateModel)
async def signup(user: UserCreateBase, db: db_dependency):
    # Check if the username already exists
    existing_user = (
        db.query(models.User).filter(models.User.username == user.username).first()
    )
    # If the username already exists, raise an exception
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Hash the password
    hashed_password = get_hashed_password(user.password)

    # Generate a unique user ID
    user_count = db.query(models.User).count() + 1
    u_id = f"U{user_count}"

    # Create a new user
    db_user = models.User(
        user_id=u_id, username=user.username, password=hashed_password
    )

    # Assigning all exercises to the new user
    exercises = db.query(models.Exercise).all()
    for exercise in exercises:
        db_assigned_exercise = models.AssignedExercise(
            user_id=db_user.user_id,
            exercise_id=exercise.exercise_id,
            date=datetime.now(),
        )
        db.add(db_assigned_exercise)

    # Add the new user to the database
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@app.post("/api/login/")
async def login(user: UserLoginBase, db: db_dependency, response: Response):
    # Check if the username exists in the database
    db_user = (
        db.query(models.User).filter(models.User.username == user.username).first()
    )
    # If the username does not exist, raise an exception
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    # If the password is incorrect, raise an exception
    if not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Incorrect password")

    # Set the session token in the response cookie
    session = uuid4()
    db_user.session_token = str(session)
    db.commit()
    db.refresh(db_user)

    response.set_cookie(key="session_token", value=session, httponly=True)
    return {"message": "Login successful", "session_id": str(session)}


@app.get("/api/exercises/")
async def get_exercises(
    db: db_dependency, current_user: models.User = Depends(get_current_user)
):
    # Get the user's assigned exercises
    assigned_exercises = (
        db.query(models.AssignedExercise)
        .filter(models.AssignedExercise.user_id == current_user.user_id)
        .all()
    )
    # Get the exercise data for each assigned exercise
    exercises = []
    for assigned_exercise in assigned_exercises:
        exercise = (
            db.query(models.Exercise)
            .filter(models.Exercise.exercise_id == assigned_exercise.exercise_id)
            .first()
        )
        exercises.append(exercise)

    return exercises


@app.post("/api/feedback/{exercise_id}")
@app.post("/api/feedback/{exercise_id}/")
async def get_feedback(
    exercise_id: str,
    db: db_dependency,
    body: FeedbackRequest,
    current_user: models.User = Depends(get_current_user),
):
    # Ensure user can only request feedback for assigned exercises.
    if not is_exercise_assigned_to_user(db, current_user.user_id, exercise_id):
        raise HTTPException(status_code=403, detail="Exercise not assigned to user")

    # Offload CPU-bound angle calculations away from event loop.
    feedback = await run_in_threadpool(
        generate_feedback,
        exercise_id,
        body.referenceJointValues,
        body.currentJointValues,
    )

    dtw_values = []
    if body.includeDtw:
        # DTW is optional because it is the heaviest operation in this endpoint.
        dtw_value = await run_in_threadpool(
            dtw,
            body.currentJointValues,
            body.referenceJointValues,
        )
        dtw_values = [dtw_value]

    result = {
        "feedback_dtw": dtw_values,
        **feedback,  # primary_instruction, overall_accuracy, feedback_details, ...
    }
    return JSONResponse(content=result)


@app.post("/api/clinical_score/{exercise_id}")
async def clinical_score(
    exercise_id: str,
    db: db_dependency,
    csv_data: CsvStringModel,
    current_user: models.User = Depends(get_current_user),
):
    # Get the user's assigned exercises and check if the chosen exercise is among the exercises assigned to the user
    if not is_exercise_assigned_to_user(db, current_user.user_id, exercise_id):
        raise HTTPException(status_code=403, detail="Exercise not assigned to user")

    # Use pre-loaded model from cache (loaded once at startup)
    model = ml_models_cache.get(exercise_id)
    if not model:
        load_error = model_load_errors.get(exercise_id)
        if load_error:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model failed to load for exercise {exercise_id}. "
                    f"Please check server model compatibility."
                ),
            )
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for exercise {exercise_id}",
        )
    max_length = MAX_LENGTH_MAPPING.get(exercise_id, 0)

    if not csv_data.csvString:
        raise HTTPException(status_code=400, detail="csvString is required")

    result = await run_in_threadpool(
        _predict_clinical_score_from_csv,
        csv_data.csvString,
        max_length,
        exercise_id,
        model,
    )

    # Use calibrated score (penalized if no/low movement)
    final_score = result["calibrated_score"]

    # Checking if there are previous entries for the same user and exercise
    previous_entry = (
        db.query(models.ProgressTracker)
        .filter(
            models.ProgressTracker.user_id == current_user.user_id,
            models.ProgressTracker.exercise_id == exercise_id,
        )
        .order_by(models.ProgressTracker.date.desc())
        .first()
    )

    # Update the performance count
    if previous_entry:
        performance_count = previous_entry.performance_count + 1
    else:
        performance_count = 1

    # Store the calibrated clinical score in the database
    new_clinical_score = models.ProgressTracker(
        user_id=current_user.user_id,
        exercise_id=exercise_id,
        date=datetime.now(),
        score=final_score,
        performance_count=performance_count,
    )
    db.add(new_clinical_score)
    db.commit()
    db.refresh(new_clinical_score)

    return JSONResponse(content={
        "clinical_score": [[final_score]],
        "raw_score": result["raw_score"],
        "lstm_score": result["lstm_score"],
        "rf_score": result["rf_score"],
        "ensemble_alpha": result["ensemble_alpha"],
        "rf_available": result["rf_available"],
        "motion_detected": result["motion_detected"],
        "motion_energy": result["motion_energy"],
        "quality_factor": result["quality_factor"],
        "motion_details": result["motion_details"],
    })


# ---- Diagnostic Endpoint ----


@app.get("/api/diagnostic/{exercise_id}")
async def diagnostic_endpoint(exercise_id: str):
    """Run synthetic inference tests to check if the model produces varied scores.
    
    This endpoint tests the model with 3 different inputs:
    1. All-zeros (no movement)
    2. Standing pose (static)
    3. Arm-raise exercise (dynamic)
    
    If all 3 produce nearly identical scores (~53), the model has collapsed.
    """
    model = ml_models_cache.get(exercise_id)
    if not model:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model not loaded for {exercise_id}",
                     "available_models": list(ml_models_cache.keys()),
                     "load_errors": model_load_errors},
        )
    max_length = MAX_LENGTH_MAPPING.get(exercise_id, 0)

    try:
        results = await run_in_threadpool(
            test_model_inference, model, exercise_id, max_length
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

    # Diagnosis
    diagnosis = {
        "exercise_id": exercise_id,
        "model_input_shape": str(model.input_shape),
        "raw_scores": results["all_scores"],
        "calibrated_scores": results.get("calibrated_scores", {}),
        "raw_score_range": round(results["score_range"], 2),
        "calibrated_score_range": round(results.get("calibrated_range", 0), 2),
        "model_is_responsive": results["model_is_responsive"],
        "motion_analysis": {
            "no_movement": results.get("zeros_motion", {}),
            "standing_still": results.get("standing_motion", {}),
            "arm_raise": results.get("arm_raise_motion", {}),
        },
    }

    if not results["model_is_responsive"]:
        diagnosis["diagnosis"] = (
            "MODEL HAS LIMITED DISCRIMINATION — but motion calibration compensates. "
            "Raw scores vary by only {:.1f} points, but calibrated scores vary by {:.1f} points. "
            "The motion detection gate penalizes idle/no-movement inputs.".format(
                results["score_range"],
                results.get("calibrated_range", 0),
            )
        )
    else:
        diagnosis["diagnosis"] = (
            f"Model IS responsive. Raw score varies by {results['score_range']:.1f} points. "
            f"Motion calibration further separates active vs idle inputs."
        )

    return JSONResponse(content=diagnosis)


# ---- WebSocket Real-Time Feedback Channel ----


@app.websocket("/ws/session/{exercise_id}")
async def exercise_session_ws(websocket: WebSocket, exercise_id: str):
    """WebSocket for clinical score prediction.

    Message types (client → server):
      - {"type": "clinical_score", "csvString": "..."}

    Response types (server → client):
      - {"type": "clinical_score", "score": float, "raw_score": float,
         "motion_detected": bool, "motion_energy": float, "quality_factor": float}
      - {"type": "error", "message": str}

    Note: Real-time feedback has been removed (low reliability with Spearman ρ < 0.3).
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for exercise {exercise_id}")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            start_time = time.monotonic()

            try:
                if msg_type == "clinical_score":
                    csv_string = data.get("csvString", "")
                    if not csv_string:
                        await websocket.send_json({
                            "type": "error",
                            "message": "csvString is required",
                        })
                        continue

                    model = ml_models_cache.get(exercise_id)
                    if not model:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Model not available for {exercise_id}",
                        })
                        continue

                    max_length = MAX_LENGTH_MAPPING.get(exercise_id, 0)
                    score_result = await run_in_threadpool(
                        _predict_clinical_score_from_csv,
                        csv_string,
                        max_length,
                        exercise_id,
                        model,
                    )

                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    await websocket.send_json({
                        "type": "clinical_score",
                        "score": score_result["calibrated_score"],
                        "raw_score": score_result["raw_score"],
                        "motion_detected": score_result["motion_detected"],
                        "motion_energy": score_result["motion_energy"],
                        "quality_factor": score_result["quality_factor"],
                        "latency_ms": round(elapsed_ms, 1),
                    })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })

            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for exercise {exercise_id}")


@app.post("/api/logout/")
def logout(
    response: Response,
    db: db_dependency,
    current_user: models.User = Depends(get_current_user),
):
    # Delete the session token from the response cookie
    response.delete_cookie("session_token")

    # Remove the session token from the user
    current_user.session_token = None
    db.commit()

    return {"message": "Logout successful"}
