# Feedback Pipeline Runbook and Optimization Plan

## 1) Scope
This document covers:
- How to run the feedback pipeline in this repository.
- Notebook to Python conversion status.
- Deep bottleneck and latency analysis.
- A concrete optimization plan with execution phases.
- Initial backend refactor work already applied.

## 2) How To Run Feedback Pipeline

### 2.1 Recommended path (Docker)
Prerequisites:
- Docker Desktop is running.
- Ports 5173 and 8000 are free.

Commands:
1. Go to repository root: C:/RehabAI
2. Start services:
   docker compose up -d --build
3. Check status:
   docker compose ps
4. Wait until backend health is healthy.
5. Open frontend: http://localhost:5173
6. Login and open an exercise page.
7. Start exercise session; frontend will call feedback endpoint periodically.

Health checks:
- Backend docs: http://localhost:8000/docs
- If request fails right after container restart, retry after model warmup completes.

### 2.2 Local path (without Docker)
Backend:
1. Create and activate Python env.
2. Install dependencies from backend/requirements.txt.
3. Run backend from backend folder:
   uvicorn main:app --host 0.0.0.0 --port 8000

Frontend:
1. In frontend folder: npm install
2. Start dev server: npm run dev
3. Ensure VITE_API_URL points to http://localhost:8000

## 3) Feedback Pipeline Architecture (Current)

### 3.1 Real-time feedback path
1. Frontend captures webcam frames and MoveNet keypoints in browser.
2. Every checkpoint, frontend sends:
   - referenceJointValues (51 floats)
   - currentJointValues (51 floats)
   to /api/feedback/{exercise_id}/
3. Backend endpoint computes:
   - angle-based biomechanical feedback (main signal)
   - optional DTW metric (legacy similarity signal)
4. Frontend renders top correction messages.

### 3.2 Clinical score path
1. Frontend sends captured CSV sequence to /api/clinical_score/{exercise_id}
2. Backend preprocesses sequence via ml_wrapper
3. Backend runs Keras model inference
4. Score is saved to ProgressTracker and returned

## 4) Notebook To Python Conversion Status
All notebooks were converted to Python files:
- eda/data_eda.ipynb -> eda/data_eda.py
- eda/data_profiling.ipynb -> eda/data_profiling.py
- feedback/dtw.ipynb -> feedback/dtw.py
- feedback/feedback_threshold_experiment.ipynb -> feedback/feedback_threshold_experiment.py
- feedback/movenet.ipynb -> feedback/movenet.py

Post-conversion cleanup completed:
- Notebook-only commands (! and %) were converted to comments/import-safe equivalents.
- All files compile successfully via Python compileall.

## 5) Bottleneck And Latency Analysis

### 5.1 High impact bottlenecks
A) Redirect overhead on feedback endpoint
- Frontend calls trailing slash route.
- Backend originally declared non-trailing slash route only.
- Result: extra redirect hop in hot path.

B) CPU-bound work inside async endpoints
- Angle calculations, DTW, CSV preprocessing, and model inference were executed in async handlers.
- This can block event loop under concurrency and increase tail latency.

C) DTW in hot real-time loop
- DTW is heavier than angle comparison and was computed every feedback request.
- Under multiple concurrent users this creates CPU pressure and jitter.

D) O(n) assignment check query pattern
- Assignment check previously loaded all assigned exercises then looped in Python.
- Adds avoidable DB and Python overhead.

### 5.2 Medium impact risks
A) Cold start delay
- Backend loads all TensorFlow models at startup.
- During cold start, docs/health endpoints can be temporarily unavailable.

B) N+1 query pattern in exercises endpoint
- Assigned exercises loaded, then one query per exercise.

C) Data model schema inconsistencies
- SQLAlchemy types and foreign keys are weakly aligned with actual values.
- SQLite tolerance hides this, but scaling/migration risk remains.

D) Model serialization compatibility risk
- Current .keras files may be saved with newer Keras metadata not fully compatible with runtime TF/Keras in container.
- Without fail-safe loading, this can crash backend startup.

### 5.3 Low impact risks
A) Unbounded frontend feedback history list growth in session.
B) Limited observability: no endpoint timing instrumentation by default.

## 6) Refactor Work Applied (Initial Optimization Batch)

### Backend
1) feedback endpoint compatibility and redirect removal
- Added both route forms:
  - /api/feedback/{exercise_id}
  - /api/feedback/{exercise_id}/

2) feedback endpoint CPU offload
- Moved angle-feedback generation to threadpool.
- Moved optional DTW calculation to threadpool.

3) optional DTW gating
- Added includeDtw in request schema.
- Default is false for lower latency.

4) assignment check optimization
- Replaced list-load plus Python loop with SQL EXISTS query.

5) clinical_score endpoint CPU offload
- Added helper for CSV parse + prepare_data + model predict.
- Entire heavy path executed in threadpool.
- Added csvString required validation.

6) feedback engine micro-optimizations
- Reduced temporary allocations in angle calculation.
- Precompiled rule joint indices once at import time.
- Switched keypoint parsing to np.asarray float32.

7) robust model file resolution
- Backend now searches both naming conventions:
   - ml_model_EsX.keras / ml_model_EsX.h5
   - ml_model_EsX_best.keras / ml_model_EsX_best.h5

8) fail-safe model loading at startup
- Model load failures are captured per exercise and logged as warnings.
- Backend no longer crashes if one or more model files are incompatible.
- clinical_score returns controlled 503 for exercises whose model failed to load.

### Frontend
1) feedback API request update
- includeDtw is sent as false to keep real-time path lightweight.

## 7) Detailed Execution Plan (Next Iterations)

### Phase 0 - Baseline and observability (1 day)
Goal: make latency measurable.
Tasks:
- Add timing logs around feedback and clinical_score stages.
- Capture p50, p95, p99 latency and request rate.
- Add basic request id correlation in logs.
Exit criteria:
- Dashboard or structured log report for both endpoints.

### Phase 1 - Hot path hardening (1-2 days)
Goal: stabilize real-time feedback under load.
Tasks:
- Keep DTW optional and disabled by default in all clients.
- Add strict payload validation for 51-value vectors.
- Add small timeout and guarded exception handling for feedback computation.
Exit criteria:
- No event loop blocking warnings.
- p95 latency stable under multi-user test.

### Phase 2 - Compute optimization (2-3 days)
Goal: lower CPU per request.
Tasks:
- Evaluate replacing tslearn DTW with faster approximation for optional mode.
- Batch or throttle feedback calls at client/server contract level.
- Consider worker process split for heavy ML scoring path.
Exit criteria:
- >= 30% CPU reduction at same QPS in benchmark.

### Phase 3 - Data and DB optimization (2 days)
Goal: reduce DB overhead and correctness risk.
Tasks:
- Remove N+1 query in exercises endpoint via join.
- Add indexes for hot lookup columns (session_token, user_id, exercise_id).
- Plan schema normalization migration for exercise and FK columns.
Exit criteria:
- Fewer SQL queries per request and stable query times.

### Phase 4 - Cold start and resiliency (2 days)
Goal: faster startup and safer deploy.
Tasks:
- Add model load readiness signal.
- Consider lazy model load for clinical_score path.
- Add startup probes with retry budget and clearer health endpoints.
Exit criteria:
- Predictable readiness and lower startup fail windows.

## 8) Suggested Immediate Next Actions
1. Add timing instrumentation in backend/main.py for feedback and clinical_score.
2. Load-test /api/feedback with includeDtw=false at concurrent users.
3. Refactor /api/exercises endpoint to remove N+1 query.
4. Prepare DB schema migration plan before production scaling.
