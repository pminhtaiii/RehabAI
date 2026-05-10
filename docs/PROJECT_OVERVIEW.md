# RehabAI - Project Overview & Architecture

**Last Updated:** May 4, 2026

## 📋 Table of Contents

1. [Project Purpose](#project-purpose)
2. [Architecture Overview](#architecture-overview)
3. [Backend Structure](#backend-structure)
4. [Data Preparation Pipeline](#data-preparation-pipeline)
5. [Frontend Structure](#frontend-structure)
6. [ML Training & Models](#ml-training--models)
7. [Analysis & Feedback Tools](#analysis--feedback-tools)
8. [Key Technical Flows](#key-technical-flows)
9. [Technology Stack](#technology-stack)

---

## Project Purpose

**RehabAI** is an **AI-powered rehabilitation system** that provides real-time feedback for therapeutic exercises using pose detection and machine learning. It combines:

- **Real-time pose estimation** (MoveNet) running in the browser
- **Biomechanical feature extraction** from body joints
- **Clinical score prediction** using LSTM neural networks
- **Personalized feedback** based on exercise execution quality
- **Motion detection** to validate actual exercise performance

The system supports **5 therapeutic exercises (Es1-Es5)**:
- **Es1**: Arm lifts
- **Es2**: Lateral trunk tilt
- **Es3**: Trunk rotation
- **Es4**: Pelvis rotation
- **Es5**: Squatting movements

--

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Frontend (React 18 + TensorFlow.js)        │
│     - MoveNet Pose Detection (30 FPS LIGHTNING)         │
│     - Real-time Skeleton Visualization                  │
│     - WebSocket Feedback Stream                         │
└─────────────────┬───────────────────────────────────────┘
                  │ REST API + WebSocket
                  ▼
┌─────────────────────────────────────────────────────────┐
│        Backend (FastAPI + TensorFlow 2.21)              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ ML Inference Pipeline                              │ │
│  │ - Joint Feature Extraction                         │ │
│  │ - LSTM Model Inference                             │ │
│  │ - Motion Detection & Validation                    │ │
│  │ - Real-time Feedback Generation                    │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────────┘
                  │ SQLAlchemy ORM
                  ▼
        ┌─────────────────────┐
        │  SQLite Database    │
        │  - Users            │
        │  - Exercises        │
        │  - Progress Tracking│
        └─────────────────────┘
```

---

## Backend Structure (`backend/`)

### Core Application Files

#### **main.py** - FastAPI Server
- **Purpose**: Central REST API server with WebSocket support
- **Key Responsibilities**:
  - User authentication endpoints (`/api/login/`, `/api/signup/`)
  - Exercise data endpoints (list, details, assignment)
  - File upload handling
  - Real-time feedback WebSocket (`/ws/feedback`)
  - Model health checks (`/api/health`)
  - Model caching and lazy loading
  - Lifespan management (startup/shutdown)
- **Tech**: FastAPI, Uvicorn, CORS enabled

#### **models.py** - SQLAlchemy ORM Models
- **Purpose**: Database schema definition
- **Models**:
  - `User` - Authentication and profile data
  - `Exercise` - Exercise metadata and references
  - `AssignedExercise` - User-exercise relationships
  - `ProgressTracker` - Performance history and clinical scores
- **Relationships**: Many-to-many with tracking

#### **database.py** - Database Setup
- **Purpose**: SQLAlchemy configuration
- **Functions**:
  - SQLite database connection setup
  - Session factory initialization
  - Declarative base for ORM models
  - Database path: `/app/data/RehabAI.db` (Docker) or local directory
- **Pattern**: Using SQLAlchemy sessionmaker

### ML & Data Processing

#### **ml_wrapper.py** - ML Inference Pipeline
- **Purpose**: Core machine learning inference engine
- **Key Features**:
  - Lazy loads pre-trained Keras models and StandardScalers
  - Temporal data preprocessing:
    - 5-frame downsampling (capture rate normalization)
    - Padding to fixed 150-frame sequences
  - Model inference execution
  - Auto-detects output normalization (0-1 or 0-100 range)
  - Bridges raw input data to clinical scores (0-100 range)
- **Critical**: Ensures inference consistency with training pipeline

#### **joint_features.py** - Biomechanical Feature Extraction
- **Purpose**: Real-time extraction of biomechanical features from 3D pose
- **Per-Exercise Feature Sets**:
  - **Es1/Es2** (Arm Lifts, Lateral Tilt): 6 features
    - Elbow/shoulder angles, hand-shoulder ratios, torso tilt
  - **Es3** (Trunk Rotation): 9 features
    - Extended angle variations including rotation components
  - **Es4** (Pelvis Rotation): 2 features
    - Minimal set: torso tilt, knee-hip ratio
  - **Es5** (Squatting): 7 features
    - Comprehensive lower body angles and positions
- **Data Source**: MediaPipe 3D landmarks (12 keypoints)
- **Output**: Normalized feature vectors for ML model input

#### **motion_detector.py** - Motion Validation
- **Purpose**: Determine if user is actively performing exercise
- **Metrics**:
  - Temporal variance of joint positions
  - Range of motion (ROM) analysis
  - Frame-to-frame displacement
  - Motion quality confidence factors
- **Function**: Modulates clinical scores based on motion validity

#### **feedback_engine.py** - Real-time Feedback Generation
- **Purpose**: Generate immediate corrective feedback
- **Features**:
  - Rule-based feedback using biomechanical angle thresholds
  - Per-joint positioning corrections (e.g., shoulder flexion angles)
  - Vietnamese-language feedback messages
  - Immediate display to user during exercise
- **Integration**: Runs in WebSocket handler with low latency

### Data Management & Utilities

#### **data_wrapper.py** - Database Seeding
- **Purpose**: Initialize database with exercise definitions
- **Seeded Data**:
  - 5 exercise definitions (Es1-Es5)
  - Exercise metadata (names, descriptions)
  - Reference images and videos
  - Reference CSV files for biomechanical comparison
- **Timing**: Called on server startup

#### **utils.py** - Helper Functions
- **Purpose**: Reusable utilities for data operations
- **Functions**:
  - Exercise assignment queries
  - User-exercise relationship checks
  - Data validation helpers

#### **hashing.py** - Password Security
- **Purpose**: Secure password handling
- **Implementation**:
  - Uses bcrypt for hashing
  - passlib integration for verification
  - Salting and security best practices

### Diagnostic & Testing Tools

#### **diagnostic_check.py** - Inference Validation
- **Purpose**: Validate ML pipeline integrity before deployment
- **Checks**:
  - Scaler file availability (all 5 exercises)
  - Model loading success
  - Inference variability testing
  - Output range correctness (0-100)
- **Prevention**: Prevents "stuck at 53" inference failures

#### **check_health.py** - Backend Health Check
- **Purpose**: Simple HTTP health monitoring
- **Function**: Calls `/api/health` endpoint to verify backend status

#### **Test Files** (`test_*.py`)
- `test_deep_diagnostic.py` - Comprehensive pipeline diagnostics
- `test_es3_raw.py` - Exercise-specific diagnostic for Es3
- `test_pipeline_gap.py` - Gap analysis between training and inference

---

## Data Preparation Pipeline (`data-preparation/`)

### 01_extract_joint_positions.py
- **Input**: Exercise videos
- **Process**: 
  - Uses MediaPipe PoseLandmarker (Tasks API)
  - Extracts 3D pose landmarks across frames
  - Detects 12 key body joints
- **Output**: CSV files with 48 columns per frame
  - 12 joints × 4 channels (x, y, z, visibility)
- **Models**: Supports lite/full/heavy MediaPipe variants

### 02_prepare_dataset.py
- **Input**: Joint position CSVs + Excel clinical assessments
- **Process**:
  - Matches motion data with clinical labels
  - Creates metadata mapping
  - Structures data for feature extraction
- **Output**: Master metadata CSV (KiMoRe_final.csv)

### 03_extract_joint_features.py
- **Input**: Raw joint positions + metadata
- **Process**:
  - Computes biomechanical features (angles, distances, ratios)
  - Fits StandardScaler per exercise
  - Scales feature distributions
- **Output**: 
  - Per-exercise feature CSVs
  - `.joblib` scalers used at inference time
- **Critical**: Mirrors exact training pipeline for consistency

---

## Frontend Structure (`frontend/`)

### Main Application Layout

#### **App.jsx** - React Router Setup
- **Routes**:
  - `/` - Login page
  - `/exercises` - Patient home (exercise dashboard)
  - `/exercises/:id` - Individual exercise interface
- **Purpose**: Application navigation foundation
- **Tech**: React Router v6

#### **api.js** - HTTP Client Configuration
- **Purpose**: Centralized API communication
- **Features**:
  - Axios instance with backend base URL
  - Environment variable configuration
  - Cross-domain credentials handling
  - Request/response interceptors ready

### Pages

#### **Login.jsx** - Authentication
- **Features**:
  - Signup form with user registration
  - Login form with credentials validation
  - API calls to `/api/login/` and `/api/signup/`
  - Material Design styling
  - RehabAI branding
- **Flow**: Authenticate → Redirect to exercises

#### **PatientHome.jsx** - Exercise Dashboard
- **Layout**:
  - Featured exercise card (prominent)
  - Grid of assigned exercises
- **Data Source**: `/api/exercises/` endpoint
- **Interaction**: Click to navigate to exercise page

#### **Exercise/Exercise.jsx** - Exercise Execution Interface
- **Core Interface** for patient rehabilitation
- **Real-time Features**:
  - Webcam pose detection (MoveNet LIGHTNING @ 30 FPS)
  - Reference video playback (side-by-side comparison)
  - Live skeleton visualization on canvas overlay
  - WebSocket feedback stream with real-time messages
- **Monitoring**:
  - Motion detection alerts
  - Frame capture at 60 FPS (buffered)
  - Performance metrics collection
- **Output**: 
  - Final clinical score display (0-100)
  - Performance summary save

### Components (`src/components/`)

#### **VideoDisplay.jsx**
- Shows reference exercise video and user's live canvas
- Side-by-side layout for exercise comparison
- Canvas overlay for skeleton drawing

#### **FeedbackDisplay.jsx**
- Real-time feedback message stream
- Horizontal scrollable card layout
- Dynamic messages ("Keep going!" during motion, final score on completion)

#### **TopBar.jsx**
- Navigation header with RehabAI logo
- User menu integration

#### **AvatarMenu.jsx**
- User profile/account options
- Logout functionality

#### **TimeDisplay.jsx**
- Exercise timer countdown
- Elapsed time tracking

#### **ControlButtons.jsx**
- Start exercise button
- Stop/pause controls
- Save performance button

### Exercise Utilities (`src/pages/Exercise/`)

#### **renderer_canvas2d.js**
- **Purpose**: Canvas rendering engine
- **Functionality**:
  - Draws skeleton keypoints from MoveNet detections
  - Renders limb connections
  - Real-time visualization on HTML5 canvas
  - Overlayed on video feed

#### **utils.js**
- **MiniDataFrame**: In-memory CSV data structure for frame buffering
- **CSV Parser**: Parses reference exercise CSVs
- **Frame Management**: 60 FPS frame buffer collection
- **Data Structures**: Joint coordinate storage and manipulation

### Frontend Configuration

#### **package.json** - Dependencies
- **React**: v18.x
- **Routing**: react-router-dom v6.x
- **ML**: 
  - @tensorflow/tfjs v4.15
  - @tensorflow-models/movenet (pose detection)
- **UI**: 
  - Material-UI for components
  - Tailwind CSS v4.2 for styling
- **Build**: Vite (fast development)
- **Total**: 50+ dependencies

#### **vite.config.js** - Build Configuration
- React plugin for JSX
- Development server setup
- Optimized production builds

#### **tailwind.config.js** - CSS Framework
- Custom RehabAI theme
- Color schemes
- Component styling

#### **postcss.config.js** - CSS Processing
- Tailwind plugin pipeline
- Post-processing for browser compatibility

### Public Assets (`public/`)
- **Reference CSVs**: `E_ID*_Es*.csv`
  - Ground truth biomechanical data
  - Used for live comparison during exercise
  - Patient-specific exercise references

---

## ML Training & Models

### clinical_score_prediction_model.py
- **Architecture**: LSTM v6 (evolved from v5)
  ```
  Input (batch, 150, features)
    ↓
  LSTM(32, dropout=0.3)
    ↓
  LSTM(16, dropout=0.3)
    ↓
  Dense(8, activation='relu')
    ↓
  Dense(1, activation='sigmoid')  → Output [0,1]
  ```
- **Output Scaling**: [0,1] → [0,100] clinical score range
- **Loss Function**: 
  - Primary: Mean Absolute Error (MAE)
  - Alternative: Huber (robust to outliers)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler
- **Evaluation**:
  - 5-fold cross-validation
  - Spearman correlation analysis
  - MAE/RMSE on test sets

### evaluate_models.py
- Post-training evaluation and validation
- Comparative analysis across exercise models
- Performance reporting

### Model Artifacts (`models/`)
**Pre-trained Models**:
- `ml_model_Es1.keras` through `ml_model_Es5.keras`
  - Keras format (.keras extension)
  - TensorFlow 2.21 compatible
  - Output: Clinical score (0-50 or normalized 0-1)

**Scalers**:
- `scaler_Es1.joblib` through `scaler_Es5.joblib`
  - StandardScaler per exercise
  - Fitted on training feature distributions
  - Used for inference-time normalization

---

## Analysis & Feedback Tools

### EDA (Exploratory Data Analysis)

#### **data_eda.py**
- **Purpose**: Understand data distributions and patterns
- **Visualizations**:
  - Joint position plots
  - Feature comparisons (expert vs. back pain patients)
  - Optional smoothing analysis
- **Output**: Saved plots for data understanding

#### **data_profiling.py**
- Dataset distribution profiling
- Statistical summaries
- Data quality reports

### Feedback Development

#### **feedback_threshold_experiment.py**
- **Purpose**: Optimize feedback sensitivity
- **Method**: DTW (Dynamic Time Warping) analysis
  - Distance calculation between ground truth and patient performance
  - Threshold optimization for feedback triggers
- **Goal**: Tune feedback timing and intensity

#### **movenet.py**
- MoveNet integration utilities
- Keypoint mapping helpers
- Model initialization

#### **dtw.py**
- Dynamic Time Warping implementation
- Sequence comparison algorithm
- Distance metrics for motion analysis

---

## Key Technical Flows

### 1. User Exercise Execution Flow (Frontend)

```
┌─────────────────────────────────────────────────────────┐
│  Patient Login                                          │
└────────────────┬────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Select Exercise from Dashboard                         │
└────────────────┬────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Load Exercise Page & Initialize MoveNet               │
│  - Load reference video                                 │
│  - Initialize pose detection model (LIGHTNING)          │
│  - Setup canvas for skeleton rendering                  │
└────────────────┬────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Capture Video Frames (30 FPS MoveNet)                 │
│  - Detect pose landmarks                               │
│  - Extract joint coordinates (12 keypoints)            │
│  - Buffer frames (60 FPS throughput)                    │
└────────────────┬────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Post Frames to Backend (WebSocket)                    │
│  - Stream pose data in real-time                        │
│  - Maintain persistent connection                       │
└────────────────┬────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Display Real-time Feedback                            │
│  - Show skeleton on canvas                              │
│  - Display feedback messages                            │
│  - Render motion validation alerts                      │
└────────────────┬────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Exercise Completion                                    │
│  - Show final clinical score                            │
│  - Display performance summary                          │
│  - Save progress to database                            │
└─────────────────────────────────────────────────────────┘
```

### 2. ML Inference Pipeline (Backend)

```
Raw Frame Input (640×480 RGB)
         │
         ▼
Extract MediaPipe Joints
(12 keypoints with x,y,z,visibility)
         │
         ▼
Temporal Preprocessing
┌─────────────────────────────────────┐
│ - 5-frame downsampling              │
│ - Normalization to 150-frame length │
│ - Padding/truncation                │
└──────────┬──────────────────────────┘
           │
           ▼
Extract Biomechanical Features
┌─────────────────────────────────────┐
│ Per-exercise feature set:           │
│ - Joint angles (elbow, shoulder)    │
│ - Distance ratios                   │
│ - Position deltas                   │
│ - Rotation angles                   │
└──────────┬──────────────────────────┘
           │
           ▼
Feature Normalization
(StandardScaler per exercise)
           │
           ▼
LSTM Model Inference
(2-layer LSTM + Dense layers)
           │
           ▼
Output Scaling
(Normalize 0-1 → 0-100 range)
           │
           ▼
Motion Validation
(Confidence adjustment)
           │
           ▼
Biomechanical Feedback Generation
(Rule-based angle corrections)
           │
           ▼
Send Results via WebSocket
(Clinical score + feedback messages)
```

### 3. Feature Engineering Example (Es1: Arm Lifts)

```
Input: 12 MediaPipe Keypoints (NOSE, L_SHOULDER, R_SHOULDER, 
        L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_KNEE, R_KNEE,
        L_ANKLE, R_ANKLE, L_HIP, R_HIP)
         │
         ▼
Extract 6 Features:
┌────────────────────────────────────────┐
│ 1. Left Elbow Angle                    │
│    (L_SHOULDER → L_ELBOW → L_WRIST)    │
│                                        │
│ 2. Right Elbow Angle                   │
│    (R_SHOULDER → R_ELBOW → R_WRIST)    │
│                                        │
│ 3. Hand-Shoulder Distance Ratio        │
│    (WRIST_Y / SHOULDER_Y)              │
│                                        │
│ 4. Torso Tilt                          │
│    (Angle between shoulders and hips)  │
│                                        │
│ 5. Hand Tilt Asymmetry                 │
│    (Difference in wrist heights)       │
│                                        │
│ 6. Elbow Flexion Difference            │
│    (Right angle - Left angle)          │
└────────────────────────────────────────┘
         │
         ▼
Scale with StandardScaler
         │
         ▼
Feed to LSTM Model
```

---

## Technology Stack

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.x | UI framework |
| React Router | 6.x | Client-side routing |
| TensorFlow.js | 4.15 | Browser-based ML |
| MoveNet | Latest | Pose detection |
| Tailwind CSS | 4.2 | Styling framework |
| Material-UI | Latest | UI components |
| Vite | Latest | Build tool |
| Axios | Latest | HTTP client |

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| FastAPI | Latest | REST API framework |
| TensorFlow | 2.21 | ML inference |
| Keras | 3.x | Neural network models |
| SQLAlchemy | Latest | ORM |
| SQLite | Latest | Database |
| scikit-learn | Latest | Feature scaling |
| pandas | Latest | Data manipulation |
| joblib | Latest | Model serialization |
| bcrypt | Latest | Password hashing |
| uvicorn | Latest | ASGI server |

### Data Processing
| Technology | Purpose |
|-----------|---------|
| MediaPipe | Pose landmark extraction |
| NumPy | Numerical computation |
| tslearn | DTW implementation |
| pandas | Data analysis |
| OpenCV | Video processing |

### DevOps
| Technology | Purpose |
|-----------|---------|
| Docker | Containerization |
| Docker Compose | Multi-container orchestration |
| Python 3.10 | Backend runtime |
| Node.js | Frontend runtime |

---

## Quick Start Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker Compose
```bash
docker-compose up -d
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

---

## Key Files Reference

### Critical Files for Development
- **Backend Entry**: `backend/main.py`
- **ML Inference**: `backend/ml_wrapper.py`
- **Feature Extraction**: `backend/joint_features.py`
- **Frontend App**: `frontend/src/App.jsx`
- **Exercise UI**: `frontend/src/pages/Exercise/Exercise.jsx`
- **Database Models**: `backend/models.py`

### Configuration Files
- **Backend Dependencies**: `backend/requirements.txt`
- **Frontend Dependencies**: `frontend/package.json`
- **Docker Setup**: `docker-compose.yml`

### Documentation
- **Implementation Details**: `docs/implementation_plan.md`
- **Model Compatibility**: `docs/model_backend_compatibility_analysis.md`
- **Feedback System**: `docs/feedback_pipeline_plan.md`

---

## Contact & Support

For questions about this project architecture, refer to:
- Implementation details in `docs/implementation_plan.md`
- Model analysis in `docs/model_backend_compatibility_analysis.md`
- Feedback design in `docs/feedback_pipeline_plan.md`

---

**Project Status**: Active Development  
**Last Modified**: May 2026
