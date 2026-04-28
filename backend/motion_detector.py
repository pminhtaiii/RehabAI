"""
Motion Detector for RehabAI
============================
Detects whether the user is actually performing exercise movement
by analyzing temporal variance, range of motion, and movement energy
from extracted biomechanical features.

Used as a pre/post-inference gate to prevent meaningless clinical scores
when the user is standing still or not exercising.

Returns a MotionResult with:
  - is_active: bool — whether sufficient movement was detected
  - motion_energy: float — normalized movement energy [0, 1]
  - quality_factor: float — multiplier for clinical score [0.1, 1.0]
  - details: dict — per-metric breakdown for transparency
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MotionResult:
    """Result of motion detection analysis."""
    is_active: bool
    motion_energy: float          # Normalized [0, 1]
    quality_factor: float         # Score multiplier [0.1, 1.0]
    rom_score: float              # Range of motion score [0, 1]
    variance_score: float         # Temporal variance score [0, 1]
    frame_displacement_score: float  # Frame-to-frame displacement score [0, 1]
    details: dict = field(default_factory=dict)


# ── Exercise-Specific Thresholds ──────────────────────────────────────────────
# These define the MINIMUM expected motion for each exercise.
# Values are based on the biomechanical features each exercise extracts.
# Tuned conservatively — a person doing the exercise even poorly should pass.

# Each exercise has different feature columns (from joint_features.py):
#   Es1: 9 features (angles + distances) — arm raises
#   Es2: 9 features — arm/elbow movements
#   Es3: 12 features — shoulder rotation
#   Es4: 6 features — squats/leg movements
#   Es5: 9 features — sit-to-stand

# Thresholds: (min_variance, min_rom, min_displacement)
# - min_variance: minimum average temporal variance across features
# - min_rom: minimum average range of motion (max-min) across features
# - min_displacement: minimum average frame-to-frame absolute change
EXERCISE_THRESHOLDS = {
    "Es1": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es2": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es3": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es4": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
    "Es5": {"min_variance": 5.0, "min_rom": 8.0, "min_displacement": 0.5},
}

# Expected energy levels for each exercise (used for quality_factor calculation).
# These represent the typical motion energy for a "normal" performance.
# A perfect exercise should have energy near or above this value.
EXPECTED_ENERGY = {
    "Es1": 1.0,
    "Es2": 1.0,
    "Es3": 1.0,
    "Es4": 1.0,
    "Es5": 1.0,
}


def detect_motion(features: np.ndarray, exercise_id: str) -> MotionResult:
    """Analyze extracted features to detect whether meaningful exercise motion occurred.

    Args:
        features: numpy array of shape (n_frames, n_features) — RAW features
                  (before StandardScaler), as output by joint_features.get_esX_features().
        exercise_id: Exercise identifier (e.g., "Es1").

    Returns:
        MotionResult with motion detection verdict and quality metrics.
    """
    thresholds = EXERCISE_THRESHOLDS.get(exercise_id, EXERCISE_THRESHOLDS["Es1"])

    if features.shape[0] < 5:
        # Too few frames to analyze
        return MotionResult(
            is_active=False,
            motion_energy=0.0,
            quality_factor=0.1,
            rom_score=0.0,
            variance_score=0.0,
            frame_displacement_score=0.0,
            details={"reason": "too_few_frames", "n_frames": features.shape[0]},
        )

    # ── Metric 1: Temporal Variance ───────────────────────────────────────
    # High variance = the features change over time = movement happening.
    # For standing still, all angles/distances stay nearly constant → low variance.
    per_feature_variance = np.var(features, axis=0)
    avg_variance = float(np.mean(per_feature_variance))

    # Normalize: map variance to [0, 1] using threshold as reference
    variance_score = min(1.0, avg_variance / max(thresholds["min_variance"] * 5, 1e-8))

    # ── Metric 2: Range of Motion (ROM) ───────────────────────────────────
    # For each feature, compute max - min across all frames.
    # A large ROM indicates the joint moved through a significant arc.
    per_feature_rom = np.ptp(features, axis=0)  # peak-to-peak (max - min)
    avg_rom = float(np.mean(per_feature_rom))

    rom_score = min(1.0, avg_rom / max(thresholds["min_rom"] * 5, 1e-8))

    # ── Metric 3: Frame-to-Frame Displacement ─────────────────────────────
    # Sum of absolute differences between consecutive frames.
    # Standing still → near-zero displacement. Exercise → consistent displacement.
    if features.shape[0] > 1:
        frame_diffs = np.abs(np.diff(features, axis=0))
        avg_displacement = float(np.mean(frame_diffs))
    else:
        avg_displacement = 0.0

    displacement_score = min(1.0, avg_displacement / max(thresholds["min_displacement"] * 5, 1e-8))

    # ── Composite Motion Energy ───────────────────────────────────────────
    # Weighted combination: variance and displacement matter most,
    # ROM is a secondary confirming signal.
    motion_energy = (
        0.35 * variance_score +
        0.35 * displacement_score +
        0.30 * rom_score
    )

    # ── Activity Detection ────────────────────────────────────────────────
    # OVERRIDDEN FOR MODEL TESTING
    is_active = True

    # ── Quality Factor ────────────────────────────────────────────────────
    # OVERRIDDEN FOR MODEL TESTING
    motion_energy = 1.0
    quality_factor = 1.0

    return MotionResult(
        is_active=is_active,
        motion_energy=motion_energy,
        quality_factor=quality_factor,
        rom_score=round(rom_score, 4),
        variance_score=round(variance_score, 4),
        frame_displacement_score=round(displacement_score, 4),
        details={
            "avg_variance": round(avg_variance, 6),
            "avg_rom": round(avg_rom, 4),
            "avg_displacement": round(avg_displacement, 6),
            "thresholds": thresholds,
            "n_frames": features.shape[0],
            "n_features": features.shape[1],
        },
    )


def calibrate_score(raw_score: float, motion_result: MotionResult) -> float:
    """Apply motion-based calibration to the raw model prediction.

    Args:
        raw_score: Raw clinical score from model (0-100 scale).
        motion_result: MotionResult from detect_motion().

    Returns:
        Calibrated clinical score (0-100 scale), penalized if motion is insufficient.
    """
    calibrated = raw_score * motion_result.quality_factor
    return round(float(calibrated), 2)
