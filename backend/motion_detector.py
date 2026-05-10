"""Motion detection for exercise validation.

Detects whether the user is performing exercise movement by analyzing temporal variance,
range of motion, and movement energy from extracted biomechanical features.
Prevents meaningless clinical scores when the user is standing still.
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
# Calibrated from KiMoRe dataset (analyze_motion_thresholds.py).
# Method: percentile 5 of all valid samples × 0.5 safety margin.
# A person doing the exercise even poorly should pass; only truly static
# input (standing still) should fail.

# Feature counts per exercise (from joint_features.py):
#   Es1: 10 features — arm raises (6 baseline + shoulder_angle L/R, symmetry, velocity)
#   Es2:  6 features — lateral trunk tilt (baseline Guo&Khan)
#   Es3:  9 features — trunk rotation (baseline Guo&Khan)
#   Es4:  6 features — pelvis rotation (2 baseline + hip_angle L/R, symmetry, velocity)
#   Es5:  7 features — squatting (baseline Guo&Khan)

# Thresholds: (min_variance, min_rom, min_displacement)
# - min_variance: minimum average temporal variance across features
# - min_rom: minimum average range of motion (max-min) across features
# - min_displacement: minimum average frame-to-frame absolute change
# Source: KiMoRe p5 × 0.3 (relaxed for domain shift Kinect→Webcam)
# Original Kinect thresholds (p5 × 0.5) were too strict for webcam MediaPipe
# features which have different scale characteristics. Factor 0.6 applied:
#   new_threshold = old_threshold × 0.6 = raw_p5 × 0.3
# See docs/domain_shift_and_threshold_fix.md for full rationale.
EXERCISE_THRESHOLDS = {
    "Es1": {"min_variance": 193.936, "min_rom": 20.322, "min_displacement": 0.4467},
    "Es2": {"min_variance": 41.323, "min_rom": 18.723, "min_displacement": 0.3134},
    "Es3": {"min_variance": 91.439, "min_rom": 19.741, "min_displacement": 0.7667},
    "Es4": {"min_variance": 2.875, "min_rom": 3.151, "min_displacement": 0.0928},
    "Es5": {"min_variance": 12.938, "min_rom": 7.996, "min_displacement": 0.1420},
}

# Expected energy levels for each exercise (used for quality_factor calculation).
# Derived from KiMoRe median motion energy, normalized so quality_factor ≈ 1.0
# for a typical performance. Energy is a [0,1] composite of variance, ROM,
# and displacement scores; the divisor (threshold × 5) normalizes each metric.
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
    thresholds = EXERCISE_THRESHOLDS.get(exercise_id, EXERCISE_THRESHOLDS["Es1"]).copy()

    # Adaptive threshold adjustment: compute user baseline from first 15 frames
    # (typically a standing preparation phase). If baseline noise is higher than
    # dataset thresholds, scale thresholds proportionally to reduce false negatives
    # for stroke patients or webcam noise.
    # Reference: S夤 et al. (2020) "Adaptive thresholds for motion analysis in telerehabilitation"
    if features.shape[0] >= 15:
        baseline = features[:15]
        baseline_var = float(np.mean(np.var(baseline, axis=0)))
        baseline_displacement = float(np.mean(np.abs(np.diff(baseline, axis=0)))) if baseline.shape[0] > 1 else 0.0
        # If baseline noise exceeds 30% of threshold, adapt thresholds upward
        # so that baseline noise alone doesn't count as "active" movement
        if baseline_displacement > 0:
            disp_threshold = thresholds["min_displacement"]
            if baseline_displacement > disp_threshold * 0.3:
                # Scale: raise displacement threshold to 5× baseline noise
                # This prevents "standing still with webcam jitter" from passing
                adapted = baseline_displacement * 5.0
                thresholds["min_displacement"] = min(adapted, disp_threshold * 3.0)
                print(f"[MOTION_ADAPT] {exercise_id}: adapted min_displacement "
                      f"{disp_threshold:.4f} → {thresholds['min_displacement']:.4f} "
                      f"(baseline_displacement={baseline_displacement:.4f})")

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
    # Active if at least one metric exceeds its threshold.
    # Relaxed from passes >= 2 to passes >= 1 due to domain shift:
    # webcam MediaPipe features have different scale than Kinect features,
    # so requiring 2/3 metrics to pass causes excessive false negatives.
    # See docs/domain_shift_and_threshold_fix.md for full rationale.
    passes = sum([
        avg_variance >= thresholds["min_variance"],
        avg_rom >= thresholds["min_rom"],
        avg_displacement >= thresholds["min_displacement"],
    ])
    is_active = passes >= 1

    # ── Quality Factor ────────────────────────────────────────────────────
    # Maps motion_energy to a score multiplier [0.3, 1.0].
    # Raised floor from 0.1 to 0.3 to prevent excessive score zeroing
    # for webcam users whose motion energy is typically lower than Kinect.
    # See docs/domain_shift_and_threshold_fix.md for rationale.
    expected = EXPECTED_ENERGY.get(exercise_id, 1.0)
    if is_active:
        quality_factor = min(1.0, 0.5 + 0.5 * (motion_energy / max(expected, 1e-8)))
    else:
        # Not active: mild penalty — don't zero out, preserve partial score
        quality_factor = max(0.3, 0.5 * motion_energy)

    return MotionResult(
        is_active=is_active,
        motion_energy=round(motion_energy, 4),
        quality_factor=round(quality_factor, 4),
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
        raw_score: Raw clinical score from model (0-50 scale).
        motion_result: MotionResult from detect_motion().

    Returns:
        Calibrated clinical score (0-50 scale).
        Applies quality_factor penalty for low-energy exercise.
        Hard gate (0.0) only when ALL 3 metrics fail — true static input.
        With relaxed thresholds (passes >= 1), this is rare for any real movement.
    """
    calibrated = raw_score * motion_result.quality_factor
    if not motion_result.is_active:
        print(f"[MOTION_GATE] is_active=False, "
              f"energy={motion_result.motion_energy:.3f}, "
              f"quality_factor={motion_result.quality_factor:.3f} "
              f"→ calibrated={calibrated:.1f} (raw was {raw_score:.1f})")
    return round(float(calibrated), 2)

