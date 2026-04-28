"""
Feedback Engine for RehabAI

Generates human-readable rehabilitation feedback by comparing
patient's joint angles against reference exercise poses.
Uses biomechanical angle analysis instead of raw DTW distance.
"""

import math
import numpy as np

# MoveNet keypoint order (matches ml_wrapper.py and frontend KEYPOINT_DICT)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def _angle_at_vertex(p1, p2, p3):
    """Calculate angle (degrees) at vertex p2 formed by rays p2->p1 and p2->p3.
    Each point is [y, x] matching MoveNet output order."""
    v1y = float(p1[0]) - float(p2[0])
    v1x = float(p1[1]) - float(p2[1])
    v2y = float(p3[0]) - float(p2[0])
    v2x = float(p3[1]) - float(p2[1])

    norm1 = math.hypot(v1y, v1x)
    norm2 = math.hypot(v2y, v2x)
    norm_product = norm1 * norm2
    if norm_product < 1e-8:
        return 0.0

    dot = (v1y * v2y) + (v1x * v2x)
    cos_angle = max(-1.0, min(1.0, dot / norm_product))
    return float(math.degrees(math.acos(cos_angle)))


# ---------------------------------------------------------------------------
# Exercise-specific angle rules
# Each rule: (first, vertex, end) joints, tolerance in degrees, feedback text
# ---------------------------------------------------------------------------

_COMMON_RULES = [
    {
        "name": "left_shoulder_flexion",
        "joints": ("left_elbow", "left_shoulder", "left_hip"),
        "tolerance": 15,
        "label": "Vai trái",
        "feedback_low": "Hãy nâng tay trái lên cao hơn",
        "feedback_high": "Hãy hạ tay trái xuống một chút",
        "feedback_good": "Tay trái đang ở vị trí tốt ✓",
    },
    {
        "name": "right_shoulder_flexion",
        "joints": ("right_elbow", "right_shoulder", "right_hip"),
        "tolerance": 15,
        "label": "Vai phải",
        "feedback_low": "Hãy nâng tay phải lên cao hơn",
        "feedback_high": "Hãy hạ tay phải xuống một chút",
        "feedback_good": "Tay phải đang ở vị trí tốt ✓",
    },
    {
        "name": "left_elbow_extension",
        "joints": ("left_shoulder", "left_elbow", "left_wrist"),
        "tolerance": 20,
        "label": "Khuỷu tay trái",
        "feedback_low": "Hãy duỗi thẳng khuỷu tay trái hơn",
        "feedback_high": "Hãy co khuỷu tay trái lại một chút",
        "feedback_good": "Khuỷu tay trái ổn ✓",
    },
    {
        "name": "right_elbow_extension",
        "joints": ("right_shoulder", "right_elbow", "right_wrist"),
        "tolerance": 20,
        "label": "Khuỷu tay phải",
        "feedback_low": "Hãy duỗi thẳng khuỷu tay phải hơn",
        "feedback_high": "Hãy co khuỷu tay phải lại một chút",
        "feedback_good": "Khuỷu tay phải ổn ✓",
    },
    {
        "name": "left_knee_extension",
        "joints": ("left_hip", "left_knee", "left_ankle"),
        "tolerance": 15,
        "label": "Đầu gối trái",
        "feedback_low": "Hãy duỗi thẳng đầu gối trái hơn",
        "feedback_high": "Hãy co đầu gối trái lại một chút",
        "feedback_good": "Đầu gối trái ổn ✓",
    },
    {
        "name": "right_knee_extension",
        "joints": ("right_hip", "right_knee", "right_ankle"),
        "tolerance": 15,
        "label": "Đầu gối phải",
        "feedback_low": "Hãy duỗi thẳng đầu gối phải hơn",
        "feedback_high": "Hãy co đầu gối phải lại một chút",
        "feedback_good": "Đầu gối phải ổn ✓",
    },
]

# Es1 focuses on arm raises (from joint_features.py analysis)
_ES1_RULES = _COMMON_RULES  # all body angles relevant

# Es2-Es5 use the same comprehensive rule set for now;
# add exercise-specific overrides here when clinical definitions are available.
EXERCISE_RULES = {
    "Es1": _ES1_RULES,
    "Es2": _COMMON_RULES,
    "Es3": _COMMON_RULES,
    "Es4": _COMMON_RULES,
    "Es5": _COMMON_RULES,
}


def _compile_rules(rules):
    compiled = []
    for rule in rules:
        j1, j2, j3 = rule["joints"]
        compiled.append(
            {
                "indices": (KEYPOINT_INDEX[j1], KEYPOINT_INDEX[j2], KEYPOINT_INDEX[j3]),
                "tolerance": float(rule["tolerance"]),
                "label": rule["label"],
                "feedback_low": rule["feedback_low"],
                "feedback_high": rule["feedback_high"],
                "feedback_good": rule["feedback_good"],
            }
        )
    return compiled


COMPILED_EXERCISE_RULES = {
    exercise_id: _compile_rules(rules)
    for exercise_id, rules in EXERCISE_RULES.items()
}
COMPILED_COMMON_RULES = _compile_rules(_COMMON_RULES)


def parse_keypoints(flat_values):
    """Convert flat list (51 floats: 17 keypoints × [y, x, confidence]) to 17×3 array."""
    arr = np.asarray(flat_values, dtype=np.float32)
    if arr.size != 51:
        raise ValueError(f"Expected 51 values (17 keypoints × 3), got {arr.size}")
    return arr.reshape(17, 3)


def generate_feedback(exercise_id, reference_values, current_values):
    """Compare current pose against reference and return structured feedback.

    Args:
        exercise_id: Exercise identifier (e.g. "Es1")
        reference_values: Flat list of 51 floats for reference pose
        current_values: Flat list of 51 floats for current pose

    Returns:
        dict with primary_instruction, overall_accuracy, encouragement,
        and per-joint feedback_details.
    """
    ref_kps = parse_keypoints(reference_values)
    cur_kps = parse_keypoints(current_values)
    rules = COMPILED_EXERCISE_RULES.get(exercise_id, COMPILED_COMMON_RULES)

    feedback_items = []
    for rule in rules:
        i1, i2, i3 = rule["indices"]

        ref_angle = _angle_at_vertex(
            ref_kps[i1, :2],
            ref_kps[i2, :2],
            ref_kps[i3, :2],
        )
        cur_angle = _angle_at_vertex(
            cur_kps[i1, :2],
            cur_kps[i2, :2],
            cur_kps[i3, :2],
        )

        diff = cur_angle - ref_angle
        abs_diff = abs(diff)

        if abs_diff <= rule["tolerance"]:
            status = "good"
            message = rule["feedback_good"]
            severity = 0.0
        elif diff < 0:
            status = "needs_correction"
            message = rule["feedback_low"]
            severity = min(abs_diff / 45.0, 1.0)
        else:
            status = "needs_correction"
            message = rule["feedback_high"]
            severity = min(abs_diff / 45.0, 1.0)

        feedback_items.append({
            "joint_group": rule["label"],
            "status": status,
            "message": message,
            "angle_difference": round(float(diff), 1),
            "severity": round(float(severity), 2),
        })

    # Aggregate
    good_count = sum(1 for f in feedback_items if f["status"] == "good")
    total = len(feedback_items)
    overall_accuracy = good_count / total if total > 0 else 0.0

    # Priority: most severe corrections first
    corrections = sorted(
        [f for f in feedback_items if f["status"] == "needs_correction"],
        key=lambda x: x["severity"],
        reverse=True,
    )

    if corrections:
        primary_instruction = corrections[0]["message"]
    else:
        primary_instruction = "Tuyệt vời! Bạn đang thực hiện rất tốt! 👏"

    if overall_accuracy >= 0.8:
        encouragement = "Rất tốt! Hãy tiếp tục giữ phong độ này! 💪"
    elif overall_accuracy >= 0.5:
        encouragement = "Khá tốt! Hãy chú ý điều chỉnh thêm một chút."
    else:
        encouragement = "Hãy cố gắng điều chỉnh theo hướng dẫn nhé!"

    return {
        "primary_instruction": primary_instruction,
        "overall_accuracy": round(overall_accuracy, 2),
        "encouragement": encouragement,
        "corrections_needed": len(corrections),
        "feedback_details": feedback_items,
    }
