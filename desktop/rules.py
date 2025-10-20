from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .geometry import compute_angle_deg, normalize_landmarks, shoulder_line_tilt_deg


@dataclass
class Feedback:
    type: str  # 'alignment' | 'balance' | 'range'
    message: str
    severity: str  # 'info' | 'warn' | 'error'


def tadasana_feedback(landmarks_norm: np.ndarray) -> List[Feedback]:
    """Simple rules for Tadasana (Mountain Pose).
    Expectations:
      - Knees straight (~175°+)
      - Hips stacked over ankles (approx vertical shins)
      - Shoulders level (tilt near 0°)
      - Arms relaxed by side (skip for v1)
    """
    cues: List[Feedback] = []

    # Indices
    L_ANKLE, R_ANKLE = 27 + 2, 28 + 2  # MP indices 27, 28 are knees; 31, 32 are ankles. We'll compute knee angles at (hip-knee-ankle)
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14

    # Knee angles
    left_knee = compute_angle_deg(landmarks_norm[L_HIP, :2], landmarks_norm[L_KNEE, :2], landmarks_norm[L_ANKLE, :2])
    right_knee = compute_angle_deg(landmarks_norm[R_HIP, :2], landmarks_norm[R_KNEE, :2], landmarks_norm[R_ANKLE, :2])
    if left_knee < 170:
        cues.append(Feedback('range', 'Straighten left knee', 'warn'))
    if right_knee < 170:
        cues.append(Feedback('range', 'Straighten right knee', 'warn'))

    # Shoulder level
    tilt = shoulder_line_tilt_deg(landmarks_norm)
    if abs(tilt) > 5:
        cues.append(Feedback('alignment', 'Level your shoulders', 'info'))

    # Simple stance width check: ankles roughly under hips in x after normalization
    hip_center_x = (landmarks_norm[L_HIP, 0] + landmarks_norm[R_HIP, 0]) / 2.0
    left_ankle_dx = abs(landmarks_norm[L_ANKLE, 0] - hip_center_x)
    right_ankle_dx = abs(landmarks_norm[R_ANKLE, 0] - hip_center_x)
    if left_ankle_dx > 0.4 or right_ankle_dx > 0.4:
        cues.append(Feedback('alignment', 'Bring feet under hips', 'info'))

    return cues


