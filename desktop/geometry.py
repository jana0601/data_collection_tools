from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    ab_norm = np.linalg.norm(ab)
    cb_norm = np.linalg.norm(cb)
    if ab_norm == 0 or cb_norm == 0:
        return 0.0
    cos = float(np.clip(np.dot(ab, cb) / (ab_norm * cb_norm), -1.0, 1.0))
    return abs(np.degrees(np.arccos(cos)))


def hip_center(landmarks: np.ndarray) -> np.ndarray:
    # MediaPipe indices: left hip 23, right hip 24
    return (landmarks[23, :2] + landmarks[24, :2]) / 2.0


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate to hip center and scale by shoulder-hip distance.
    Expects landmarks in normalized [0,1] coords (x,y,z). Returns Nx3 normalized array.
    """
    lm = landmarks.copy()
    center = hip_center(lm)
    lm[:, 0] -= center[0]
    lm[:, 1] -= center[1]
    # scale by average of shoulder-to-hip distances (left side as reference)
    left_shoulder = lm[11, :2]
    left_hip = lm[23, :2]
    scale = float(np.linalg.norm(left_shoulder - left_hip))
    if scale <= 1e-6:
        scale = 1.0
    lm[:, 0] /= scale
    lm[:, 1] /= scale
    return lm


def shoulder_line_tilt_deg(landmarks: np.ndarray) -> float:
    left_shoulder = landmarks[11, :2]
    right_shoulder = landmarks[12, :2]
    dy = right_shoulder[1] - left_shoulder[1]
    dx = right_shoulder[0] - left_shoulder[0]
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    return float(np.degrees(np.arctan2(dy, dx)))


