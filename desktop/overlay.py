from typing import Optional

import cv2
import numpy as np


# MediaPipe pose indices reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
    (23, 24),            # hips
    (11, 23), (12, 24),  # torso
    (23, 25), (25, 27), (27, 29), (29, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # right leg
]


def draw_pose_overlay(rgb_image: np.ndarray, landmarks_norm: np.ndarray) -> np.ndarray:
    h, w, _ = rgb_image.shape
    pts = np.empty((landmarks_norm.shape[0], 2), dtype=np.int32)
    pts[:, 0] = (landmarks_norm[:, 0] * w).astype(np.int32)
    pts[:, 1] = (landmarks_norm[:, 1] * h).astype(np.int32)

    # Draw connections
    for a, b in POSE_CONNECTIONS:
        pa = tuple(pts[a])
        pb = tuple(pts[b])
        cv2.line(rgb_image, pa, pb, (0, 255, 0), 2)

    # Draw joints
    for x, y in pts:
        cv2.circle(rgb_image, (int(x), int(y)), 3, (255, 0, 0), -1)

    return rgb_image


