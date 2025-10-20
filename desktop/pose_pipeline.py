from dataclasses import dataclass
from typing import Optional, List

import mediapipe as mp
import numpy as np


@dataclass
class PoseResult:
    landmarks: Optional[np.ndarray]  # shape: (33, 3) in normalized coords
    visibility: Optional[np.ndarray]  # shape: (33,)


class PosePipeline:
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1, enable_segmentation: bool = False) -> None:
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame: np.ndarray) -> PoseResult:
        # MediaPipe expects RGB uint8
        results = self._pose.process(rgb_frame)
        if results.pose_landmarks is None:
            return PoseResult(landmarks=None, visibility=None)

        # Convert to Nx3 numpy (x,y,z) normalized in [0,1]
        lms = results.pose_landmarks.landmark
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        vis = np.array([lm.visibility for lm in lms], dtype=np.float32)
        return PoseResult(landmarks=arr, visibility=vis)


