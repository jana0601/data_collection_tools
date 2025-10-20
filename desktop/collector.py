from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CollectorConfig:
    out_dir: Path
    save_images: bool = False
    min_visibility: float = 0.6
    sampling_hz: float = 3.0
    stability_window_frames: int = 10
    trim_start_count: int = 3  # Delete first N images
    trim_end_count: int = 2    # Delete last N images
    subject_id: str = "default"  # Subject identifier
    video_mode: bool = False  # True for video (30 frames), False for single images
    video_frames: int = 30  # Number of frames for video mode


class StaticGestureCollector:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self._ensure_dirs()
        self._label: Optional[str] = None
        self._last_save_ts_ms: int = 0
        self._saved_count: int = 0
        self._recent_landmarks: list[np.ndarray] = []
        # Session tracking for trimming
        self._session_images: list[Path] = []
        self._session_label: Optional[str] = None
        # Frame counting
        self._total_frames_processed: int = 0
        self._current_session_frames: int = 0
        # Video mode frame tracking
        self._video_frame_count: int = 0
        self._video_frames_buffer: list[np.ndarray] = []

    def _ensure_dirs(self) -> None:
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        (self.config.out_dir / "images").mkdir(parents=True, exist_ok=True)

    def _get_images_dir(self) -> Path:
        """Get the images directory, creating it if needed."""
        images_dir = self.config.out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        return images_dir

    def set_label(self, label: str) -> None:
        label = label.strip()
        self._label = label if label else None

    def _trim_session_images(self) -> None:
        """Delete first and last few images from the current session."""
        if len(self._session_images) <= self.config.trim_start_count + self.config.trim_end_count:
            return
        
        # Delete first N images
        for i in range(min(self.config.trim_start_count, len(self._session_images))):
            try:
                self._session_images[i].unlink(missing_ok=True)
            except Exception:
                pass
        
        # Delete last N images
        for i in range(max(0, len(self._session_images) - self.config.trim_end_count), len(self._session_images)):
            try:
                self._session_images[i].unlink(missing_ok=True)
            except Exception:
                pass
        
        # Clear session tracking
        self._session_images.clear()
        self._session_label = None

    @property
    def saved_count(self) -> int:
        return self._saved_count

    @property
    def total_frames_processed(self) -> int:
        return self._total_frames_processed

    @property
    def current_session_frames(self) -> int:
        return self._current_session_frames

    @property
    def video_frame_count(self) -> int:
        return self._video_frame_count

    def _quality_ok(self, landmarks_norm: np.ndarray, visibility: np.ndarray) -> bool:
        if landmarks_norm is None:
            return False
        if float(np.mean(visibility)) < self.config.min_visibility:
            return False
        # Stability: recent std dev on hips and shoulders
        focus_idx = [11, 12, 23, 24]
        self._recent_landmarks.append(landmarks_norm.copy())
        if len(self._recent_landmarks) > self.config.stability_window_frames:
            self._recent_landmarks.pop(0)
        if len(self._recent_landmarks) >= self.config.stability_window_frames:
            stack = np.stack(self._recent_landmarks, axis=0)  # (T, 33, 3)
            std_focus = np.std(stack[:, focus_idx, :2], axis=0).mean()
            if std_focus > 0.02:
                return False
        return True

    def consider(self, *,
                 label: Optional[str],
                 landmarks_norm: Optional[np.ndarray],
                 visibility: Optional[np.ndarray],
                 image_rgb: Optional[np.ndarray],
                 image_size: tuple[int, int],
                 ts_ms: int,
                 camera_index: int) -> Optional[Path]:
        if label is None:
            return None
        
        # Handle video mode vs single image mode
        if self.config.video_mode:
            return self._handle_video_mode(label, landmarks_norm, visibility, image_rgb, image_size, ts_ms, camera_index)
        else:
            return self._handle_single_image_mode(label, landmarks_norm, visibility, image_rgb, image_size, ts_ms, camera_index)

    def _handle_single_image_mode(self, label: str, landmarks_norm: Optional[np.ndarray], 
                                visibility: Optional[np.ndarray], image_rgb: Optional[np.ndarray],
                                image_size: tuple[int, int], ts_ms: int, camera_index: int) -> Optional[Path]:
        """Handle single image saving mode."""
        # Simple cadence check only for single image mode
        min_interval_ms = int(1000.0 / max(1e-6, self.config.sampling_hz))
        if ts_ms - self._last_save_ts_ms < min_interval_ms:
            return None

        # Increment frame counters only when we actually process a frame
        self._total_frames_processed += 1
        self._current_session_frames += 1

        sample_id = time.strftime("%Y%m%dT%H%M%S") + f"_{ts_ms%1000:03d}"
        record = {
            "id": sample_id,
            "subject_id": self.config.subject_id,
            "label": label,
            "landmarks": landmarks_norm.tolist() if landmarks_norm is not None else [],
            "visibility": visibility.tolist() if visibility is not None else [],
            "image_size": [int(image_size[0]), int(image_size[1])],
            "timestamp_ms": int(ts_ms),
            "camera_index": int(camera_index),
            "note": ""
        }
        records_path = self.config.out_dir / "records.jsonl"
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Save single image if enabled
        if self.config.save_images and image_rgb is not None:
            print(f"Saving single image for label: {label}")
            try:
                import cv2
                clean_gesture = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_gesture = clean_gesture.replace(' ', '_').lower()
                
                images_dir = self._get_images_dir()
                filename = f"{self.config.subject_id}_{clean_gesture}_{sample_id}.jpg"
                img_path = images_dir / filename
                
                bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(str(img_path), bgr)
                if success:
                    self._session_images.append(img_path)
                    print(f"Successfully saved image: {img_path}")
                else:
                    print(f"Failed to save image: {img_path}")
            except Exception as e:
                print(f"Error saving image: {e}")

        self._last_save_ts_ms = ts_ms
        self._saved_count += 1
        return records_path

    def _handle_video_mode(self, label: str, landmarks_norm: Optional[np.ndarray], 
                          visibility: Optional[np.ndarray], image_rgb: Optional[np.ndarray],
                          image_size: tuple[int, int], ts_ms: int, camera_index: int) -> Optional[Path]:
        """Handle video mode - count frames and save video every 30 frames."""
        if image_rgb is not None:
            # Increment frame count
            self._video_frame_count += 1
            self._video_frames_buffer.append(image_rgb.copy())
            
            print(f"Frame {self._video_frame_count}/30")
            
            # Check if we have 30 frames
            if self._video_frame_count >= self.config.video_frames:
                # Save video
                sample_id = time.strftime("%Y%m%dT%H%M%S") + f"_{ts_ms%1000:03d}"
                try:
                    import cv2
                    clean_gesture = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).strip()
                    clean_gesture = clean_gesture.replace(' ', '_').lower()
                    
                    images_dir = self._get_images_dir()
                    video_filename = f"{self.config.subject_id}_{clean_gesture}_{sample_id}.mp4"
                    video_path = images_dir / video_filename
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 10  # 10 fps for the video
                    h, w = self._video_frames_buffer[0].shape[:2]
                    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                    
                    # Write frames
                    for frame in self._video_frames_buffer:
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(bgr_frame)
                    
                    out.release()
                    print(f"Successfully saved video: {video_path}")
                    
                    # Save record
                    record = {
                        "id": sample_id,
                        "subject_id": self.config.subject_id,
                        "label": label,
                        "landmarks": landmarks_norm.tolist() if landmarks_norm is not None else [],
                        "visibility": visibility.tolist() if visibility is not None else [],
                        "image_size": [int(image_size[0]), int(image_size[1])],
                        "timestamp_ms": int(ts_ms),
                        "camera_index": int(camera_index),
                        "video_frames": self.config.video_frames,
                        "note": ""
                    }
                    records_path = self.config.out_dir / "records.jsonl"
                    with records_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    # Reset frame count and buffer for next sample
                    self._video_frame_count = 0
                    self._video_frames_buffer.clear()
                    self._current_session_frames = 0
                    self._last_save_ts_ms = ts_ms
                    self._saved_count += 1
                    return records_path
                    
                except Exception as e:
                    print(f"Error saving video: {e}")
                    self._video_frame_count = 0
                    self._video_frames_buffer.clear()
                    self._current_session_frames = 0
        
        return None

    def finalize_session(self) -> None:
        """Call this when collection stops to trim the final session."""
        if self._session_images:
            self._trim_session_images()
        # Reset session frame count
        self._current_session_frames = 0

    def reset_session(self) -> None:
        """Reset session counters when starting new collection."""
        self._current_session_frames = 0
        self._video_frame_count = 0
        self._video_frames_buffer.clear()


