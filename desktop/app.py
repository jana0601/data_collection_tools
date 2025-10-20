import sys
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QMessageBox, QHBoxLayout, QLineEdit, QPushButton, QCheckBox
from pathlib import Path

from .pose_pipeline import PosePipeline
from .overlay import draw_pose_overlay
from .geometry import normalize_landmarks
from .rules import tadasana_feedback, Feedback
from .collector import StaticGestureCollector, CollectorConfig


class VideoWindow(QMainWindow):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self.setWindowTitle("YogaPose - Desktop")
        self.resize(960, 720)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.feedback_label = QLabel()
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("QLabel { color: #1f2937; font-size: 14px; }")

        central = QWidget()
        layout = QVBoxLayout(central)

        controls_row = QHBoxLayout()
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("Subject ID (e.g., user001)")
        self.subject_input.setText("default")
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Gesture ID (e.g., tadasana)")
        self.start_btn = QPushButton("Start Collecting")
        self.stop_btn = QPushButton("Stop")
        self.save_images_cb = QCheckBox("Save images")
        self.video_mode_cb = QCheckBox("Video mode (30 frames)")
        self.open_folder_btn = QPushButton("Open Folder")
        controls_row.addWidget(self.subject_input)
        controls_row.addWidget(self.label_input)
        controls_row.addWidget(self.start_btn)
        controls_row.addWidget(self.stop_btn)
        controls_row.addWidget(self.save_images_cb)
        controls_row.addWidget(self.video_mode_cb)
        controls_row.addWidget(self.open_folder_btn)

        layout.addLayout(controls_row)
        layout.addWidget(self.video_label)
        layout.addWidget(self.feedback_label)

        # Saved info row
        saved_row = QHBoxLayout()
        self.saved_count_label = QLabel("Saved: 0")
        self.mode_status_label = QLabel("Mode: Single Image")
        self.frame_count_label = QLabel("Frames: 0")
        self.last_saved_thumb = QLabel()
        self.last_saved_thumb.setFixedSize(160, 90)
        self.last_saved_thumb.setStyleSheet("QLabel { background: #e5e7eb; border: 1px solid #cbd5e1; }")
        self.last_saved_thumb.setAlignment(Qt.AlignCenter)
        saved_row.addWidget(self.saved_count_label)
        saved_row.addWidget(self.mode_status_label)
        saved_row.addWidget(self.frame_count_label)
        saved_row.addWidget(self.last_saved_thumb)
        layout.addLayout(saved_row)
        self.setCentralWidget(central)

        self.cap: Optional[cv2.VideoCapture] = None
        self.pipeline = PosePipeline(static_image_mode=False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)

        self._open_camera(camera_index)
        self._collector: StaticGestureCollector | None = None
        self.start_btn.clicked.connect(self._on_start_collect)
        self.stop_btn.clicked.connect(self._on_stop_collect)
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        self._saved_count_ui = 0

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            super().closeEvent(event)

    def _open_camera(self, index: int) -> None:
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Failed to open webcam.")
            return
        # Set a reasonable resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.timer.start(0)  # run as fast as possible; Qt throttles to display refresh

    def _on_frame(self) -> None:
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok:
            return

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pipeline.process(rgb)

        # Draw overlay on a copy
        annotated = rgb.copy()
        if result.landmarks is not None:
            annotated = draw_pose_overlay(annotated, result.landmarks)
            norm = normalize_landmarks(result.landmarks)
            cues = tadasana_feedback(norm)
            if cues:
                text = " | ".join([c.message for c in cues])
            else:
                text = "Good alignment"
            self.feedback_label.setText(text)

            # collector
            if self._collector is not None:
                h, w, _ = rgb.shape
                ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                # For video mode, always call collector regardless of pose detection
                if self._collector.config.video_mode:
                    saved_path = self._collector.consider(
                        label=self.label_input.text().strip() or None,
                        landmarks_norm=norm,
                        visibility=result.visibility,
                        image_rgb=rgb,  # Always pass image for video mode
                        image_size=(w, h),
                        ts_ms=ts_ms,
                        camera_index=0,
                    )
                else:
                    # For single image mode, only call when pose is detected
                    saved_path = self._collector.consider(
                        label=self.label_input.text().strip() or None,
                        landmarks_norm=norm,
                        visibility=result.visibility,
                        image_rgb=rgb if self.save_images_cb.isChecked() else None,
                        image_size=(w, h),
                        ts_ms=ts_ms,
                        camera_index=0,
                    )
                
                # Update frame counter display
                session_frames = self._collector.current_session_frames
                total_frames = self._collector.total_frames_processed
                if self._collector.config.video_mode:
                    video_frame_count = self._collector.video_frame_count
                    self.frame_count_label.setText(f"Frames: {video_frame_count}/30")
                else:
                    self.frame_count_label.setText(f"Frames: {session_frames}")
                
                if saved_path is not None:
                    self._saved_count_ui += 1
                    self.saved_count_label.setText(f"Saved: {self._saved_count_ui}")
                    
                    # Reset frame count display if video was saved
                    if self._collector.config.video_mode:
                        self.frame_count_label.setText("Frames: 0/30")
                    
                    if self.save_images_cb.isChecked():
                        # Show the last RGB image as thumbnail
                        thumb = self._to_qimage(rgb)
                        pix = QPixmap.fromImage(thumb).scaled(self.last_saved_thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.last_saved_thumb.setPixmap(pix)
        else:
            self.feedback_label.setText("No person detected")
            # Still try to save images even without pose detection
            if self._collector is not None:
                h, w, _ = rgb.shape
                ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                # For video mode, always call collector regardless of pose detection
                if self._collector.config.video_mode:
                    saved_path = self._collector.consider(
                        label=self.label_input.text().strip() or None,
                        landmarks_norm=None,
                        visibility=None,
                        image_rgb=rgb,  # Always pass image for video mode
                        image_size=(w, h),
                        ts_ms=ts_ms,
                        camera_index=0,
                    )
                else:
                    # For single image mode, only call when pose is detected
                    saved_path = self._collector.consider(
                        label=self.label_input.text().strip() or None,
                        landmarks_norm=None,
                        visibility=None,
                        image_rgb=rgb if self.save_images_cb.isChecked() else None,
                        image_size=(w, h),
                        ts_ms=ts_ms,
                        camera_index=0,
                    )
                
                # Update frame counter display
                session_frames = self._collector.current_session_frames
                if self._collector.config.video_mode:
                    video_frame_count = self._collector.video_frame_count
                    self.frame_count_label.setText(f"Frames: {video_frame_count}/30")
                else:
                    self.frame_count_label.setText(f"Frames: {session_frames}")
                
                if saved_path is not None:
                    self._saved_count_ui += 1
                    self.saved_count_label.setText(f"Saved: {self._saved_count_ui}")
                    
                    # Reset frame count display if video was saved
                    if self._collector.config.video_mode:
                        self.frame_count_label.setText("Frames: 0/30")
                    
                    if self.save_images_cb.isChecked():
                        # Show the last RGB image as thumbnail
                        thumb = self._to_qimage(rgb)
                        pix = QPixmap.fromImage(thumb).scaled(self.last_saved_thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.last_saved_thumb.setPixmap(pix)

        # Show camera frame on UI
        qimg = self._to_qimage(annotated)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def _to_qimage(self, rgb: np.ndarray) -> QImage:
        h, w, _ = rgb.shape
        bytes_per_line = 3 * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    def _on_start_collect(self) -> None:
        out_dir = Path("data/static_gestures")
        subject_id = self.subject_input.text().strip() or "default"
        video_mode = self.video_mode_cb.isChecked()
        cfg = CollectorConfig(out_dir=out_dir, save_images=self.save_images_cb.isChecked(), 
                            subject_id=subject_id, video_mode=video_mode)
        self._collector = StaticGestureCollector(cfg)
        self._collector.reset_session()  # Reset frame counters
        self._saved_count_ui = 0
        self.saved_count_label.setText("Saved: 0")
        self.frame_count_label.setText("Frames: 0")
        
        # Update mode status
        if video_mode:
            self.mode_status_label.setText("Mode: Video (30 frames)")
        else:
            self.mode_status_label.setText("Mode: Single Image")

    def _on_stop_collect(self) -> None:
        # Stop data collection cleanly
        if self._collector is not None:
            self._collector.finalize_session()
        self._collector = None

    def _on_open_folder(self) -> None:
        folder = Path("data/static_gestures").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def main() -> None:
    app = QApplication(sys.argv)
    win = VideoWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


