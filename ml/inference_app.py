"""
Real-time Gesture Classification Inference Application.
Uses trained models to classify gestures from live camera feed.
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Dict, Any
import json
import joblib
from collections import deque
import time
# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QPushButton, QSlider, QTextEdit, 
                               QComboBox, QGroupBox, QGridLayout, QFrame)
from PySide6.QtCore import QTimer, Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QImage, QFont


class GestureInferenceEngine:
    """Core inference engine for gesture classification."""
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Try to find models relative to current script location
            script_dir = Path(__file__).parent
            models_dir = script_dir / "models_2class"
        self.models_dir = Path(models_dir)
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.class_names = []
        self.scaler = None
        self.target_frames = 29  # Use last 29 frames + current = 30 total
        self.frame_skip = 1  # Process every frame for better accuracy
        self.frame_counter = 0
        
        # Initialize MediaPipe with optimized settings for speed
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Reduced from 1 to 0 for speed
            enable_segmentation=False,
            smooth_landmarks=False,  # Disabled for speed
            min_detection_confidence=0.3,  # Reduced from 0.5
            min_tracking_confidence=0.3,  # Reduced from 0.5
        )
        
        # Frame buffer for video processing
        self.frame_buffer = deque(maxlen=self.target_frames)
        self.is_loaded = False
        
        # Performance optimization: cache last features
        self.last_features = None
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # Predict every 100ms max
        
    def load_models(self) -> bool:
        """Load trained models."""
        try:
            print(f"Looking for models in: {self.models_dir}")
            # Load metadata - try both multiclass and 2class versions
            metadata_path = self.models_dir / "gesture_multiclass_metadata.json"
            if not metadata_path.exists():
                metadata_path = self.models_dir / "gesture_2class_metadata.json"
                if not metadata_path.exists():
                    print(f"Metadata not found in: {self.models_dir}")
                    return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.class_names = metadata['class_names']
            self.best_model_name = metadata['best_model']
            
            # Load models - try multiclass first, then 2class
            model_files = {
                'random_forest': 'gesture_multiclass_random_forest.joblib',
                'svm': 'gesture_multiclass_svm.joblib',
                'neural_network': 'gesture_multiclass_neural_network.joblib'
            }
            
            # If multiclass models don't exist, try 2class models
            if not (self.models_dir / model_files['neural_network']).exists():
                model_files = {
                    'random_forest': 'gesture_2class_random_forest.joblib',
                    'svm': 'gesture_2class_svm.joblib',
                    'neural_network': 'gesture_2class_neural_network.joblib'
                }
            
            for name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                    print(f"Loaded {name} model")
            
            if self.best_model_name in self.models:
                self.best_model = self.models[self.best_model_name]
                self.is_loaded = True
                print(f"Best model: {self.best_model_name}")
                print(f"Classes: {self.class_names}")
                return True
            else:
                print(f"Best model {self.best_model_name} not found")
                return False
                
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def extract_features_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from a single frame."""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is not None:
            # Extract landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
            
            # Extract geometric features
            features = self._extract_geometric_features(landmarks, visibility)
            return features
        else:
            # No pose detected - return zero features
            return np.zeros(23)
    
    def _extract_geometric_features(self, landmarks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """Extract geometric features from pose landmarks."""
        features = []
        
        # MediaPipe pose landmark indices
        nose = 0
        left_shoulder, right_shoulder = 11, 12
        left_elbow, right_elbow = 13, 14
        left_wrist, right_wrist = 15, 16
        left_hip, right_hip = 23, 24
        left_knee, right_knee = 25, 26
        left_ankle, right_ankle = 27, 28
        
        # 1. Joint angles
        angles = []
        
        # Shoulder angles
        if visibility[left_shoulder] > 0.5 and visibility[left_elbow] > 0.5 and visibility[left_wrist] > 0.5:
            angles.extend(self._calculate_angles(landmarks[left_shoulder], landmarks[left_elbow], landmarks[left_wrist]))
        else:
            angles.extend([0, 0, 0])
            
        if visibility[right_shoulder] > 0.5 and visibility[right_elbow] > 0.5 and visibility[right_wrist] > 0.5:
            angles.extend(self._calculate_angles(landmarks[right_shoulder], landmarks[right_elbow], landmarks[right_wrist]))
        else:
            angles.extend([0, 0, 0])
        
        # Hip angles
        if visibility[left_hip] > 0.5 and visibility[left_knee] > 0.5 and visibility[left_ankle] > 0.5:
            angles.extend(self._calculate_angles(landmarks[left_hip], landmarks[left_knee], landmarks[left_ankle]))
        else:
            angles.extend([0, 0, 0])
            
        if visibility[right_hip] > 0.5 and visibility[right_knee] > 0.5 and visibility[right_ankle] > 0.5:
            angles.extend(self._calculate_angles(landmarks[right_hip], landmarks[right_knee], landmarks[right_ankle]))
        else:
            angles.extend([0, 0, 0])
        
        features.extend(angles)
        
        # 2. Body proportions and symmetry
        proportions = []
        
        # Shoulder width
        if visibility[left_shoulder] > 0.5 and visibility[right_shoulder] > 0.5:
            shoulder_width = np.linalg.norm(landmarks[left_shoulder] - landmarks[right_shoulder])
            proportions.append(shoulder_width)
        else:
            proportions.append(0)
        
        # Hip width
        if visibility[left_hip] > 0.5 and visibility[right_hip] > 0.5:
            hip_width = np.linalg.norm(landmarks[left_hip] - landmarks[right_hip])
            proportions.append(hip_width)
        else:
            proportions.append(0)
        
        # Torso length
        if visibility[left_shoulder] > 0.5 and visibility[left_hip] > 0.5:
            torso_length = np.linalg.norm(landmarks[left_shoulder] - landmarks[left_hip])
            proportions.append(torso_length)
        else:
            proportions.append(0)
        
        features.extend(proportions)
        
        # 3. Pose stability (center of mass)
        if np.sum(visibility) > 10:
            center_of_mass = np.average(landmarks, axis=0, weights=visibility)
            features.extend(center_of_mass)
        else:
            features.extend([0, 0, 0])
        
        # 4. Arm and leg extensions
        extensions = []
        
        # Left arm extension
        if visibility[left_shoulder] > 0.5 and visibility[left_wrist] > 0.5:
            left_arm_ext = np.linalg.norm(landmarks[left_shoulder] - landmarks[left_wrist])
            extensions.append(left_arm_ext)
        else:
            extensions.append(0)
        
        # Right arm extension
        if visibility[right_shoulder] > 0.5 and visibility[right_wrist] > 0.5:
            right_arm_ext = np.linalg.norm(landmarks[right_shoulder] - landmarks[right_wrist])
            extensions.append(right_arm_ext)
        else:
            extensions.append(0)
        
        # Left leg extension
        if visibility[left_hip] > 0.5 and visibility[left_ankle] > 0.5:
            left_leg_ext = np.linalg.norm(landmarks[left_hip] - landmarks[left_ankle])
            extensions.append(left_leg_ext)
        else:
            extensions.append(0)
        
        # Right leg extension
        if visibility[right_hip] > 0.5 and visibility[right_ankle] > 0.5:
            right_leg_ext = np.linalg.norm(landmarks[right_hip] - landmarks[right_ankle])
            extensions.append(right_leg_ext)
        else:
            extensions.append(0)
        
        features.extend(extensions)
        
        # 5. Pose orientation (body tilt)
        if visibility[left_shoulder] > 0.5 and visibility[right_shoulder] > 0.5 and visibility[left_hip] > 0.5 and visibility[right_hip] > 0.5:
            shoulder_center = (landmarks[left_shoulder] + landmarks[right_shoulder]) / 2
            hip_center = (landmarks[left_hip] + landmarks[right_hip]) / 2
            body_tilt = np.arctan2(shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
            features.append(body_tilt)
        else:
            features.append(0)
        
        return np.array(features)
    
    def _calculate_angles(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> list:
        """Calculate angles between three points."""
        # Vector from p2 to p1
        v1 = p1 - p2
        # Vector from p2 to p3
        v2 = p3 - p2
        
        # Calculate angle in radians
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return [angle, np.linalg.norm(v1), np.linalg.norm(v2)]
    
    def predict_from_frame(self, frame: np.ndarray, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict gesture using last 29 frames + current frame."""
        if not self.is_loaded:
            return {'error': 'Models not loaded'}
        
        # Extract features from current frame
        current_features = self.extract_features_from_frame(frame)
        
        # Add current frame to buffer (this will maintain last 29 frames + current = 30 total)
        self.frame_buffer.append(current_features)
        
        # Need at least 5 frames for prediction
        if len(self.frame_buffer) < 5:
            return {'error': 'Not enough frames', 'frames': len(self.frame_buffer)}
        
        # Use specified model or best model
        model = self.models.get(model_name, self.best_model)
        if model is None:
            return {'error': f'Model {model_name} not found'}
        
        # Prepare features: use last 29 frames + current frame (30 total)
        video_features = []
        frames_to_use = min(30, len(self.frame_buffer))  # Use up to 30 frames
        
        # Take the last frames from buffer
        recent_frames = list(self.frame_buffer)[-frames_to_use:]
        
        for frame_features in recent_frames:
            video_features.extend(frame_features)
        
        # Pad with zeros if we have fewer than 30 frames
        while len(video_features) < 30 * 23:  # 30 frames * 23 features per frame
            video_features.extend([0] * 23)
        
        video_features = np.array(video_features).reshape(1, -1)
        
        # Make prediction
        try:
            prediction = model.predict(video_features)[0]
            probabilities = model.predict_proba(video_features)[0]
            
            predicted_class = self.class_names[prediction]
            confidence = probabilities[prediction]
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_predictions = [
                {
                    'gesture': self.class_names[i],
                    'confidence': float(probabilities[i])
                }
                for i in top_indices
            ]
            
            return {
                'predicted_gesture': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'frames_processed': len(self.frame_buffer),
                'frames_used': frames_to_use
            }
            
        except Exception as e:
            return {'error': str(e)}


class CameraThread(QThread):
    """Thread for camera capture."""
    frame_ready = Signal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.running = False
        
    def start_camera(self, camera_index: int = 0):
        """Start camera capture."""
        self.camera = cv2.VideoCapture(camera_index)
        if self.camera.isOpened():
            self.running = True
            self.start()
        else:
            print(f"Failed to open camera {camera_index}")
    
    def stop_camera(self):
        """Stop camera capture."""
        self.running = False
        if self.camera:
            self.camera.release()
        self.quit()
        self.wait()
    
    def run(self):
        """Main camera loop."""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                break
            self.msleep(33)  # ~30 FPS


class GestureInferenceApp(QMainWindow):
    """Main application window for gesture inference."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI first
        self.init_ui()
        
        # Then initialize inference engine
        self.inference_engine = GestureInferenceEngine()
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.on_frame_received)
        
        self.current_frame = None
        self.prediction_history = deque(maxlen=10)
        
        # Load models after UI is ready
        QTimer.singleShot(100, self.load_models)  # Delay model loading slightly
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Real-time Gesture Classification")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - Camera and controls
        left_panel = QVBoxLayout()
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Feed")
        left_panel.addWidget(self.camera_label)
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Camera controls
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        controls_layout.addWidget(self.start_btn, 0, 0)
        
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn, 0, 1)
        
        # Model selection
        controls_layout.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["random_forest", "svm", "neural_network"])
        controls_layout.addWidget(self.model_combo, 1, 1)
        
        # Confidence threshold
        controls_layout.addWidget(QLabel("Confidence Threshold:"), 2, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        controls_layout.addWidget(self.confidence_slider, 2, 1)
        
        self.confidence_label = QLabel("0.50")
        controls_layout.addWidget(self.confidence_label, 2, 2)
        
        left_panel.addWidget(controls_group)
        
        # Right panel - Predictions and info
        right_panel = QVBoxLayout()
        
        # Current prediction
        prediction_group = QGroupBox("Current Prediction")
        prediction_layout = QVBoxLayout(prediction_group)
        
        self.prediction_label = QLabel("No prediction")
        self.prediction_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.prediction_label)
        
        self.confidence_label_pred = QLabel("Confidence: --")
        self.confidence_label_pred.setFont(QFont("Arial", 12))
        self.confidence_label_pred.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.confidence_label_pred)
        
        right_panel.addWidget(prediction_group)
        
        # Top predictions
        top_predictions_group = QGroupBox("Top Predictions")
        top_predictions_layout = QVBoxLayout(top_predictions_group)
        
        self.top_predictions_text = QTextEdit()
        self.top_predictions_text.setMaximumHeight(150)
        self.top_predictions_text.setReadOnly(True)
        top_predictions_layout.addWidget(self.top_predictions_text)
        
        right_panel.addWidget(top_predictions_group)
        
        # System info
        info_group = QGroupBox("System Info")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(200)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        right_panel.addWidget(info_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # Update every 100ms
        
    def load_models(self):
        """Load trained models."""
        try:
            self.info_text.append(f"🔍 Looking for models in: {self.inference_engine.models_dir}")
            self.info_text.append(f"🔍 Script location: {Path(__file__).parent}")
            
            if self.inference_engine.load_models():
                self.info_text.append("✅ Models loaded successfully!")
                self.info_text.append(f"Classes: {', '.join(self.inference_engine.class_names)}")
                self.info_text.append(f"Best model: {self.inference_engine.best_model_name}")
            else:
                self.info_text.append("❌ Failed to load models!")
                self.info_text.append("Make sure to train models first using train_multiclass_mediapipe.py")
                # Add more debug info
                self.info_text.append(f"Debug: models_dir exists: {self.inference_engine.models_dir.exists()}")
                if self.inference_engine.models_dir.exists():
                    files = list(self.inference_engine.models_dir.glob("*"))
                    self.info_text.append(f"Debug: files in directory: {[f.name for f in files]}")
        except Exception as e:
            self.info_text.append(f"❌ Error loading models: {str(e)}")
            self.info_text.append(f"Exception type: {type(e).__name__}")
            import traceback
            self.info_text.append(f"Traceback: {traceback.format_exc()}")
    
    def start_camera(self):
        """Start camera capture."""
        self.camera_thread.start_camera()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.info_text.append("📹 Camera started")
    
    def stop_camera(self):
        """Stop camera capture."""
        self.camera_thread.stop_camera()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_label.setText("Camera Feed")
        self.info_text.append("📹 Camera stopped")
    
    def on_frame_received(self, frame: np.ndarray):
        """Handle frame from camera."""
        self.current_frame = frame
        
        # Make prediction
        if self.inference_engine.is_loaded:
            result = self.inference_engine.predict_from_frame(
                frame, 
                self.model_combo.currentText()
            )
            
            if 'error' not in result:
                self.prediction_history.append(result)
    
    def on_confidence_changed(self, value: int):
        """Handle confidence threshold change."""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
    
    def update_display(self):
        """Update the display."""
        if self.current_frame is not None:
            # Display camera feed
            height, width, channel = self.current_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.current_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Update predictions
        if self.prediction_history:
            latest_prediction = self.prediction_history[-1]
            confidence_threshold = self.confidence_slider.value() / 100.0
            
            if latest_prediction['confidence'] >= confidence_threshold:
                self.prediction_label.setText(f"🎯 {latest_prediction['predicted_gesture']}")
                self.prediction_label.setStyleSheet("color: green;")
            else:
                self.prediction_label.setText("❓ Uncertain")
                self.prediction_label.setStyleSheet("color: orange;")
            
            self.confidence_label_pred.setText(f"Confidence: {latest_prediction['confidence']:.3f}")
            
            # Update top predictions
            top_text = "Top Predictions:\n"
            for i, pred in enumerate(latest_prediction['top_predictions'], 1):
                top_text += f"{i}. {pred['gesture']}: {pred['confidence']:.3f}\n"
            self.top_predictions_text.setText(top_text)
            
            # Update system info
            info_text = f"Frames processed: {latest_prediction['frames_processed']}\n"
            info_text += f"Frames used: {latest_prediction.get('frames_used', 'N/A')}\n"
            info_text += f"Model: {self.model_combo.currentText()}\n"
            info_text += f"Threshold: {confidence_threshold:.2f}\n"
            self.info_text.setPlainText(info_text)


def main():
    """Main function."""
    print("Creating QApplication...")
    app = QApplication(sys.argv)
    
    print("Creating GestureInferenceApp window...")
    window = GestureInferenceApp()
    
    print("Showing window...")
    window.show()
    
    print(f"Window visible: {window.isVisible()}")
    print(f"Window geometry: {window.geometry()}")
    print("Window should be visible now!")
    
    # Force window to front
    window.raise_()
    window.activateWindow()
    
    print("Starting GUI event loop...")
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
