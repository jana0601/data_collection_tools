"""
Command-line Gesture Classification Inference.
Simple CLI tool for gesture prediction from video files or camera.
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import joblib
import time
from collections import deque


class CLIGestureInference:
    """Command-line gesture inference tool."""
    
    def __init__(self, models_dir: str = "ml/models_2class"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.class_names = []
        self.target_frames = 30
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=self.target_frames)
        
    def load_models(self) -> bool:
        """Load trained models."""
        try:
            # Load metadata
            metadata_path = self.models_dir / "gesture_multiclass_metadata.json"
            if not metadata_path.exists():
                print(f"[ERROR] Metadata not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.class_names = metadata['class_names']
            self.best_model_name = metadata['best_model']
            
            # Load models
            model_files = {
                'random_forest': 'gesture_multiclass_random_forest.joblib',
                'svm': 'gesture_multiclass_svm.joblib',
                'neural_network': 'gesture_multiclass_neural_network.joblib'
            }
            
            for name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                    print(f"[OK] Loaded {name} model")
            
            if self.best_model_name in self.models:
                self.best_model = self.models[self.best_model_name]
                print(f"[OK] Best model: {self.best_model_name}")
                print(f"[OK] Classes: {self.class_names}")
                return True
            else:
                print(f"[ERROR] Best model {self.best_model_name} not found")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            return False
    
    def extract_features_from_frame(self, frame: np.ndarray) -> np.ndarray:
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
    
    def predict_from_frame(self, frame: np.ndarray, model_name: str = None) -> dict:
        """Predict gesture from a single frame."""
        if not self.best_model:
            return {'error': 'Models not loaded'}
        
        # Extract features from frame
        features = self.extract_features_from_frame(frame)
        
        # Add to frame buffer
        self.frame_buffer.append(features)
        
        # Need at least some frames for prediction
        if len(self.frame_buffer) < 5:
            return {'error': 'Not enough frames', 'frames': len(self.frame_buffer)}
        
        # Use specified model or best model
        model = self.models.get(model_name, self.best_model)
        if model is None:
            return {'error': f'Model {model_name} not found'}
        
        # Prepare features (pad with zeros if needed)
        video_features = []
        for i in range(self.target_frames):
            if i < len(self.frame_buffer):
                video_features.extend(self.frame_buffer[i])
            else:
                video_features.extend([0] * 23)  # Zero padding
        
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
                'frames_processed': len(self.frame_buffer)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_from_video_file(self, video_path: str, model_name: str = None, confidence_threshold: float = 0.5) -> list:
        """Predict gestures from a video file."""
        cap = cv2.VideoCapture(video_path)
        predictions = []
        
        print(f"Processing video: {video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            result = self.predict_from_frame(frame, model_name)
            
            if 'error' not in result:
                if result['confidence'] >= confidence_threshold:
                    predictions.append({
                        'frame': frame_count,
                        'prediction': result['predicted_gesture'],
                        'confidence': result['confidence']
                    })
                    
                    if frame_count % 30 == 0:  # Print every 30 frames
                        print(f"Frame {frame_count}: {result['predicted_gesture']} (confidence: {result['confidence']:.3f})")
        
        cap.release()
        return predictions
    
    def predict_from_camera(self, model_name: str = None, confidence_threshold: float = 0.5, duration: int = 10):
        """Predict gestures from live camera feed."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Failed to open camera")
            return
        
        print(f"[CAMERA] Starting camera prediction for {duration} seconds...")
        print("Press 'q' to quit early")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            result = self.predict_from_frame(frame, model_name)
            
            if 'error' not in result:
                if result['confidence'] >= confidence_threshold:
                    print(f"Frame {frame_count}: {result['predicted_gesture']} (confidence: {result['confidence']:.3f})")
                else:
                    print(f"Frame {frame_count}: Uncertain (confidence: {result['confidence']:.3f})")
            
            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"[OK] Processed {frame_count} frames")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Gesture Classification Inference CLI')
    parser.add_argument('--models-dir', default='ml/models_2class',
                       help='Directory containing trained models')
    parser.add_argument('--model', choices=['random_forest', 'svm', 'neural_network'],
                       help='Specific model to use (default: best model)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--video', type=str,
                       help='Path to video file for prediction')
    parser.add_argument('--camera', action='store_true',
                       help='Use live camera feed')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration for camera prediction (seconds)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = CLIGestureInference(args.models_dir)
    
    # Load models
    if not inference.load_models():
        print("[ERROR] Failed to load models. Exiting.")
        return
    
    print(f"[OK] Using model: {args.model or inference.best_model_name}")
    print(f"[OK] Confidence threshold: {args.confidence}")
    
    # Run prediction
    if args.video:
        predictions = inference.predict_from_video_file(
            args.video, 
            args.model, 
            args.confidence
        )
        
        print(f"\n[SUMMARY]")
        print(f"Total predictions: {len(predictions)}")
        
        # Count predictions by gesture
        gesture_counts = {}
        for pred in predictions:
            gesture = pred['prediction']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        for gesture, count in gesture_counts.items():
            print(f"{gesture}: {count} predictions")
    
    elif args.camera:
        inference.predict_from_camera(
            args.model,
            args.confidence,
            args.duration
        )
    
    else:
        print("Please specify --video or --camera")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
