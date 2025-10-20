"""
Multi-Class Gesture Classification using MediaPipe.
Automatically detects number of classes from video filenames.
Extracts pose landmarks from video frames and creates confusion matrix.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gesture_classifier import GestureClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Import MediaPipe
import mediapipe as mp


class MediaPipeVideoDataPreprocessor:
    """Video data preprocessor using MediaPipe for pose landmark extraction.
    Automatically detects number of classes from video filenames."""
    
    def __init__(self, data_path: str = "E:/From_C/llm_project/YogaPose/data/static_gestures/images", 
                 target_frames: int = 30):
        self.data_path = Path(data_path)
        self.target_frames = target_frames
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
    def extract_class_from_filename(self, filename: str) -> str:
        """Extract class label from filename (e.g., S1_g1_20251020T165918_687.mp4 -> g1)."""
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1]  # g1, g2, etc.
        return 'unknown'
    
    def load_video_frames(self, video_path: Path) -> list:
        """Load frames from video file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def extract_pose_features_from_frames(self, frames: list) -> np.ndarray:
        """Extract pose features from video frames using MediaPipe."""
        features_per_frame = []
        
        for frame in frames:
            # Process frame with MediaPipe
            results = self.pose.process(frame)
            
            if results.pose_landmarks is not None:
                # Extract landmarks (33 keypoints x 3 coordinates)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
                
                # Extract geometric features for this frame
                frame_features = self._extract_geometric_features(landmarks, visibility)
                features_per_frame.append(frame_features)
            else:
                # No pose detected - use zero features
                zero_features = np.zeros(23)  # Same as single frame features
                features_per_frame.append(zero_features)
        
        # Convert to numpy array
        features_array = np.array(features_per_frame)
        
        # Zero padding if less than target_frames
        if len(features_per_frame) < self.target_frames:
            padding_needed = self.target_frames - len(features_per_frame)
            padding = np.zeros((padding_needed, features_array.shape[1]))
            features_array = np.vstack([features_array, padding])
        
        # Truncate if more than target_frames
        elif len(features_per_frame) > self.target_frames:
            features_array = features_array[:self.target_frames]
        
        return features_array
    
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
    
    def detect_classes(self):
        """Detect all unique classes from video filenames."""
        video_files = list(self.data_path.glob("*.mp4"))
        unique_classes = set()
        
        for video_path in video_files:
            class_label = self.extract_class_from_filename(video_path.name)
            unique_classes.add(class_label)
        
        return sorted(list(unique_classes))
    
    def load_video_data(self):
        """Load all video data and extract features."""
        print(f"Loading video data from {self.data_path}")
        
        video_files = list(self.data_path.glob("*.mp4"))
        print(f"Found {len(video_files)} video files")
        
        # Detect classes automatically
        detected_classes = self.detect_classes()
        print(f"Automatically detected {len(detected_classes)} classes: {detected_classes}")
        
        # Count classes
        class_counts = {}
        for video_path in video_files:
            class_label = self.extract_class_from_filename(video_path.name)
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        print("Class distribution:")
        for class_label, count in class_counts.items():
            print(f"  {class_label}: {count} videos")
        
        all_features = []
        all_labels = []
        
        for i, video_path in enumerate(video_files):
            print(f"Processing {i+1}/{len(video_files)}: {video_path.name}")
            
            # Extract class from filename
            class_label = self.extract_class_from_filename(video_path.name)
            
            # Load video frames
            frames = self.load_video_frames(video_path)
            print(f"  Loaded {len(frames)} frames")
            
            # Extract features using MediaPipe
            features = self.extract_pose_features_from_frames(frames)
            
            all_features.append(features)
            all_labels.append(class_label)
        
        # Convert to numpy arrays
        X = np.array(all_features)  # Shape: (n_videos, target_frames, n_features)
        y = np.array(all_labels)
        
        print(f"Loaded {len(video_files)} videos")
        print(f"Feature shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        print(f"Number of classes: {len(detected_classes)}")
        
        return X, y
    
    def preprocess(self, test_size: float = 0.2, random_state: int = 42):
        """Complete preprocessing pipeline."""
        # Load video data
        X, y = self.load_video_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        class_names = self.label_encoder.classes_.tolist()
        
        # Reshape features for ML models (flatten time dimension)
        n_videos, n_frames, n_features = X.shape
        X_flat = X.reshape(n_videos, n_frames * n_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} videos")
        print(f"Test set: {X_test_scaled.shape[0]} videos")
        print(f"Features per video: {X_train_scaled.shape[1]} ({n_frames} frames × {n_features} features)")
        print(f"Classes: {class_names}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, class_names


def train_multiclass_model_mediapipe():
    """Train multi-class gesture classification model using MediaPipe."""
    print("Multi-Class Gesture Classification Training (MediaPipe)")
    print("=" * 60)
    
    # Initialize MediaPipe video data preprocessor
    preprocessor = MediaPipeVideoDataPreprocessor(
        data_path="E:/From_C/llm_project/YogaPose/data/static_gestures/images",
        target_frames=30
    )
    
    # Preprocess data
    print("\n1. Preprocessing video data...")
    X_train, X_test, y_train, y_test, class_names = preprocessor.preprocess()
    
    # Train models
    print("\n2. Training models...")
    classifier = GestureClassifier("models_2class")
    training_scores = classifier.train_models(X_train, y_train, class_names, optimize=False)
    
    print("Training completed!")
    for model_name, score in training_scores.items():
        print(f"{model_name}: {score:.4f}")
    
    # Evaluate models
    print("\n3. Evaluating models...")
    evaluation_results = classifier.evaluate_models(X_test, y_test)
    
    print("Test results:")
    for model_name, results in evaluation_results.items():
        print(f"{model_name}: {results['accuracy']:.4f}")
    
    # Save models
    print("\n4. Saving models...")
    classifier.save_models("gesture_multiclass")
    
    return evaluation_results, class_names, classifier


def show_confusion_matrix_multiclass(evaluation_results, class_names):
    """Display confusion matrix for multi-class classification."""
    print("\nCONFUSION MATRICES")
    print("=" * 50)
    
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print("-" * 30)
        
        cm = np.array(results['confusion_matrix'])
        accuracy = results['accuracy']
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Test samples: {cm.sum()}")
        print()
        
        # Print confusion matrix as table
        print("Confusion Matrix:")
        print("Rows = True Gesture, Columns = Predicted Gesture")
        print()
        
        # Header
        header = "True\\Pred".ljust(12)
        for gesture in class_names:
            header += gesture.ljust(12)
        print(header)
        print("-" * len(header))
        
        # Rows
        for i, true_gesture in enumerate(class_names):
            row = true_gesture.ljust(12)
            for j, pred_gesture in enumerate(class_names):
                count = cm[i, j]
                row += str(count).ljust(12)
            print(row)
        
        print()
        
        # Calculate precision, recall, F1 for each class
        print("Per-Class Metrics:")
        print("-" * 20)
        
        for i, gesture in enumerate(class_names):
            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{gesture}:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1-Score:  {f1:.3f}")
            print()
    
    # Create visualization
    create_confusion_matrix_plot_multiclass(evaluation_results, class_names)


def create_confusion_matrix_plot_multiclass(evaluation_results, class_names):
    """Create confusion matrix visualization for multi-class classification."""
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, results) in enumerate(evaluation_results.items()):
        cm = np.array(results['confusion_matrix'])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, 
                   ax=axes[i], cbar_kws={'label': 'Count'})
        
        axes[i].set_title(f'{model_name.title()}\nAccuracy: {results["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted Gesture')
        axes[i].set_ylabel('True Gesture')
        
        # Rotate labels for better readability
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results_multiclass")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'confusion_matrices_multiclass.png', 
                dpi=300, bbox_inches='tight')
    
    print(f"Confusion matrix plot saved to: {results_dir / 'confusion_matrices_multiclass.png'}")
    
    # Show plot
    plt.show()


def main():
    """Main function to run multi-class gesture classification."""
    try:
        # Train model
        evaluation_results, class_names, classifier = train_multiclass_model_mediapipe()
        
        # Show confusion matrix
        show_confusion_matrix_multiclass(evaluation_results, class_names)
        
        print("\n" + "=" * 60)
        print("MULTI-CLASS GESTURE CLASSIFICATION COMPLETED!")
        print("=" * 60)
        print(f"Detected {len(class_names)} classes: {class_names}")
        print("Using MediaPipe pose landmark extraction")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
