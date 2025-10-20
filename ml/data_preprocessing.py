"""
Data preprocessing pipeline for gesture classification.
Handles loading, cleaning, and feature extraction from records.jsonl.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class GestureDataPreprocessor:
    """Preprocesses gesture data from records.jsonl for machine learning."""
    
    def __init__(self, data_path: str = "data/records.jsonl"):
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self) -> pd.DataFrame:
        """Load data from records.jsonl file and discover gesture classes."""
        if not self.data_path.exists():
            print(f"Warning: {self.data_path} not found. Creating sample data...")
            return self._create_sample_data()
            
        records = []
        gesture_classes = set()
        
        # Labels to exclude
        excluded_labels = {'downward_dog', 'tadasana', 'tree_pose', 'warrior_1', 'warrior_2'}
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    # Validate required fields
                    if not all(key in record for key in ['id', 'label', 'landmarks', 'visibility']):
                        print(f"Warning: Skipping record {line_num} - missing required fields")
                        continue
                    
                    # Skip excluded labels
                    if record['label'] in excluded_labels:
                        print(f"Skipping record {line_num} - excluded label: {record['label']}")
                        continue
                    
                    # Discover gesture classes dynamically
                    gesture_classes.add(record['label'])
                    records.append(record)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
                    
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} records from {self.data_path}")
        print(f"Excluded labels: {sorted(excluded_labels)}")
        
        # Show discovered gesture classes
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            print(f"Discovered gesture classes: {sorted(gesture_classes)}")
            print(f"Label distribution: {dict(label_counts)}")
            print(f"Total unique gestures: {len(gesture_classes)}")
        
        return df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate random gesture labels - discover classes dynamically
        gestures = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
        labels = np.random.choice(gestures, n_samples)
        
        # Generate random pose landmarks (33 keypoints x 3 coordinates)
        landmarks = []
        for i in range(n_samples):
            # Generate realistic pose-like data
            landmark_data = np.random.rand(33, 3) * 0.8 + 0.1  # Normalized coordinates
            landmarks.append(landmark_data.tolist())
        
        # Generate visibility scores
        visibility = []
        for i in range(n_samples):
            vis_data = np.random.rand(33) * 0.5 + 0.5  # Visibility between 0.5-1.0
            visibility.append(vis_data.tolist())
        
        data = {
            'id': [f'sample_{i:04d}' for i in range(n_samples)],
            'subject_id': [f'user_{np.random.randint(1, 6):03d}' for _ in range(n_samples)],
            'label': labels,
            'landmarks': landmarks,
            'visibility': visibility,
            'image_size': [[640, 480] for _ in range(n_samples)],
            'timestamp_ms': [i * 1000 for i in range(n_samples)],
            'camera_index': [0] * n_samples,
            'note': [''] * n_samples
        }
        
        df = pd.DataFrame(data)
        print(f"Created {len(df)} sample records")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from pose landmarks."""
        features = []
        labels = []
        
        print(f"Processing {len(df)} records...")
        
        for idx, row in df.iterrows():
            if not row['landmarks'] or len(row['landmarks']) == 0:
                print(f"Skipping record {idx}: no landmarks")
                continue
                
            landmarks = np.array(row['landmarks'])
            visibility = np.array(row['visibility']) if row['visibility'] else np.ones(33)
            
            # Validate landmarks shape (should be 33 keypoints x 3 coordinates)
            if landmarks.shape != (33, 3):
                print(f"Skipping record {idx}: invalid landmarks shape {landmarks.shape}")
                continue
            
            # Filter out low-visibility landmarks
            valid_mask = visibility > 0.5
            if np.sum(valid_mask) < 10:  # Need at least 10 visible landmarks
                print(f"Skipping record {idx}: insufficient visible landmarks ({np.sum(valid_mask)})")
                continue
            
            # Extract geometric features
            feature_vector = self._extract_geometric_features(landmarks, visibility)
            features.append(feature_vector)
            labels.append(row['label'])
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Labels found: {np.unique(y)}")
        return X, y
    
    def analyze_gesture_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze gesture patterns in the dataset."""
        if 'label' not in df.columns:
            return {'error': 'No label column found'}
        
        analysis = {}
        
        # Basic statistics
        gesture_counts = df['label'].value_counts()
        analysis['gesture_counts'] = dict(gesture_counts)
        analysis['total_gestures'] = len(gesture_counts)
        analysis['total_samples'] = len(df)
        
        # Gesture distribution
        analysis['gesture_distribution'] = {
            gesture: {
                'count': count,
                'percentage': (count / len(df)) * 100
            }
            for gesture, count in gesture_counts.items()
        }
        
        # Subject analysis (if available)
        if 'subject_id' in df.columns:
            subject_gestures = df.groupby('subject_id')['label'].value_counts()
            analysis['subject_gesture_counts'] = subject_gestures.to_dict()
        
        # Temporal analysis (if timestamp available)
        if 'timestamp_ms' in df.columns:
            df_sorted = df.sort_values('timestamp_ms')
            gesture_transitions = []
            for i in range(1, len(df_sorted)):
                prev_gesture = df_sorted.iloc[i-1]['label']
                curr_gesture = df_sorted.iloc[i]['label']
                if prev_gesture != curr_gesture:
                    gesture_transitions.append((prev_gesture, curr_gesture))
            
            analysis['gesture_transitions'] = gesture_transitions[:10]  # First 10 transitions
        
        return analysis
    
    def _extract_geometric_features(self, landmarks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """Extract geometric features from pose landmarks."""
        features = []
        
        # MediaPipe pose landmark indices
        # Key body parts for yoga poses
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
    
    def _calculate_angles(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> List[float]:
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
    
    def preprocess(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Complete preprocessing pipeline."""
        # Load data
        df = self.load_data()
        
        # Extract features
        X, y = self.extract_features(df)
        
        if len(X) == 0:
            raise ValueError("No valid samples found in data")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        class_names = self.label_encoder.classes_.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature names for later use
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        print(f"Classes: {class_names}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, class_names
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names
