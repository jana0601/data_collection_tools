"""
Model inference interface for real-time gesture classification.
Provides easy-to-use API for making predictions on new pose data.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from .data_preprocessing import GestureDataPreprocessor
from .gesture_classifier import GestureClassifier


class GestureInference:
    """Real-time gesture inference interface."""
    
    def __init__(self, models_dir: str = "ml/models", 
                 data_path: str = "data/records.jsonl"):
        self.models_dir = Path(models_dir)
        self.data_path = data_path
        
        # Initialize components
        self.preprocessor = GestureDataPreprocessor(data_path)
        self.classifier = GestureClassifier(str(self.models_dir))
        
        self.is_loaded = False
        self.class_names = []
        
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """Load trained model for inference."""
        try:
            self.classifier.load_models()
            self.class_names = self.classifier.class_names
            self.is_loaded = True
            
            if model_name and model_name in self.classifier.models:
                self.classifier.best_model_name = model_name
                self.classifier.best_model = self.classifier.models[model_name]
            
            print(f"Model loaded successfully. Best model: {self.classifier.best_model_name}")
            print(f"Available classes: {', '.join(self.class_names)}")
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Please train a model first using train.py")
            return False
    
    def predict_from_landmarks(self, landmarks: Union[np.ndarray, List[List[float]]], 
                              visibility: Optional[Union[np.ndarray, List[float]]] = None,
                              model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict gesture from pose landmarks."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy array if needed
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        
        if visibility is not None and isinstance(visibility, list):
            visibility = np.array(visibility)
        
        # Extract features
        features = self.preprocessor._extract_geometric_features(landmarks, visibility)
        
        # Make prediction
        result = self.classifier.predict_single(features, model_name)
        
        return result
    
    def predict_from_record(self, record: Dict[str, Any], 
                           model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict gesture from a data record."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        landmarks = np.array(record['landmarks'])
        visibility = np.array(record['visibility']) if record['visibility'] else None
        
        return self.predict_from_landmarks(landmarks, visibility, model_name)
    
    def batch_predict(self, records: List[Dict[str, Any]], 
                     model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Predict gestures for multiple records."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for record in records:
            try:
                result = self.predict_from_record(record, model_name)
                results.append(result)
            except Exception as e:
                print(f"Error processing record {record.get('id', 'unknown')}: {e}")
                results.append({
                    'predicted_gesture': 'unknown',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def predict_with_confidence_threshold(self, landmarks: Union[np.ndarray, List[List[float]]], 
                                        visibility: Optional[Union[np.ndarray, List[float]]] = None,
                                        threshold: float = 0.5,
                                        model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict gesture with confidence threshold."""
        result = self.predict_from_landmarks(landmarks, visibility, model_name)
        
        if result['confidence'] < threshold:
            result['predicted_gesture'] = 'uncertain'
            result['confidence'] = result['confidence']
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        if not self.is_loaded:
            return {'error': 'No model loaded'}
        
        return self.classifier.get_model_info()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if not self.is_loaded:
            return []
        
        return list(self.classifier.models.keys())
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different loaded model."""
        if not self.is_loaded:
            print("No model loaded")
            return False
        
        if model_name not in self.classifier.models:
            print(f"Model {model_name} not available")
            return False
        
        self.classifier.best_model_name = model_name
        self.classifier.best_model = self.classifier.models[model_name]
        print(f"Switched to model: {model_name}")
        return True
    
    def evaluate_on_test_data(self, test_records: List[Dict[str, Any]], 
                             model_name: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate model on test data."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        predictions = self.batch_predict(test_records, model_name)
        
        # Calculate accuracy
        correct = 0
        total = len(test_records)
        
        for i, (record, pred) in enumerate(zip(test_records, predictions)):
            if pred.get('predicted_gesture') == record.get('label'):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate confidence statistics
        confidences = [pred.get('confidence', 0) for pred in predictions]
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'predictions': predictions
        }


class RealTimeGestureClassifier:
    """Real-time gesture classification for live camera feed."""
    
    def __init__(self, models_dir: str = "ml/models"):
        self.inference = GestureInference(models_dir)
        self.is_ready = False
        self.confidence_threshold = 0.5
        self.current_model = None
        
    def initialize(self, model_name: Optional[str] = None) -> bool:
        """Initialize the real-time classifier."""
        success = self.inference.load_model(model_name)
        if success:
            self.is_ready = True
            self.current_model = self.inference.classifier.best_model_name
        return success
    
    def classify_frame(self, landmarks: np.ndarray, 
                      visibility: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Classify a single frame."""
        if not self.is_ready:
            return {'error': 'Classifier not initialized'}
        
        try:
            result = self.inference.predict_with_confidence_threshold(
                landmarks, visibility, self.confidence_threshold
            )
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for predictions."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold set to {self.confidence_threshold}")
    
    def get_classification_history(self, max_length: int = 100) -> List[Dict[str, Any]]:
        """Get recent classification history (for smoothing)."""
        # This would be implemented with a circular buffer in a real application
        return []
    
    def smooth_predictions(self, predictions: List[Dict[str, Any]], 
                          window_size: int = 5) -> Dict[str, Any]:
        """Smooth predictions over a window."""
        if len(predictions) < window_size:
            return predictions[-1] if predictions else {'predicted_gesture': 'unknown'}
        
        # Get recent predictions
        recent = predictions[-window_size:]
        
        # Count predictions
        gesture_counts = {}
        confidences = []
        
        for pred in recent:
            gesture = pred.get('predicted_gesture', 'unknown')
            confidence = pred.get('confidence', 0)
            
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            confidences.append(confidence)
        
        # Find most common gesture
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        avg_confidence = np.mean(confidences)
        
        return {
            'predicted_gesture': most_common[0],
            'confidence': avg_confidence,
            'vote_count': most_common[1],
            'window_size': window_size
        }


def demo_inference():
    """Demonstration of the inference interface."""
    print("Gesture Classification Inference Demo")
    print("=" * 40)
    
    # Initialize inference
    inference = GestureInference()
    
    # Load model
    if not inference.load_model():
        print("Failed to load model. Please train a model first.")
        return
    
    # Create sample landmarks (33 keypoints x 3 coordinates)
    sample_landmarks = np.random.rand(33, 3) * 0.8 + 0.1
    sample_visibility = np.random.rand(33) * 0.5 + 0.5
    
    # Make prediction
    result = inference.predict_from_landmarks(sample_landmarks, sample_visibility)
    
    print(f"Predicted gesture: {result['predicted_gesture']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nTop 3 predictions:")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"{i}. {pred['gesture']}: {pred['confidence']:.3f}")
    
    # Test with confidence threshold
    result_threshold = inference.predict_with_confidence_threshold(
        sample_landmarks, sample_visibility, threshold=0.7
    )
    
    print(f"\nWith threshold 0.7: {result_threshold['predicted_gesture']}")
    print(f"Confidence: {result_threshold['confidence']:.3f}")


if __name__ == "__main__":
    demo_inference()
