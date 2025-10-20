"""
Gesture classification model using multiple ML algorithms.
Supports Random Forest, SVM, and Neural Network classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import json
from pathlib import Path


class GestureClassifier:
    """Multi-algorithm gesture classifier with model comparison."""
    
    def __init__(self, models_dir: str = "ml/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.class_names = []
        
        # Initialize models with default parameters
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        }
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    class_names: List[str], optimize: bool = True) -> Dict[str, float]:
        """Train all models and return their scores."""
        self.class_names = class_names
        
        print("Training gesture classification models...")
        scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if optimize and name in ['random_forest', 'svm']:
                model = self._optimize_model(name, X_train, y_train)
                self.models[name] = model
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            scores[name] = cv_scores.mean()
            
            print(f"{name} CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        self.best_model_name = max(scores.keys(), key=lambda k: scores[k])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest model: {self.best_model_name} (accuracy: {scores[self.best_model_name]:.4f})")
        
        return scores
    
    def _optimize_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """Optimize model hyperparameters using GridSearchCV."""
        print(f"Optimizing {model_name} hyperparameters...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            base_model = SVC(random_state=42, probability=True)
        
        else:
            return self.models[model_name]
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='accuracy', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models on test data."""
        results = {}
        
        print("\nEvaluating models on test data...")
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            print(f"{name} test accuracy: {accuracy:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using specified model or best model."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities
    
    def predict_single(self, features: np.ndarray, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict gesture for a single sample."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions, probabilities = self.predict(features, model_name)
        
        gesture = self.class_names[predictions[0]]
        confidence = probabilities[0][predictions[0]] if probabilities is not None else 1.0
        
        # Get top 3 predictions
        if probabilities is not None:
            top_indices = np.argsort(probabilities[0])[-3:][::-1]
            top_predictions = [
                {
                    'gesture': self.class_names[i],
                    'confidence': float(probabilities[0][i])
                }
                for i in top_indices
            ]
        else:
            top_predictions = [{'gesture': gesture, 'confidence': confidence}]
        
        return {
            'predicted_gesture': gesture,
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    
    def save_models(self, filename_prefix: str = "gesture_classifier"):
        """Save all trained models."""
        for name, model in self.models.items():
            model_path = self.models_dir / f"{filename_prefix}_{name}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'class_names': self.class_names,
            'model_count': len(self.models)
        }
        
        metadata_path = self.models_dir / f"{filename_prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")
    
    def load_models(self, filename_prefix: str = "gesture_classifier"):
        """Load trained models."""
        metadata_path = self.models_dir / f"{filename_prefix}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.class_names = metadata['class_names']
        self.best_model_name = metadata['best_model']
        
        # Load models
        for name in self.models.keys():
            model_path = self.models_dir / f"{filename_prefix}_{name}.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                print(f"Loaded {name} model from {model_path}")
        
        self.best_model = self.models[self.best_model_name]
        print(f"Best model: {self.best_model_name}")
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get feature importance from tree-based models."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models."""
        info = {
            'best_model': self.best_model_name,
            'class_names': self.class_names,
            'models': {}
        }
        
        for name, model in self.models.items():
            info['models'][name] = {
                'type': type(model).__name__,
                'parameters': model.get_params() if hasattr(model, 'get_params') else {}
            }
        
        return info
