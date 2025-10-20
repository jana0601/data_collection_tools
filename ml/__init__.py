"""
Machine Learning module for gesture classification.
Provides complete pipeline from data preprocessing to model inference.
"""

from .data_preprocessing import GestureDataPreprocessor
from .gesture_classifier import GestureClassifier
from .visualization import GestureVisualizer
from .inference import GestureInference, RealTimeGestureClassifier

__all__ = [
    'GestureDataPreprocessor',
    'GestureClassifier', 
    'GestureVisualizer',
    'GestureInference',
    'RealTimeGestureClassifier'
]

__version__ = '1.0.0'
