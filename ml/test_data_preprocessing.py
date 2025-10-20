#!/usr/bin/env python3
"""
Test script to verify data preprocessing works with new dataset format.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import GestureDataPreprocessor

def test_data_preprocessing():
    """Test data preprocessing with gesture discovery."""
    print("Testing gesture discovery and data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = GestureDataPreprocessor("data/records.jsonl")
    
    # Load data and discover gestures
    print("\n1. Loading data and discovering gesture classes...")
    df = preprocessor.load_data()
    
    if len(df) == 0:
        print("No data loaded - this is expected if records.jsonl doesn't exist")
        return
    
    # Analyze gesture patterns
    print("\n2. Analyzing gesture patterns...")
    analysis = preprocessor.analyze_gesture_patterns(df)
    
    print(f"Gesture Analysis Results:")
    print(f"- Total gestures discovered: {analysis['total_gestures']}")
    print(f"- Total samples: {analysis['total_samples']}")
    print(f"- Gesture distribution:")
    for gesture, info in analysis['gesture_distribution'].items():
        print(f"  * {gesture}: {info['count']} samples ({info['percentage']:.1f}%)")
    
    # Extract features
    print("\n3. Extracting features...")
    X, y = preprocessor.extract_features(df)
    
    print(f"\nFeature Extraction Results:")
    print(f"- Features shape: {X.shape}")
    print(f"- Labels shape: {y.shape}")
    print(f"- Unique labels: {set(y)}")
    print(f"- Feature names count: {len(preprocessor.feature_names)}")
    
    # Test preprocessing pipeline
    print("\n4. Testing full preprocessing pipeline...")
    try:
        X_train, X_test, y_train, y_test, class_names = preprocessor.preprocess()
        print(f"- Training set: {X_train.shape}")
        print(f"- Test set: {X_test.shape}")
        print(f"- Discovered class names: {class_names}")
        print("SUCCESS: Preprocessing pipeline successful!")
    except Exception as e:
        print(f"ERROR: Preprocessing pipeline failed: {e}")

if __name__ == "__main__":
    test_data_preprocessing()
