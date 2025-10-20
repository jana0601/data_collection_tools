#!/usr/bin/env python3
"""
Gesture Discovery Demo - Shows how the system automatically discovers gesture classes
from any dataset without prior knowledge.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import GestureDataPreprocessor

def demo_gesture_discovery():
    """Demonstrate automatic gesture discovery."""
    print("=" * 60)
    print("GESTURE DISCOVERY DEMO")
    print("=" * 60)
    print("This demo shows how the system automatically discovers")
    print("gesture classes from any dataset without prior knowledge.")
    print()
    
    # Initialize preprocessor
    preprocessor = GestureDataPreprocessor("data/records.jsonl")
    
    # Load and discover gestures
    print("1. LOADING DATASET AND DISCOVERING GESTURES")
    print("-" * 50)
    df = preprocessor.load_data()
    
    if len(df) == 0:
        print("No dataset found. The system would work with any records.jsonl file.")
        print("Expected format: {'id': '...', 'label': 'GESTURE_NAME', 'landmarks': [...], 'visibility': [...]}")
        return
    
    # Analyze patterns
    print("\n2. ANALYZING GESTURE PATTERNS")
    print("-" * 50)
    analysis = preprocessor.analyze_gesture_patterns(df)
    
    print(f"Dataset Summary:")
    print(f"  - Total samples: {analysis['total_samples']}")
    print(f"  - Gesture classes discovered: {analysis['total_gestures']}")
    print(f"  - Classes: {sorted(analysis['gesture_counts'].keys())}")
    
    print(f"\nGesture Distribution:")
    for gesture, info in analysis['gesture_distribution'].items():
        print(f"  - {gesture}: {info['count']} samples ({info['percentage']:.1f}%)")
    
    # Show feature extraction
    print("\n3. FEATURE EXTRACTION")
    print("-" * 50)
    X, y = preprocessor.extract_features(df)
    print(f"  - Extracted {X.shape[0]} valid samples")
    print(f"  - {X.shape[1]} features per sample")
    print(f"  - Labels: {sorted(set(y))}")
    
    # Show preprocessing pipeline
    print("\n4. PREPROCESSING PIPELINE")
    print("-" * 50)
    try:
        X_train, X_test, y_train, y_test, class_names = preprocessor.preprocess()
        print(f"  - Training set: {X_train.shape[0]} samples")
        print(f"  - Test set: {X_test.shape[0]} samples")
        print(f"  - Class names: {class_names}")
        print("\nSUCCESS: Ready for model training!")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("KEY BENEFITS:")
    print("- Automatically discovers gesture classes")
    print("- Works with any dataset format")
    print("- No need to predefine gesture names")
    print("- Handles missing/invalid data gracefully")
    print("- Provides detailed analysis and statistics")
    print("=" * 60)

if __name__ == "__main__":
    demo_gesture_discovery()
