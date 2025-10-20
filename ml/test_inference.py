#!/usr/bin/env python3
"""
Simple command-line test of the inference engine.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_app import GestureInferenceEngine

def test_inference():
    print("Testing GestureInferenceEngine...")
    
    # Initialize engine
    engine = GestureInferenceEngine()
    print(f"Models directory: {engine.models_dir}")
    
    # Load models
    if not engine.load_models():
        print("ERROR: Failed to load models!")
        return False
    
    print("SUCCESS: Models loaded successfully!")
    print(f"Classes: {engine.class_names}")
    print(f"Best model: {engine.best_model_name}")
    
    # Create a dummy frame (random image)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test prediction with multiple frames
    print("\nTesting prediction...")
    
    # Feed multiple frames to build up the buffer
    for i in range(10):
        result = engine.predict_from_frame(dummy_frame)
        if 'error' not in result:
            break
        print(f"Frame {i+1}: {result.get('error', 'Processing...')}")
    
    if 'error' in result:
        print(f"ERROR: Prediction error: {result['error']}")
        return False
    
    print("SUCCESS: Prediction successful!")
    print(f"Predicted gesture: {result['predicted_gesture']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Frames processed: {result['frames_processed']}")
    
    return True

if __name__ == "__main__":
    success = test_inference()
    if success:
        print("\nALL TESTS PASSED!")
    else:
        print("\nTESTS FAILED!")
        sys.exit(1)
