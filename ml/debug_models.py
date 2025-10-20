#!/usr/bin/env python3
"""
Debug script to test model loading in different contexts.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_app import GestureInferenceEngine

def test_model_loading():
    print("=" * 50)
    print("DEBUG: Model Loading Test")
    print("=" * 50)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {Path(__file__).parent}")
    print(f"Script absolute path: {Path(__file__).absolute()}")
    
    # Test 1: Default initialization
    print("\n--- Test 1: Default initialization ---")
    engine1 = GestureInferenceEngine()
    print(f"Models dir: {engine1.models_dir}")
    print(f"Models dir exists: {engine1.models_dir.exists()}")
    if engine1.models_dir.exists():
        files = list(engine1.models_dir.glob("*"))
        print(f"Files in models dir: {[f.name for f in files]}")
    
    success1 = engine1.load_models()
    print(f"Load models success: {success1}")
    
    # Test 2: Explicit path
    print("\n--- Test 2: Explicit path ---")
    explicit_path = Path(__file__).parent / "models_2class"
    print(f"Explicit path: {explicit_path}")
    print(f"Explicit path exists: {explicit_path.exists()}")
    
    engine2 = GestureInferenceEngine(str(explicit_path))
    print(f"Engine models dir: {engine2.models_dir}")
    success2 = engine2.load_models()
    print(f"Load models success: {success2}")
    
    # Test 3: Absolute path
    print("\n--- Test 3: Absolute path ---")
    abs_path = Path(__file__).parent.absolute() / "models_2class"
    print(f"Absolute path: {abs_path}")
    print(f"Absolute path exists: {abs_path.exists()}")
    
    engine3 = GestureInferenceEngine(str(abs_path))
    print(f"Engine models dir: {engine3.models_dir}")
    success3 = engine3.load_models()
    print(f"Load models success: {success3}")

if __name__ == "__main__":
    test_model_loading()
