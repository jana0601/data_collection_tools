"""
Simple test script for the ML pipeline.
Tests the complete gesture classification system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from pathlib import Path

# Import our modules
from data_preprocessing import GestureDataPreprocessor
from gesture_classifier import GestureClassifier
from visualization import GestureVisualizer


def test_pipeline():
    """Test the complete ML pipeline."""
    print("Testing Gesture Classification Pipeline")
    print("=" * 50)
    
    try:
        # 1. Test data preprocessing
        print("\n1. Testing data preprocessing...")
        preprocessor = GestureDataPreprocessor()
        X_train, X_test, y_train, y_test, class_names = preprocessor.preprocess()
        
        print(f"[OK] Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"[OK] Features: {X_train.shape[1]}")
        print(f"[OK] Classes: {class_names}")
        
        # 2. Test model training
        print("\n2. Testing model training...")
        classifier = GestureClassifier("models")
        training_scores = classifier.train_models(X_train, y_train, class_names, optimize=False)
        
        print(f"[OK] Models trained successfully")
        for model_name, score in training_scores.items():
            print(f"  - {model_name}: {score:.4f}")
        
        # 3. Test evaluation
        print("\n3. Testing model evaluation...")
        evaluation_results = classifier.evaluate_models(X_test, y_test)
        
        print(f"[OK] Models evaluated successfully")
        for model_name, results in evaluation_results.items():
            print(f"  - {model_name}: {results['accuracy']:.4f}")
        
        # 4. Test model saving
        print("\n4. Testing model saving...")
        classifier.save_models()
        print("[OK] Models saved successfully")
        
        # 5. Test inference
        print("\n5. Testing inference...")
        
        # Test prediction using the trained classifier directly
        sample_features = preprocessor._extract_geometric_features(
            np.random.rand(33, 3), np.random.rand(33)
        )
        sample_features = sample_features.reshape(1, -1)
        
        # Scale features using the same scaler
        sample_features_scaled = preprocessor.scaler.transform(sample_features)
        
        # Make prediction
        prediction, probabilities = classifier.predict(sample_features_scaled)
        
        predicted_class = class_names[prediction[0]]
        confidence = probabilities[0][prediction[0]]
        
        print(f"[OK] Prediction successful: {predicted_class} (confidence: {confidence:.3f})")
        
        # 6. Test visualization (create a simple plot)
        print("\n6. Testing visualization...")
        visualizer = GestureVisualizer("results")
        
        # Create a simple model comparison plot
        visualizer.plot_model_comparison(evaluation_results)
        print("[OK] Visualizations created successfully")
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample records
    records = []
    gestures = ['tadasana', 'warrior_1', 'warrior_2', 'downward_dog', 'tree_pose']
    
    for i in range(100):
        record = {
            'id': f'sample_{i:03d}',
            'subject_id': f'user_{i%5+1:03d}',
            'label': np.random.choice(gestures),
            'landmarks': np.random.rand(33, 3).tolist(),
            'visibility': np.random.rand(33).tolist(),
            'image_size': [640, 480],
            'timestamp_ms': i * 1000,
            'camera_index': 0,
            'note': ''
        }
        records.append(record)
    
    # Save to JSONL
    records_path = data_dir / "records.jsonl"
    with open(records_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
        print(f"[OK] Created {len(records)} sample records in {records_path}")


if __name__ == "__main__":
    # Create sample data if needed
    if not Path("data/records.jsonl").exists():
        create_sample_data()
    
    # Run tests
    success = test_pipeline()
    
    if success:
        print("\n[SUCCESS] Gesture Classification ML Pipeline is working correctly!")
        print("\nNext steps:")
        print("1. Collect real pose data using the desktop app")
        print("2. Run: python ml/train.py")
        print("3. Use the trained models for real-time classification")
    else:
        print("\n[ERROR] There were issues with the pipeline. Check the error messages above.")
