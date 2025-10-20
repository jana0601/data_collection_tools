"""
Complete example demonstrating the gesture classification pipeline.
Shows data preprocessing, model training, evaluation, and inference.
"""

import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ml import (
    GestureDataPreprocessor, 
    GestureClassifier, 
    GestureVisualizer,
    GestureInference
)


def main():
    """Complete gesture classification pipeline example."""
    print("=" * 60)
    print("GESTURE CLASSIFICATION PIPELINE DEMO")
    print("=" * 60)
    
    # 1. Data Preprocessing
    print("\n1. DATA PREPROCESSING")
    print("-" * 30)
    
    preprocessor = GestureDataPreprocessor("data/records.jsonl")
    X_train, X_test, y_train, y_test, class_names = preprocessor.preprocess()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {class_names}")
    
    # 2. Model Training
    print("\n2. MODEL TRAINING")
    print("-" * 30)
    
    classifier = GestureClassifier("ml/models")
    training_scores = classifier.train_models(X_train, y_train, class_names, optimize=False)
    
    print("Training completed!")
    for model_name, score in training_scores.items():
        print(f"{model_name}: {score:.4f}")
    
    # 3. Model Evaluation
    print("\n3. MODEL EVALUATION")
    print("-" * 30)
    
    evaluation_results = classifier.evaluate_models(X_test, y_test)
    
    print("Test results:")
    for model_name, results in evaluation_results.items():
        print(f"{model_name}: {results['accuracy']:.4f}")
    
    # 4. Visualization
    print("\n4. GENERATING VISUALIZATIONS")
    print("-" * 30)
    
    visualizer = GestureVisualizer("ml/results")
    
    # Model comparison
    visualizer.plot_model_comparison(evaluation_results)
    
    # Confusion matrices
    for model_name, results in evaluation_results.items():
        visualizer.plot_confusion_matrix(
            np.array(results['confusion_matrix']), 
            class_names, 
            model_name
        )
    
    # Classification reports
    visualizer.plot_classification_reports(evaluation_results, class_names)
    
    # Feature importance
    feature_importance = classifier.get_feature_importance()
    if feature_importance is not None:
        feature_names = preprocessor.get_feature_names()
        visualizer.plot_feature_importance(feature_importance, feature_names)
    
    # Training summary
    training_results = {
        'training_scores': training_scores,
        'evaluation_results': evaluation_results,
        'class_names': class_names,
        'data_shape': {
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X_train.shape[1]
        }
    }
    visualizer.plot_training_summary(training_results)
    
    # 5. Model Inference
    print("\n5. MODEL INFERENCE")
    print("-" * 30)
    
    # Save models
    classifier.save_models()
    
    # Test inference
    inference = GestureInference()
    inference.load_model()
    
    # Test on a few samples
    for i in range(3):
        sample_features = X_test[i:i+1]
        landmarks = np.random.rand(33, 3)  # Mock landmarks for demo
        visibility = np.random.rand(33)
        
        result = inference.predict_from_landmarks(landmarks, visibility)
        
        print(f"\nSample {i+1}:")
        print(f"Predicted: {result['predicted_gesture']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"True label: {class_names[y_test[i]]}")
    
    # 6. Batch Prediction
    print("\n6. BATCH PREDICTION")
    print("-" * 30)
    
    # Create sample records
    sample_records = []
    for i in range(5):
        record = {
            'id': f'sample_{i}',
            'landmarks': np.random.rand(33, 3).tolist(),
            'visibility': np.random.rand(33).tolist(),
            'label': np.random.choice(class_names)
        }
        sample_records.append(record)
    
    batch_results = inference.batch_predict(sample_records)
    
    print("Batch prediction results:")
    for record, result in zip(sample_records, batch_results):
        print(f"ID: {record['id']}")
        print(f"True: {record['label']}, Predicted: {result['predicted_gesture']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print()
    
    # 7. Model Information
    print("\n7. MODEL INFORMATION")
    print("-" * 30)
    
    model_info = inference.get_model_info()
    print("Model information:")
    print(json.dumps(model_info, indent=2))
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFiles created:")
    print("- ml/models/: Trained models")
    print("- ml/results/: Visualizations and reports")
    print("- Check the results directory for generated plots")


def quick_demo():
    """Quick demo with minimal output."""
    print("Quick Gesture Classification Demo")
    print("=" * 40)
    
    # Initialize components
    preprocessor = GestureDataPreprocessor()
    classifier = GestureClassifier()
    
    # Process data
    X_train, X_test, y_train, y_test, class_names = preprocessor.preprocess()
    
    # Train models
    training_scores = classifier.train_models(X_train, y_train, class_names, optimize=False)
    
    # Evaluate
    evaluation_results = classifier.evaluate_models(X_test, y_test)
    
    # Show results
    print(f"Best model: {classifier.best_model_name}")
    print(f"Best accuracy: {training_scores[classifier.best_model_name]:.4f}")
    print(f"Test accuracy: {evaluation_results[classifier.best_model_name]['accuracy']:.4f}")
    
    # Save models
    classifier.save_models()
    print("Models saved successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        main()
