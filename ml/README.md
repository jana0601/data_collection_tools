# Gesture Classification ML Module

A comprehensive machine learning pipeline for yoga pose gesture classification using pose landmarks from MediaPipe.

## Features

- **Data Preprocessing**: Load and process pose landmark data from `records.jsonl`
- **Multiple ML Models**: Random Forest, SVM, and Neural Network classifiers
- **Hyperparameter Optimization**: Automatic tuning for best performance
- **Comprehensive Visualizations**: Confusion matrices, feature importance, model comparison
- **Real-time Inference**: Easy-to-use API for live gesture classification
- **Model Persistence**: Save and load trained models

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ml/requirements.txt
```

### 2. Train Models

```bash
# Full training with optimization
python ml/train.py

# Quick training without optimization
python ml/train.py --no-optimize

# Use custom data path
python ml/train.py --data-path your_data/records.jsonl
```

### 3. Run Example

```bash
# Complete pipeline demo
python ml/example.py

# Quick demo
python ml/example.py --quick
```

### 4. Use Inference API

```python
from ml import GestureInference

# Initialize inference
inference = GestureInference()
inference.load_model()

# Predict from landmarks
landmarks = np.random.rand(33, 3)  # 33 pose keypoints
result = inference.predict_from_landmarks(landmarks)

print(f"Predicted: {result['predicted_gesture']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Data Format

The system expects data in `records.jsonl` format:

```json
{
  "id": "sample_001",
  "subject_id": "user_001", 
  "label": "tadasana",
  "landmarks": [[x1, y1, z1], [x2, y2, z2], ...],  // 33 keypoints
  "visibility": [v1, v2, v3, ...],  // 33 visibility scores
  "image_size": [640, 480],
  "timestamp_ms": 1234567890,
  "camera_index": 0,
  "note": ""
}
```

## Model Architecture

### Feature Extraction
- **Joint Angles**: Shoulder, elbow, hip, knee angles
- **Body Proportions**: Shoulder width, hip width, torso length
- **Pose Stability**: Center of mass calculation
- **Limb Extensions**: Arm and leg extension distances
- **Body Orientation**: Pose tilt and alignment

### Available Models
1. **Random Forest**: Fast, interpretable, good for small datasets
2. **SVM**: Robust, works well with high-dimensional features
3. **Neural Network**: Flexible, can learn complex patterns

## File Structure

```
ml/
├── __init__.py              # Module initialization
├── requirements.txt         # Dependencies
├── data_preprocessing.py    # Data loading and feature extraction
├── gesture_classifier.py    # ML models and training
├── visualization.py         # Plotting and charts
├── inference.py            # Real-time prediction API
├── train.py               # Training script
├── example.py             # Complete pipeline demo
├── models/                # Saved models (created after training)
└── results/               # Visualizations and reports
```

## API Reference

### GestureDataPreprocessor
- `load_data()`: Load data from records.jsonl
- `extract_features()`: Extract geometric features from landmarks
- `preprocess()`: Complete preprocessing pipeline

### GestureClassifier
- `train_models()`: Train all available models
- `evaluate_models()`: Evaluate on test data
- `predict_single()`: Predict gesture for one sample
- `save_models()`: Save trained models
- `load_models()`: Load saved models

### GestureInference
- `load_model()`: Load trained model
- `predict_from_landmarks()`: Predict from pose landmarks
- `batch_predict()`: Predict multiple samples
- `predict_with_confidence_threshold()`: Predict with confidence filtering

### GestureVisualizer
- `plot_model_comparison()`: Compare model performance
- `plot_confusion_matrix()`: Confusion matrix heatmap
- `plot_feature_importance()`: Feature importance bar chart
- `plot_training_summary()`: Training statistics overview

## Performance Tips

1. **Data Quality**: Ensure landmarks have good visibility scores (>0.5)
2. **Feature Engineering**: The system automatically extracts relevant geometric features
3. **Model Selection**: Random Forest is usually fastest, Neural Network most flexible
4. **Confidence Thresholding**: Use confidence scores to filter uncertain predictions

## Troubleshooting

### Common Issues

1. **No data found**: Create sample data or check `records.jsonl` path
2. **Low accuracy**: Check data quality and try different models
3. **Memory issues**: Reduce batch size or use smaller models
4. **Import errors**: Install all requirements with `pip install -r ml/requirements.txt`

### Debug Mode

Enable verbose output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Basic Training
```python
from ml import GestureDataPreprocessor, GestureClassifier

# Load and preprocess data
preprocessor = GestureDataPreprocessor("data/records.jsonl")
X_train, X_test, y_train, y_test, class_names = preprocessor.preprocess()

# Train models
classifier = GestureClassifier()
classifier.train_models(X_train, y_train, class_names)

# Evaluate
results = classifier.evaluate_models(X_test, y_test)
print(f"Best model: {classifier.best_model_name}")
```

### Real-time Classification
```python
from ml import RealTimeGestureClassifier

# Initialize real-time classifier
rt_classifier = RealTimeGestureClassifier()
rt_classifier.initialize()

# Classify live frames
landmarks = get_pose_landmarks()  # Your pose detection
result = rt_classifier.classify_frame(landmarks)
print(f"Gesture: {result['predicted_gesture']}")
```

### Custom Feature Extraction
```python
from ml.data_preprocessing import GestureDataPreprocessor

preprocessor = GestureDataPreprocessor()

# Extract features from custom landmarks
landmarks = np.array([[x, y, z] for x, y, z in your_landmarks])
visibility = np.ones(33)  # All landmarks visible
features = preprocessor._extract_geometric_features(landmarks, visibility)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
