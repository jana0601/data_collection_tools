# Gesture Classification Inference Applications

This directory contains real-time gesture classification inference applications that use trained models to classify gestures from live camera feed or video files.

![Interface Screenshot](interface_classification.jpg)

## 🚀 Quick Start

### 1. Train Models (Required First Step)
```bash
python train_multiclass_mediapipe.py
```
**Important**: You must train models before running inference applications!

### 2. Check if Models Exist
```bash
python run_inference.py --check-models
```

### 3. Run GUI Application
```bash
python run_inference.py --mode gui
```

### 4. Run CLI Application
```bash
python run_inference.py --mode cli
```

## 📁 Files Overview

### Core Applications
- **`inference_app.py`** - GUI application with real-time camera feed
- **`inference_cli.py`** - Command-line interface for batch processing
- **`run_inference.py`** - Launcher script for easy access

### Supporting Files
- **`inference.py`** - Core inference engine (legacy)
- **`gesture_classifier.py`** - Model management and training utilities
- **`train_multiclass_mediapipe.py`** - **CRITICAL**: Model training script, process directly from video.
- **`data_preprocessing.py`** - Data loading and feature extraction

## 🎓 Model Training

### Training Script: `train_multiclass_mediapipe.py`

This script is **essential** for creating the models used by inference applications. It performs the complete machine learning pipeline:

#### What It Does
1. **Loads gesture data** from `data/static_gestures/records.jsonl`
2. **Extracts features** using MediaPipe pose landmarks
3. **Trains multiple models**:
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - Neural Network (Multi-layer Perceptron)
4. **Evaluates performance** and selects the best model
5. **Saves trained models** to `models_2class/` directory
6. **Generates metadata** for inference applications

#### Training Process
```bash
# Run the training script
python train_multiclass_mediapipe.py
```

**Expected Output:**
```
Loading data from data/static_gestures/records.jsonl...
Discovered gesture classes: ['g1', 'g2']
Label distribution: {'g1': 32, 'g2': 33}
Total samples: 65

Training Random Forest...
Training SVM...
Training Neural Network...

Model Performance:
- Random Forest: 76.92% accuracy
- SVM: 80.77% accuracy  
- Neural Network: 84.62% accuracy (BEST)

Saving models to models_2class/...
✅ Training completed successfully!
```

#### Data Requirements
- **Input**: `data/static_gestures/records.jsonl` with gesture samples
- **Format**: Each record contains `id`, `label`, `landmarks`, `visibility`
- **Minimum**: At least 2 gesture classes with multiple samples each
- **Recommended**: 20+ samples per gesture class for good performance

#### Output Files
After training, the following files are created in `models_2class/`:
- `gesture_2class_metadata.json` - Model metadata and class names
- `gesture_2class_random_forest.joblib` - Random Forest model
- `gesture_2class_svm.joblib` - SVM model  
- `gesture_2class_neural_network.joblib` - Neural Network model
- `gesture_multiclass_metadata.json` - Backup metadata file

#### Training Configuration
The script automatically:
- **Splits data**: 80% training, 20% testing
- **Scales features**: StandardScaler for neural network
- **Handles missing data**: Filters out invalid samples
- **Cross-validation**: 5-fold CV for model selection
- **Feature extraction**: 23 geometric features per sample

## 🎯 Features

### GUI Application (`inference_app.py`)
- **Real-time camera feed** with pose detection overlay
- **Live gesture classification** with confidence scores
- **Model selection** (Random Forest, SVM, Neural Network)
- **Confidence threshold** adjustment
- **Top predictions** display
- **System information** panel

### CLI Application (`inference_cli.py`)
- **Video file processing** for batch analysis
- **Live camera prediction** with duration control
- **Confidence threshold** filtering
- **Model selection** options
- **Summary statistics** for video analysis

## 🛠️ Usage Examples

### GUI Application
```bash
# Start GUI with default settings
python inference_app.py

# Or use the launcher
python run_inference.py --mode gui
```

**GUI Controls:**
- **Start Camera** - Begin live prediction
- **Stop Camera** - End prediction session
- **Model** - Select which trained model to use
- **Confidence Threshold** - Adjust minimum confidence (0.0-1.0)

### CLI Application

#### Process Video File
```bash
python inference_cli.py \
    --video "path/to/video.mp4" \
    --confidence 0.7 \
    --model neural_network
```

#### Live Camera Prediction
```bash
python inference_cli.py \
    --camera \
    --duration 30 \
    --confidence 0.8
```

#### Check Available Models
```bash
python inference_cli.py --models-dir models_2class --help
```

## 📊 Output Examples

### CLI Video Processing
```
[OK] Loaded random_forest model
[OK] Loaded svm model
[OK] Loaded neural_network model
[OK] Best model: neural_network
[OK] Classes: ['g1', 'g2']
[OK] Using model: neural_network
[OK] Confidence threshold: 0.7
Processing video: video.mp4
Frame 30: g1 (confidence: 0.994)
Frame 60: g1 (confidence: 0.987)
Frame 90: g1 (confidence: 0.992)

[SUMMARY]
Total predictions: 24
g1: 24 predictions
```

### GUI Real-time Display
- **Current Prediction**: 🎯 g1
- **Confidence**: 0.994
- **Top Predictions**:
  1. g1: 0.994
  2. g2: 0.006

## 🔧 Technical Details

### Model Requirements
- **Trained models** must exist in `models_2class/` directory
- **Metadata file**: `gesture_multiclass_metadata.json`
- **Model files**: `gesture_multiclass_*.joblib`

### Feature Extraction
- **MediaPipe pose landmarks** (33 points)
- **Geometric features** (23 features per frame):
  - Joint angles (12 features)
  - Body proportions (3 features)
  - Center of mass (3 features)
  - Arm/leg extensions (4 features)
  - Body orientation (1 feature)

### Frame Processing
- **Target frames**: 30 frames per prediction
- **Zero padding**: For videos with <30 frames
- **Buffer management**: Sliding window approach

## 🎨 GUI Interface

### Layout
```
┌─────────────────┬─────────────────┐
│   Camera Feed   │   Predictions   │
│                 │                 │
│                 │ Current: g1     │
│                 │ Confidence: 0.99│
│                 │                 │
│                 │ Top Predictions:│
│                 │ 1. g1: 0.994   │
│                 │ 2. g2: 0.006   │
│                 │                 │
│                 │ System Info:    │
│                 │ Frames: 30/30   │
│                 │ Model: Neural Net│
│                 │ Threshold: 0.70 │
└─────────────────┴─────────────────┘
```

### Controls Panel
- **Start/Stop Camera** buttons
- **Model selection** dropdown
- **Confidence threshold** slider (0.0-1.0)

## 🚨 Troubleshooting

### Common Issues

#### 1. Models Not Found
```
[ERROR] Metadata not found: models_2class/gesture_multiclass_metadata.json
```
**Solution**: 
1. **First, collect gesture data** using `desktop/app.py` (data collection)
2. **Then train models** using `train_multiclass_mediapipe.py`
3. **Finally, run inference** using these applications

**Complete Workflow:**
```bash
# Step 1: Collect gesture data
python desktop/app.py

# Step 2: Train models  
python train_multiclass_mediapipe.py

# Step 3: Run inference
python run_inference.py --mode gui
```

#### 2. Camera Not Opening
```
Failed to open camera
```
**Solution**: 
- Check camera permissions
- Try different camera index: `--camera-index 1`
- Ensure no other applications are using the camera

#### 3. Low Confidence Predictions
```
Frame 30: Uncertain (confidence: 0.45)
```
**Solution**:
- Lower confidence threshold
- Ensure good lighting
- Check pose visibility
- Verify model training quality

#### 4. Unicode Display Issues
**Solution**: Use the launcher script `run_inference.py` which handles encoding properly

### Performance Tips

1. **Lighting**: Ensure good, even lighting for pose detection
2. **Distance**: Stand 2-3 meters from camera for optimal pose detection
3. **Background**: Use plain background to avoid pose confusion
4. **Clothing**: Wear contrasting colors for better landmark detection
5. **Stability**: Hold poses for 1-2 seconds for better predictions

## 📈 Model Performance

Based on training results:
- **Neural Network**: Best overall performance (84.62% accuracy)
- **Random Forest**: Good performance with fast inference
- **SVM**: Moderate performance, good for simple gestures

## 🔄 Integration

### Complete Workflow
The inference applications are part of a complete gesture recognition pipeline:

1. **📹 Data Collection** (`desktop/app.py`)
   - Collect gesture samples using camera
   - Save landmarks and metadata to `data/static_gestures/records.jsonl`

2. **🎓 Model Training** (`train_multiclass_mediapipe.py`) 
   - **CRITICAL STEP**: Train ML models from collected data
   - Creates `models_2class/` directory with trained models
   - Generates metadata for inference applications

3. **🔮 Inference** (these applications)
   - Real-time gesture classification using trained models
   - GUI and CLI interfaces for different use cases

### With Data Collection
The inference applications work seamlessly with the data collection system:
1. **Collect data** using `desktop/app.py`
2. **Train models** using `train_multiclass_mediapipe.py` ⭐ **REQUIRED**
3. **Run inference** using these applications

### Custom Models
To use custom trained models:
1. Place models in appropriate directory
2. Update metadata file with class names
3. Modify `models_dir` parameter in applications

## 📝 Notes

- **Real-time performance**: ~30 FPS on modern hardware
- **Memory usage**: ~200MB for GUI application
- **Dependencies**: MediaPipe, OpenCV, PySide6 (GUI), scikit-learn
- **Platform**: Windows, macOS, Linux (with appropriate camera drivers)

## 🎉 Success Metrics

The inference applications successfully demonstrate:
- ✅ **Real-time classification** with high accuracy
- ✅ **User-friendly interfaces** (GUI and CLI)
- ✅ **Robust error handling** and validation
- ✅ **Flexible model selection** and configuration
- ✅ **Comprehensive logging** and feedback
